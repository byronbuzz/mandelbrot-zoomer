import {
  fixedAddScaled,
  fixedSplitF32,
  fixedSub,
  fixedToNumber
} from '../bigFixed';
import type { CameraSnapshot } from '../camera/types';
import { AcceptedTileAtlas, type AtlasSlot } from './acceptedTileAtlas';
import { AtlasHistoryPresenter, type AtlasInstance } from './atlasHistoryPresenter';
import {
  fixedDifferenceOverDyadic,
  packTransform,
  scaleOverDyadic,
  transformIsFinite
} from './presentationMath';
import {
  tileAtlasPublishShader,
  tileClearShader,
  tileColourShader,
  tileDirectIterationShader,
  tilePerturbationShader,
  tilePresentShader,
  tileResetNumericalShader
} from '../numerical/tileFieldShadersV13';
import {
  TileReferenceAtlas,
  type TileGpuReference
} from '../references/tileReferenceAtlasV13';
import {
  PERSISTENT_TILE_SIZE,
  type PersistentFieldStats,
  type PersistentTileDescriptor,
  type PersistentTileHealth,
  type PersistentTileKey,
  type PersistentTileRequest,
  type PersistentTileWork,
  type TileNumericalMode,
  type TileReferenceState
} from '../tiles/persistentTileTypes';
import {
  sampleExponentForViewport,
  tileSpanExponent,
  visibleTileDescriptors
} from '../tiles/worldTilePlanner';

const DIRECT_PARAMETER_BYTES = 64;
const PERTURB_PARAMETER_BYTES = 96;
const COLOUR_PARAMETER_BYTES = 32;
const PRESENT_PARAMETER_BYTES = 16;
const CLEAR_PARAMETER_BYTES = 16;
const RESET_PARAMETER_BYTES = 16;
const COUNTER_VALUES = 7;
const COUNTER_BYTES = COUNTER_VALUES * Uint32Array.BYTES_PER_ELEMENT;
const COUNTER_READBACK_STRIDE = 32;
const STATE_BYTES_PER_PIXEL = 16;
const META_BYTES_PER_PIXEL = 16;
const TILE_PIXEL_COUNT = PERSISTENT_TILE_SIZE * PERSISTENT_TILE_SIZE;
const INITIAL_BATCH_TILES = 8;
const MIN_BATCH_TILES = 4;
const MAX_BATCH_TILES = 24;
const RESOURCE_CREATION_BATCH_TILES = 24;
const MAX_CACHED_TILES = 224;
const CACHE_HISTORY_TILE_RESERVE = 96;
const MAX_NUMERICAL_PIXELS = 2_500_000;
const DIRECT_SAFETY_ITERATIONS = 256;
const MAX_REFERENCE_REPAIR_PASSES = 5;
const GLITCH_RATIO = 1e-6;
const PALETTE_LENGTH = 64;
const MOVING_QUANTUM_MS = 11;
const SETTLING_QUANTUM_MS = 22;
const SETTLED_QUANTUM_MS = 40;
const MOVING_BATCH_TARGET_MS = 7;
const SETTLING_BATCH_TARGET_MS = 11;
const SETTLED_BATCH_TARGET_MS = 16;
const MAX_IN_FLIGHT_BATCHES = 3;

const EMPTY_HEALTH: PersistentTileHealth = {
  activePixels: TILE_PIXEL_COUNT,
  escapedPixels: 0,
  analyticInteriorPixels: 0,
  cappedPixels: 0,
  nonFinitePixels: 0,
  glitchPixels: 0,
  orbitExhaustedPixels: 0
};

const EMPTY_STATS: PersistentFieldStats = {
  requestId: 0,
  interaction: 'settled',
  visibleTiles: 0,
  cachedTiles: 0,
  activeTiles: 0,
  convergedTiles: 0,
  directTiles: 0,
  perturbationTiles: 0,
  pendingReferences: 0,
  repairTiles: 0,
  referenceFailures: 0,
  completedChunks: 0,
  submittedChunks: 0,
  queuedChunks: 0,
  inFlightBatches: 0,
  inFlightTiles: 0,
  atlasPublications: 0,
  avoidedAtlasCopies: 0,
  lastBatchMs: 0,
  numericalFreshnessMs: 0,
  presentationHistoryMs: 0,
  sampleExponent: 0,
  tileSize: PERSISTENT_TILE_SIZE
};

type ResetMode = 'all' | 'unresolved';
type CapPresentationMode = 0 | 1 | 2;

type PlannedDescriptor = Readonly<{
  descriptor: PersistentTileDescriptor;
  levelOffset: number;
}>;

type FieldTile = {
  descriptor: PersistentTileDescriptor;
  stateBuffer: GPUBuffer;
  metaBuffer: GPUBuffer;
  counterBuffer: GPUBuffer;
  resultTexture: GPUTexture;
  qualityTexture: GPUTexture;
  colourTexture: GPUTexture;
  evidenceTexture: GPUTexture;
  directUniform: GPUBuffer;
  perturbUniform: GPUBuffer;
  colourUniform: GPUBuffer;
  presentUniform: GPUBuffer;
  resetUniform: GPUBuffer;
  directGroup: GPUBindGroup;
  perturbGroup: GPUBindGroup | null;
  colourGroup: GPUBindGroup;
  clearGroup: GPUBindGroup;
  resetGroup: GPUBindGroup;
  presentLinearGroup: GPUBindGroup;
  presentNearestGroup: GPUBindGroup;
  atlasSlot: AtlasSlot | null;
  health: PersistentTileHealth;
  iterationFrontier: number;
  coveragePixels: number;
  resolvedPixels: number;
  capPresentationMode: CapPresentationMode;
  lastVisibleAt: number;
  lastNumericalUpdateAt: number;
  createdAt: number;
  palettePhase: number;
  directMode: 0 | 1;
  requiresPerturbation: boolean;
  numericalMode: TileNumericalMode;
  referenceState: TileReferenceState;
  reference: TileGpuReference | null;
  pendingReference: TileGpuReference | null;
  pendingReset: ResetMode | null;
  referenceError: string | null;
  repairPass: number;
  referenceTarget: number;
};

type ScheduledWork = Readonly<{
  work: PersistentTileWork;
  tile: FieldTile;
  scheduledEnd: number;
  finalTarget: number;
  levelOffset: number;
  capMode: CapPresentationMode;
}>;

type PendingBatch = Readonly<{
  batch: readonly ScheduledWork[];
  request: PersistentTileRequest;
  submittedAt: number;
  slot: BatchReadbackSlot;
  completed: Promise<BatchCompletion>;
}>;

type BatchReadbackSlot = {
  buffer: GPUBuffer;
  busy: boolean;
};

type BatchCompletion = Readonly<{
  completedAt: number;
  health: ReadonlyArray<Readonly<{
    scheduled: ScheduledWork;
    value: PersistentTileHealth;
  }>>;
}>;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function schedulerYield(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

function splitChanged(
  before: readonly [number, number],
  after: readonly [number, number]
): boolean {
  return before[0] !== after[0] || before[1] !== after[1];
}

export class TileFieldRenderer {
  private readonly tileMap = new Map<PersistentTileKey, FieldTile>();
  private readonly directPipeline: GPUComputePipeline;
  private readonly perturbPipeline: GPUComputePipeline;
  private readonly colourPipeline: GPUComputePipeline;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly resetPipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly linearSampler: GPUSampler;
  private readonly nearestSampler: GPUSampler;
  private readonly clearUniform: GPUBuffer;
  private readonly referenceAtlas: TileReferenceAtlas;
  private readonly acceptedAtlas: AcceptedTileAtlas | null;
  private readonly atlasPresenter: AtlasHistoryPresenter | null;
  private readonly useLegacyPresenter: boolean;
  private latestRequest: PersistentTileRequest | null = null;
  private currentRequest: PersistentTileRequest | null = null;
  private currentPlan: PlannedDescriptor[] = [];
  private currentQueue: ScheduledWork[] = [];
  private currentVisibleKeys = new Set<PersistentTileKey>();
  private pendingLevelOffsets: number[] = [];
  private currentRenderHeight = 1;
  private currentAspect = 1;
  private maximumPlannedLevel = 2;
  private running = false;
  private suspended = false;
  private testSchedulerPaused = false;
  private testBatchPermits = 0;
  private testBatchRevision = 0;
  private testBatchExecuting = false;
  private readonly testRequestBatchCounts = new Map<number, number>();
  private testGateWaiters: Array<() => void> = [];
  private testPauseWaiters: Array<() => void> = [];
  private testBatchWaiters: Array<{
    requestId: number;
    targetCount: number;
    resolve: (result: { batchRevision: number; requestId: number; requestBatchCount: number }) => void;
    reject: (error: Error) => void;
  }> = [];
  private displayWidth = 1;
  private displayHeight = 1;
  private statsValue: PersistentFieldStats = EMPTY_STATS;
  private runtimeErrorListener: ((message: string) => void) | null = null;
  private adaptiveBatchTiles = INITIAL_BATCH_TILES;
  private completedChunks = 0;
  private lastBatchMs = 0;
  private lastBatchCompletedAt = 0;
  private readonly pendingBatches: PendingBatch[] = [];
  private readonly readbackSlots: BatchReadbackSlot[];
  private submittedChunks = 0;
  private atlasPublications = 0;
  private avoidedAtlasCopies = 0;
  private dead = false;
  private deviceLostListener: ((message: string) => void) | null = null;
  private readonly deviceErrors: string[] = [];

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly canvasFormat: GPUTextureFormat,
    directPipeline: GPUComputePipeline,
    perturbPipeline: GPUComputePipeline,
    colourPipeline: GPUComputePipeline,
    clearPipeline: GPUComputePipeline,
    resetPipeline: GPUComputePipeline,
    presentPipeline: GPURenderPipeline,
    acceptedAtlas: AcceptedTileAtlas | null,
    atlasPresenter: AtlasHistoryPresenter | null,
    useLegacyPresenter: boolean,
    readonly adapterLabel: string
  ) {
    this.directPipeline = directPipeline;
    this.perturbPipeline = perturbPipeline;
    this.colourPipeline = colourPipeline;
    this.clearPipeline = clearPipeline;
    this.resetPipeline = resetPipeline;
    this.presentPipeline = presentPipeline;
    this.acceptedAtlas = acceptedAtlas;
    this.atlasPresenter = atlasPresenter;
    this.useLegacyPresenter = useLegacyPresenter;
    this.clearUniform = device.createBuffer({
      size: CLEAR_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(
      this.clearUniform,
      0,
      new Uint32Array([PERSISTENT_TILE_SIZE, 0, 0, 0])
    );
    this.linearSampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
    this.nearestSampler = device.createSampler({
      minFilter: 'nearest',
      magFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
    this.referenceAtlas = new TileReferenceAtlas(device);
    this.readbackSlots = Array.from({ length: MAX_IN_FLIGHT_BATCHES }, (_, index) => ({
      buffer: device.createBuffer({
        label: `tile-counter-readback-ring-${index}`,
        size: MAX_BATCH_TILES * COUNTER_READBACK_STRIDE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      }),
      busy: false
    }));
  }

  static async create(canvas: HTMLCanvasElement): Promise<TileFieldRenderer> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is unavailable. Use a current browser with hardware acceleration enabled.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) throw new Error('Unable to create a WebGPU canvas context');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
    const useLegacyPresenter = new URLSearchParams(location.search).get('presenter') === 'legacy';

    const directModule = device.createShaderModule({ code: tileDirectIterationShader });
    const perturbModule = device.createShaderModule({ code: tilePerturbationShader });
    const colourModule = device.createShaderModule({
      code: useLegacyPresenter ? tileColourShader : tileAtlasPublishShader
    });
    const clearModule = device.createShaderModule({ code: tileClearShader });
    const resetModule = device.createShaderModule({ code: tileResetNumericalShader });
    const presentModule = device.createShaderModule({ code: tilePresentShader });
    await Promise.all([
      this.assertShaderValid(directModule, 'tile direct iteration'),
      this.assertShaderValid(perturbModule, 'tile perturbation'),
      this.assertShaderValid(colourModule, useLegacyPresenter ? 'tile colour' : 'tile atlas publication'),
      this.assertShaderValid(clearModule, 'tile clear'),
      this.assertShaderValid(resetModule, 'tile numerical reset'),
      this.assertShaderValid(presentModule, 'tile presentation')
    ]);

    const [directPipeline, perturbPipeline, colourPipeline, clearPipeline, resetPipeline, presentPipeline]
      = await Promise.all([
        device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module: directModule, entryPoint: 'main' }
        }),
        device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module: perturbModule, entryPoint: 'main' }
        }),
        device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module: colourModule, entryPoint: 'main' }
        }),
        device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module: clearModule, entryPoint: 'main' }
        }),
        device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module: resetModule, entryPoint: 'main' }
        }),
        device.createRenderPipelineAsync({
          layout: 'auto',
          vertex: { module: presentModule, entryPoint: 'vertexMain' },
          fragment: {
            module: presentModule,
            entryPoint: 'fragmentMain',
            targets: [{
              format: canvasFormat,
              blend: {
                color: {
                  srcFactor: 'src-alpha',
                  dstFactor: 'one-minus-src-alpha',
                  operation: 'add'
                },
                alpha: {
                  srcFactor: 'one',
                  dstFactor: 'one-minus-src-alpha',
                  operation: 'add'
                }
              }
            }]
          },
          primitive: { topology: 'triangle-list' }
        })
      ]);

    const label = adapter.info.vendor || adapter.info.description || 'GPU';
    const acceptedAtlas = useLegacyPresenter ? null : new AcceptedTileAtlas(device);
    const atlasPresenter = useLegacyPresenter
      ? null
      : await AtlasHistoryPresenter.create(device, context, canvasFormat);
    return new TileFieldRenderer(
      canvas,
      device,
      context,
      canvasFormat,
      directPipeline,
      perturbPipeline,
      colourPipeline,
      clearPipeline,
      resetPipeline,
      presentPipeline,
      acceptedAtlas,
      atlasPresenter,
      useLegacyPresenter,
      label
    );
  }

  onRuntimeError(listener: (message: string) => void): void {
    this.runtimeErrorListener = listener;
  }

  onDeviceError(listener: (message: string) => void): void {
    this.device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
      this.deviceErrors.push(event.error.message);
      if (this.deviceErrors.length > 32) this.deviceErrors.shift();
      listener(`WebGPU error: ${event.error.message}`);
    });
  }

  onDeviceLost(listener: (message: string) => void): void {
    this.deviceLostListener = listener;
    void this.device.lost.then((reason: { message?: string; reason?: string }) => {
      if (this.dead) return;
      listener(reason.message || reason.reason || 'Unknown device loss');
    });
  }

  request(request: PersistentTileRequest): void {
    if (this.dead || this.suspended) return;
    this.la×nùÚÚ$z{-®éÜj×ÇfVE—†VÇ3¢À¢6&W6VçFF–öäÖöFS¢À¢Æ7Ef—6–&ÆTC¢æ÷rÀ¢Æ7DçVÖW&–6ÅWFFTC¢À¢7&VFVDC¢æ÷rÀ¢ÆWGFU†6S¢çVÖ&W"äæâÀ¢F—&V7DÖöFRÀ¢&WV—&W5W'GW&&F–öâÀ¢çVÖW&–6ÄÖöFS¢F—&V7DÖöFRÓÓÒòvc3"ÖF—&V7Br¢vF÷V&ÆRÖfÆöBÖF—&V7BrÀ¢&VfW&Væ6U7FFS¢væöæRrÀ¢&VfW&Væ6S¢çVÆÂÀ¢VæF–æu&VfW&Væ6S¢çVÆÂÀ¢VæF–æu&W6WC¢çVÆÂÀ¢&VfW&Væ6TW'&÷#¢çVÆÂÀ¢&W—%73¢À¢&VfW&Væ6UF&vWC¢ ¢Ó°¢Ğ ¢&—fFR7&VFU&W6VçDw&÷W€¢Væ–f÷&Ó¢uT'VffW"À¢6öÆ÷W%FW‡GW&S¢uUFW‡GW&RÀ¢VÆ—G•FW‡GW&S¢uUFW‡GW&RÀ¢6×ÆW#¢uU6×ÆW ¢“¢uT&–æDw&÷W°¢&WGW&âF†—2æFWf–6Ræ7&VFT&–æDw&÷W‡°¢Æ–÷WC¢F†—2ç&W6VçE—VÆ–æRævWD&–æDw&÷WÆ–÷WBƒ’À¢VçG&–W3¢°¢²&–æF–æs¢Â&W6÷W&6S¢²'VffW#¢Væ–f÷&ÒÒÒÀ¢²&–æF–æs¢Â&W6÷W&6S¢6×ÆW"ÒÀ¢²&–æF–æs¢"Â&W6÷W&6S¢6öÆ÷W%FW‡GW&Ræ7&VFUf–Wr‚’ÒÀ¢²&–æF–æs¢2Â&W6÷W&6S¢VÆ—G•FW‡GW&Ræ7&VFUf–Wr‚’Ğ¢Ğ¢Ò“°¢Ğ ¢&—fFR7&VFUW'GW&$w&÷W‡F–ÆS¢f–VÆEF–ÆRÂ&VfW&Væ6S¢F–ÆTwU&VfW&Væ6R“¢uT&–æDw&÷W°¢&WGW&âF†—2æFWf–6Ræ7&VFT&–æDw&÷W‡°¢Æ–÷WC¢F†—2çW'GW&%—VÆ–æRævWD&–æDw&÷WÆ–÷WBƒ’À¢VçG&–W3¢°¢²&–æF–æs¢Â&W6÷W&6S¢²'VffW#¢F–ÆRçW'GW&%Væ–f÷&ÒÒÒÀ¢²&–æF–æs¢Â&W6÷W&6S¢²'VffW#¢F–ÆRç7FFT'VffW"ÒÒÀ¢²&–æF–æs¢"Â&W6÷W&6S¢²'VffW#¢F–ÆRæÖWF'VffW"ÒÒÀ¢²&–æF–æs¢2Â&W6÷W&6S¢F–ÆRç&W7VÇEFW‡GW&Ræ7&VFUf–Wr‚’ÒÀ¢²&–æF–æs¢BÂ&W6÷W&6S¢F–ÆRçVÆ—G•FW‡GW&Ræ7&VFUf–Wr‚’ÒÀ¢²&–æF–æs¢RÂ&W6÷W&6S¢²'VffW#¢F–ÆRæ6÷VçFW$'VffW"ÒÒÀ¢²&–æF–æs¢bÂ&W6÷W&6S¢²'VffW#¢&VfW&Væ6Ræ'VffW"ÒĞ¢Ğ¢Ò“°¢Ğ ¢&—fFR7V&Ö—D&F6‚€¢&F6ƒ¢&VFöæÇ’66†VGVÆVEv÷&µµÒÀ¢&WVW7C¢W'6—7FVçEF–ÆU&WVW7@¢“¢VæF–æt&F6‚°¢6öç7B¶W—2ÒæWr6WB†&F6‚æÖ‡66†VGVÆVBÓâ66†VGVÆVBçF–ÆRæFW67&—F÷"æ¶W’’“°¢–b†¶W—2ç6—¦RÓÒ&F6‚æÆVæwF‚’F‡&÷ræWrW'&÷"‚tF–ÆRv266†VGVÆVBGv–6R–âöæRuR&F6‚r“°¢f÷"†6öç7BVæF–æröbF†—2çVæF–æt&F6†W2’°¢–b‡VæF–æræ&F6‚ç6öÖR‡66†VGVÆVBÓâ¶W—2æ†2‡66†VGVÆVBçF–ÆRæFW67&—F÷"æ¶W’’’’°¢F‡&÷ræWrW'&÷"‚tF–ÆRÇ&VG’†2â–âÖfÆ–v‡BuR×WFF–öâr“°¢Ğ¢Ğ¢6öç7B6Æ÷BÒF†—2ç&VF&6µ6Æ÷G2æf–æB†6æF–FFRÓâ6æF–FFRæ'W7’“°¢–b‚6Æ÷B’F‡&÷ræWrW'&÷"‚t6÷VçFW"&VF&6²&–ærW††W7FVBr“°¢6Æ÷Bæ'W7’ÒG'VS°¢f÷"†6öç7B66†VGVÆVBöb&F6‚’°¢6öç7BF–ÆRÒ66†VGVÆVBçF–ÆS°¢–b‡F–ÆRæçVÖW&–6ÄÖöFRÓÓÒwW'GW&&F–öâr’°¢F†—2æFWf–6RçVWVRçw&—FT'VffW"€¢F–ÆRçW'GW&%Væ–f÷&ÒÀ¢À¢F†—2æ7&VFUW'GW&%&ÖWFW$FF‡66†VGVÆVB¢“°¢ÒVÇ6R°¢F†—2æFWf–6RçVWVRçw&—FT'VffW"€¢F–ÆRæF—&V7EVæ–f÷&ÒÀ¢À¢F†—2æ7&VFTF—&V7E&ÖWFW$FF‡66†VGVÆVB¢“°¢Ğ¢F†—2æFWf–6RçVWVRçw&—FT'VffW"€¢F–ÆRæ6öÆ÷W%Væ–f÷&ÒÀ¢À¢F†—2æ7&VFT6öÆ÷W%&ÖWFW$FF‡66†VGVÆVBÂ&WVW7BçÆWGFU†6R¢“°¢Ğ ¢6öç7BVæ6öFW"ÒF†—2æFWf–6Ræ7&VFT6öÖÖæDVæ6öFW"‚“°¢Væ6öFW"æ6ÆV$'VffW"‡6Æ÷Bæ'VffW"“°¢f÷"†ÆWB&F6„–æFW‚Ò²&F6„–æFW‚Â&F6‚æÆVæwFƒ²&F6„–æFW‚²²’°¢6öç7B66†VGVÆVBÒ&F6…¶&F6„–æFW…Ó°¢6öç7BF–ÆRÒ66†VGVÆVBçF–ÆS°¢–b‡F†—2ç&WV—&W4çVÖW&–6ÄF—7F6‚‡66†VGVÆVB’’°¢Væ6öFW"æ6ÆV$'VffW"‡F–ÆRæ6÷VçFW$'VffW"“°¢6öç7B—FW&F–öå72ÒVæ6öFW"æ&Vv–ä6ö×WFU72‚“°¢–b‡F–ÆRæçVÖW&–6ÄÖöFRÓÓÒwW'GW&&F–öâr’°¢–b‚F–ÆRçW'GW&$w&÷W’F‡&÷ræWrW'&÷"†Ö—76–ærW'GW&&F–öâ&–æBw&÷Wf÷"G·F–ÆRæFW67&—F÷"æ¶W—Ö“°¢—FW&F–öå72ç6WE—VÆ–æR‡F†—2çW'GW&%—VÆ–æR“°¢—FW&F–öå72ç6WD&–æDw&÷WƒÂF–ÆRçW'GW&$w&÷W“°¢ÒVÇ6R°¢—FW&F–öå72ç6WE—VÆ–æR‡F†—2æF—&V7E—VÆ–æR“°¢—FW&F–öå72ç6WD&–æDw&÷WƒÂF–ÆRæF—&V7Dw&÷W“°¢Ğ¢—FW&F–öå72æF—7F6…v÷&¶w&÷W2€¢ÖF‚æ6V–Â…U%4•5DTåEõD”ÄUõ4•¤Rò‚’À¢ÖF‚æ6V–Â…U%4•5DTåEõD”ÄUõ4•¤Rò‚¢“°¢—FW&F–öå72æVæB‚“°¢Ğ ¢6öç7B6öÆ÷W%72ÒVæ6öFW"æ&Vv–ä6ö×WFU72‚“°¢6öÆ÷W%72ç6WE—VÆ–æR‡F†—2æ6öÆ÷W%—VÆ–æR“°¢6öÆ÷W%72ç6WD&–æDw&÷WƒÂF–ÆRæ6öÆ÷W$w&÷W“°¢6öÆ÷W%72æF—7F6…v÷&¶w&÷W2€¢ÖF‚æ6V–Â…U%4•5DTåEõD”ÄUõ4•¤Rò‚’À¢ÖF‚æ6V–Â…U%4•5DTåEõD”ÄUõ4•¤Rò‚¢“°¢6öÆ÷W%72æVæB‚“°¢–b‡F†—2ç&WV—&W4çVÖW&–6ÄF—7F6‚‡66†VGVÆVB’’°¢Væ6öFW"æ6÷”'VffW%Fô'VffW"€¢F–ÆRæ6÷VçFW$'VffW"À¢À¢6Æ÷Bæ'VffW"À¢&F6„–æFW‚¢4õTåDU%õ$TD$4µõ5E$”DRÀ¢4õTåDU%ô%•DU0¢“°¢Ğ¢Ğ¢6öç7B7V&Ö—GFVDBÒW&f÷&Öæ6Rææ÷r‚“°¢F†—2æFWf–6RçVWVRç7V&Ö—B…¶Væ6öFW"æf–æ—6‚‚•Ò“°¢6öç7B6ö×ÆWFVBÒ6Æ÷Bæ'VffW"æÖ7–æ2„uTÖÖöFRå$TB’çF†Vâ‚‚’Óâ°¢6öç7BÖVBÒæWrV–çC3$'&’‡6Æ÷Bæ'VffW"ævWDÖVE&ævR‚’“°¢6öç7B†VÇF‚Ò&F6‚æfÆDÖ‚‡66†VGVÆVBÂ&F6„–æFW‚’Óâ°¢–b‚F†—2ç&WV—&W4çVÖW&–6ÄF—7F6‚‡66†VGVÆVB’’&WGW&âµÓ°¢6öç7Böfg6WBÒ&F6„–æFW‚¢„4õTåDU%õ$TD$4µõ5E$”DRòV–çC3$'&’ä%•DU5õU%ôTÄTÔTåB“°¢&WGW&â·°¢66†VGVÆVBÀ¢fÇVS¢°¢7F—fU—†VÇ3¢ÖVE¶öfg6WEÒóòÀ¢W66VE—†VÇ3¢ÖVE¶öfg6WB²ÒóòÀ¢æÇ—F–4–çFW&–÷%—†VÇ3¢ÖVE¶öfg6WB²%ÒóòÀ¢6VE—†VÇ3¢ÖVE¶öfg6WB²5ÒóòÀ¢æöäf–æ—FU—†VÇ3¢ÖVE¶öfg6WB²EÒóòÀ¢vÆ—F6…—†VÇ3¢ÖVE¶öfg6WB²UÒóòÀ¢÷&&—DW††W7FVE—†VÇ3¢ÖVE¶öfg6WB²eÒóò ¢Ğ¢ÕÓ°¢Ò“°¢6Æ÷Bæ'VffW"çVæÖ‚“°¢&WGW&â²6ö×ÆWFVDC¢W&f÷&Öæ6Rææ÷r‚’Â†VÇF‚Ó°¢Ò“°¢F†—2ç7V&Ö—GFVD6‡Væ·2³Ò&F6‚æÆVæwFƒ°¢F†—2æFÆ5V&Æ–6F–öç2³ÒF†—2æ66WFVDFÆ2ò&F6‚æÆVæwF‚¢°¢F†—2æfö–FVDFÆ46÷–W2³ÒF†—2æ66WFVDFÆ2ò&F6‚æÆVæwF‚¢2¢°¢&WGW&â²&F6‚Â&WVW7BÂ7V&Ö—GFVDBÂ6Æ÷BÂ6ö×ÆWFVBÓ°¢Ğ ¢&—fFR&WV—&W4çVÖW&–6ÄF—7F6‚‡66†VGVÆVC¢66†VGVÆVEv÷&²“¢&ööÆVâ°¢&WGW&â66†VGVÆVBçv÷&²æ6‡Væ´—FW&F–öç2â ¢ÇÂ66†VGVÆVBæ6ÖöFRâ66†VGVÆVBçF–ÆRæ6&W6VçFF–öäÖöFS°¢Ğ ¢&—fFR7&VFTF—&V7E&ÖWFW$FF‡66†VGVÆVC¢66†VGVÆVEv÷&²“¢'&”'VffW"°¢6öç7BFFÒæWr'&”'VffW"„D•$T5Eõ$ÔUDU%ô%•DU2“°¢6öç7BfÆöG2ÒæWrfÆöC3$'&’†FF“°¢6öç7BVç6–væVBÒæWrV–çC3$'&’†FF“°¢6öç7B6–væVBÒæWr–çC3$'&’†FF“°¢6öç7BF–ÆRÒ66†VGVÆVBçF–ÆS°¢6öç7B¶6VçFW%„†’Â6VçFW%„ÆõÒÒf—†VE7Æ—Dc3"‡F–ÆRæFW67&—F÷"æ6VçFW%‚“°¢6öç7B¶6VçFW%”†’Â6VçFW%”ÆõÒÒf—†VE7Æ—Dc3"‡F–ÆRæFW67&—F÷"æ6VçFW%’“°¢fÆöG5³ÒÒ6VçFW%„†“°¢fÆöG5³ÒÒ6VçFW%„Æó°¢fÆöG5³%ÒÒ6VçFW%”†“°¢fÆöG5³5ÒÒ6VçFW%”Æó°¢6–væVE³EÒÒF–ÆRæFW67&—F÷"ç6×ÆTW‡öæVçC°¢Vç6–væVE³UÒÒU%4•5DTåEõD”ÄUõ4•¤S°¢Vç6–væVE³eÒÒ66†VGVÆVBç66†VGVÆVDVæC°¢Vç6–væVE³uÒÒ66†VGVÆVBçv÷&²æ6‡Væ´—FW&F–öç3°¢Vç6–væVE³…ÒÒF–ÆRæF—&V7DÖöFS°¢Vç6–væVE³•ÒÒ66†VGVÆVBæ6ÖöFS°¢&WGW&âFF°¢Ğ ¢&—fFR7&VFUW'GW&%&ÖWFW$FF‡66†VGVÆVC¢66†VGVÆVEv÷&²“¢'&”'VffW"°¢6öç7BF–ÆRÒ66†VGVÆVBçF–ÆS°¢6öç7B&VfW&Væ6RÒF–ÆRç&VfW&Væ6S°¢–b‚&VfW&Væ6R’F‡&÷ræWrW'&÷"†Ö—76–ærF–ÆR&VfW&Væ6Rf÷"G·F–ÆRæFW67&—F÷"æ¶W—Ö“°¢6öç7BFFÒæWr'&”'VffW"…U%EU$%õ$ÔUDU%ô%•DU2“°¢6öç7BfÆöG2ÒæWrfÆöC3$'&’†FF“°¢6öç7BVç6–væVBÒæWrV–çC3$'&’†FF“°¢6öç7B6–væVBÒæWr–çC3$'&’†FF“°¢6öç7B¶6VçFW%„†’Â6VçFW%„ÆõÒÒf—†VE7Æ—Dc3"‡F–ÆRæFW67&—F÷"æ6VçFW%‚“°¢6öç7B¶6VçFW%”†’Â6VçFW%”ÆõÒÒf—†VE7Æ—Dc3"‡F–ÆRæFW67&—F÷"æ6VçFW%’“°¢6öç7B¶FVÇF„†’ÂFVÇF„ÆõÒÒf—†VE7Æ—Dc3"†f—†VE7V"‡F–ÆRæFW67&—F÷"æ6VçFW%‚Â&VfW&Væ6Ræ6VçFW%‚’“°¢6öç7B¶FVÇF”†’ÂFVÇF”ÆõÒÒf—†VE7Æ—Dc3"†f—†VE7V"‡F–ÆRæFW67&—F÷"æ6VçFW%’Â&VfW&Væ6Ræ6VçFW%’’“°¢fÆöG5³ÒÒ6VçFW%„†“°¢fÆöG5³ÒÒ6VçFW%„Æó°¢fÆöG5³%ÒÒ6VçFW%”†“°¢fÆöG5³5ÒÒ6VçFW%”Æó°¢fÆöG5³EÒÒFVÇF„†“°¢fÆöG5³UÒÒFVÇF„Æó°¢fÆöG5³eÒÒFVÇF”†“°¢fÆöG5³uÒÒFVÇF”Æó°¢6–væVE³…ÒÒF–ÆRæFW67&—F÷"ç6×ÆTW‡öæVçC°¢Vç6–væVE³•ÒÒU%4•5DTåEõD”ÄUõ4•¤S°¢Vç6–væVE³ÒÒ66†VGVÆVBç66†VGVÆVDVæC°¢Vç6–væVE³ÒÒ66†VGVÆVBçv÷&²æ6‡Væ´—FW&F–öç3°¢Vç6–væVE³%ÒÒ&VfW&Væ6RæÆVæwFƒ°¢Vç6–væVE³5ÒÒ66†VGVÆVBæ6ÖöFS°¢Vç6–væVE³EÒÒ&VfW&Væ6Ræ&—G3°¢Vç6–væVE³UÒÒF–ÆRç&W—%73°¢fÆöG5³eÒÒtÄ•D4…õ$D”ó°¢&WGW&âFF°¢Ğ ¢&—fFR7&VFT6öÆ÷W%&ÖWFW$FF€¢66†VGVÆVC¢66†VGVÆVEv÷&²À¢ÆWGFU†6S¢çVÖ&W ¢“¢'&”'VffW"°¢6öç7BFFÒæWr'&”'VffW"„4ôÄõU%õ$ÔUDU%ô%•DU2“°¢6öç7BVç6–væVBÒæWrV–çC3$'&’†FF“°¢6öç7BfÆöG2ÒæWrfÆöC3$'&’†FF“°¢Vç6–væVE³ÒÒU%4•5DTåEõD”ÄUõ4•¤S°¢–b‡F†—2æ66WFVDFÆ2bb66†VGVÆVBçF–ÆRæFÆ56Æ÷B’°¢Vç6–væVE³ÒÒÖF‚æ'2‡66†VGVÆVBçF–ÆRçÆWGFU†6RÒÆWGFU†6R’âRÓbò¢°¢Vç6–væVE³%ÒÒ66†VGVÆVBçF–ÆRæ—FW&F–öäg&öçF–W#°¢Vç6–væVE³5ÒÒ66†VGVÆVBæ6ÖöFRâ66†VGVÆVBçF–ÆRæ6&W6VçFF–öäÖöFRò¢°¢Vç6–væVE³EÒÒ66†VGVÆVBçF–ÆRæFÆ56Æ÷Bçƒ°¢Vç6–væVE³UÒÒ66†VGVÆVBçF–ÆRæFÆ56Æ÷Bç“°¢fÆöG5³eÒÒÆWGFU†6S°¢fÆöG5³uÒÒÄUEDUôÄTäuDƒ°¢ÒVÇ6R°¢fÆöG5³%ÒÒÆWGFU†6S°¢fÆöG5³5ÒÒÄUEDUôÄTäuDƒ°¢Ğ¢&WGW&âFF°¢Ğ ¢&—fFR&VæFW$†V–v‡B€¢775v–GFƒ¢çVÖ&W"À¢774†V–v‡C¢çVÖ&W"À¢FWf–6U—†VÅ&F–ó¢çVÖ&W ¢“¢çVÖ&W"°¢6öç7BG"Ò6Æ×†FWf–6U—†VÅ&F–òÂÂ"“°¢6öç7B&WVW7FVEv–GF‚ÒÖF‚æÖ‚ƒÂ775v–GF‚¢G"“°¢6öç7B&WVW7FVD†V–v‡BÒÖF‚æÖ‚ƒÂ774†V–v‡B¢G"“°¢6öç7B66ÆRÒÖF‚æÖ–âƒÂÖF‚ç7'B„Ô…ôåTÔU$”4Åõ•„TÅ2ò‡&WVW7FVEv–GF‚¢&WVW7FVD†V–v‡B’’“°¢&WGW&âÖF‚æÖ‚ƒÂÖF‚æfÆö÷"‡&WVW7FVD†V–v‡B¢66ÆR’“°¢Ğ ¢&—fFR7–æ2v—EFW7D&F6…W&Ö—B‚“¢&öÖ—6SÆ&ööÆVãâ°¢v†–ÆR‡F†—2çFW7E66†VGVÆW%W6V@¢bbF†—2çFW7D&F6…W&Ö—G2ÃÒ ¢bbF†—2æFV@¢bbF†—2ç7W7VæFV@¢bbF†—2æÆFW7E&WVW7B’°¢v—BæWr&öÖ—6SÇfö–Câ‡&W6öÇfRÓâF†—2çFW7DvFUv—FW'2çW6‚‡&W6öÇfR’“°¢Ğ¢–b‡F†—2çFW7E66†VGVÆW%W6VBbbF†—2çFW7D&F6…W&Ö—G2â’°¢F†—2çFW7D&F6…W&Ö—G2ÒÓ°¢&WGW&âG'VS°¢Ğ¢&WGW&âfÇ6S°¢Ğ ¢&—fFRæ÷FUFW7D&F6„6ö×ÆWFVB‡&WVW7D–C¢çVÖ&W"“¢fö–B°¢F†—2çFW7D&F6…&Wf—6–öâ²³°¢6öç7B&WVW7D&F6„6÷VçBÒ‡F†—2çFW7E&WVW7D&F6„6÷VçG2ævWB‡&WVW7D–B’óò’²°¢F†—2çFW7E&WVW7D&F6„6÷VçG2ç6WB‡&WVW7D–BÂ&WVW7D&F6„6÷VçB“°¢6öç7B&VG’ÒF†—2çFW7D&F6…v—FW'2æf–ÇFW"€¢v—FW"Óâv—FW"ç&WVW7D–BÓÓÒ&WVW7D–Bbbv—FW"çF&vWD6÷VçBÃÒ&WVW7D&F6„6÷Vç@¢“°¢F†—2çFW7D&F6…v—FW'2ÒF†—2çFW7D&F6…v—FW'2æf–ÇFW"‡v—FW"Óâ&VG’æ–æ6ÇVFW2‡v—FW"’“°¢f÷"†6öç7Bv—FW"öb&VG’’v—FW"ç&W6öÇfR‡°¢&F6…&Wf—6–öã¢F†—2çFW7D&F6…&Wf—6–öâÀ¢&WVW7D–BÀ¢&WVW7D&F6„6÷Vç@¢Ò“°¢Ğ ¢&—fFRVçGVÔ'VFvWB†–çFW&7F–öã¢W'6—7FVçEF–ÆU&WVW7E²v–çFW&7F–öâuÒ“¢çVÖ&W"°¢&WGW&â–çFW&7F–öâÓÓÒvÖ÷f–ærp¢òÔõd”äuõTåETÕôÕ0¢¢–çFW&7F–öâÓÓÒw6WGFÆ–ærrò4UEDÄ”äuõTåETÕôÕ2¢4UEDÄTEõTåETÕôÕ3°¢Ğ ¢&—fFR&F6…F&vWB†–çFW&7F–öã¢W'6—7FVçEF–ÆU&WVW7E²v–çFW&7F–öâuÒ“¢çVÖ&W"°¢&WGW&â–çFW&7F–öâÓÓÒvÖ÷f–ærp¢òÔõd”äuô$D4…õD$tUEôÕ0¢¢–çFW&7F–öâÓÓÒw6WGFÆ–ærrò4UEDÄ”äuô$D4…õD$tUEôÕ2¢4UEDÄTEô$D4…õD$tUEôÕ3°¢Ğ ¢&—fFRFD&F6…6—¦R€¢–çFW&7F–öã¢W'6—7FVçEF–ÆU&WVW7E²v–çFW&7F–öâuÒÀ¢VÆ6VD×3¢çVÖ&W ¢“¢fö–B°¢6öç7BF&vWBÒF†—2æ&F6…F&vWB†–çFW&7F–öâ“°¢6öç7B&F–òÒ6Æ×‡F&vWBòÖF‚æÖ‚ƒã#RÂVÆ6VD×2’ÂãbÂãR“°¢F†—2æFF—fT&F6…F–ÆW2ÒÖF‚ç&÷VæB†6Æ×€¢F†—2æFF—fT&F6…F–ÆW2¢&F–òÀ¢Ô”åô$D4…õD”ÄU2À¢Ô…ô$D4…õD”ÄU0¢’“°¢Ğ ¢&—fFR†46ö×ÆWFT6†–ÆG&Vâ‡F–ÆS¢f–VÆEF–ÆR“¢&ööÆVâ°¢6öç7B6†–ÆDW‡öæVçBÒF–ÆRæFW67&—F÷"ç6×ÆTW‡öæVçBÒ°¢6öç7B&6U‚ÒF–ÆRæFW67&—F÷"çF–ÆU‚¢&ã°¢6öç7B&6U’ÒF–ÆRæFW67&—F÷"çF–ÆU’¢&ã°¢f÷"†ÆWB’Òã²’Â&ã²’²²’°¢f÷"†ÆWB‚Òã²‚Â&ã²‚²²’°¢6öç7B¶W“¢W'6—7FVçEF–ÆT¶W’ÒG¶6†–ÆDW‡öæVçGÓ¢G²†&6U‚²‚’çFõ7G&–ær‚—Ó¢G²†&6U’²’’çFõ7G&–ær‚—Ö°¢6öç7B6†–ÆBÒF†—2çF–ÆTÖævWB†¶W’“°¢–b‚6†–ÆBÇÂ6†–ÆBæ6÷fW&vU—†VÇ2ÂD”ÄUõ•„TÅô4õTåB’&WGW&âfÇ6S°¢Ğ¢Ğ¢&WGW&âG'VS°¢Ğ ¢&—fFRF–ÆUG&ç6f÷&Ò€¢F–ÆS¢f–VÆEF–ÆRÀ¢F&vWD6ÖW&¢6ÖW&6æ6†÷BÀ¢7V7C¢çVÖ&W ¢“¢²66ÆUƒ¢çVÖ&W#²66ÆU“¢çVÖ&W#²öfg6WEƒ¢çVÖ&W#²öfg6WE“¢çVÖ&W"ÒÂçVÆÂ°¢6öç7B7äW‡öæVçBÒF–ÆU7äW‡öæVçB‡F–ÆRæFW67&—F÷"ç6×ÆTW‡öæVçB“°¢6öç7B66ÆU’Ò66ÆT÷fW$G–F–2‡F&vWD6ÖW&ç66ÆRÂ7äW‡öæVçB“°¢6öç7BG&ç6f÷&ÒÒ6µG&ç6f÷&Ò‡°¢66ÆUƒ¢66ÆU’¢7V7BÀ¢66ÆU’À¢öfg6WEƒ¢f—†VDF–ffW&Væ6T÷fW$G–F–2‡F&vWD6ÖW&æ6VçFW%‚ÂF–ÆRæFW67&—F÷"æ6VçFW%‚Â7äW‡öæVçB’À¢öfg6WE“¢f—†VDF–ffW&Væ6T÷fW$G–F–2‡F&vWD6ÖW&æ6VçFW%’ÂF–ÆRæFW67&—F÷"æ6VçFW%’Â7äW‡öæVçB¢Ò“°¢&WGW&âG&ç6f÷&Ô—4f–æ—FR‡G&ç6f÷&Ò’òG&ç6f÷&Ò¢çVÆÃ°¢Ğ ¢&—fFRWf–7D6öÆEF–ÆW2‚“¢fö–B°¢6öç7B66†UF&vWBÒÖF‚æÖ‚€¢Ô…ô44„TEõD”ÄU2À¢F†—2æ7W'&VçEf—6–&ÆT¶W—2ç6—¦R²44„Uô„•5Dõ%•õD”ÄUõ$U4U%dP¢“°¢–b‡F†—2çF–ÆTÖç6—¦RÃÒ66†UF&vWB’&WGW&ã°¢6öç7B6æF–FFW2Ò²ââçF†—2çF–ÆTÖçfÇVW2‚•Ğ¢æf–ÇFW"‡F–ÆRÓâF†—2æ7W'&VçEf—6–&ÆT¶W—2æ†2‡F–ÆRæFW67&—F÷"æ¶W’’¢ç6÷'B‚†ÆVgBÂ&–v‡B’ÓâÆVgBæÆ7Ef—6–&ÆTBÒ&–v‡BæÆ7Ef—6–&ÆTB“°¢v†–ÆR‡F†—2çF–ÆTÖç6—¦Râ66†UF&vWBbb6æF–FFW2æÆVæwF‚â’°¢6öç7BF–ÆRÒ6æF–FFW2ç6†–gB‚“°¢–b‚F–ÆR’'&V³°¢F†—2çF–ÆTÖæFVÆWFR‡F–ÆRæFW67&—F÷"æ¶W’“°¢F†—2æFW7G&÷•F–ÆR‡F–ÆR“°¢Ğ¢Ğ ¢&—fFRWf–7DöæT6öÆEF–ÆR‚“¢&ööÆVâ°¢6öç7BF–ÆRÒ²ââçF†—2çF–ÆTÖçfÇVW2‚•Ğ¢æf–ÇFW"†6æF–FFRÓâF†—2æ7W'&VçEf—6–&ÆT¶W—2æ†2†6æF–FFRæFW67&—F÷"æ¶W’’¢ç6÷'B‚†ÆVgBÂ&–v‡B’ÓâÆVgBæÆ7Ef—6–&ÆTBÒ&–v‡BæÆ7Ef—6–&ÆTB•³Ó°¢–b‚F–ÆR’&WGW&âfÇ6S°¢F†—2çF–ÆTÖæFVÆWFR‡F–ÆRæFW67&—F÷"æ¶W’“°¢F†—2æFW7G&÷•F–ÆR‡F–ÆR“°¢&WGW&âG'VS°¢Ğ ¢&—fFRFW7G&÷•F–ÆR‡F–ÆS¢f–VÆEF–ÆR“¢fö–B°¢–b‡F†—2æ66WFVDFÆ2bbF–ÆRæFÆ56Æ÷B’F†—2æ66WFVDFÆ2ç&VÆV6R‡F–ÆRæFÆ56Æ÷B“°¢F–ÆRç7FFT'VffW"æFW7G&÷’‚“°¢F–ÆRæÖWF'VffW"æFW7G&÷’‚“°¢F–ÆRæ6÷VçFW$'VffW"æFW7G&÷’‚“°¢F–ÆRç&W7VÇEFW‡GW&RæFW7G&÷’‚“°¢F–ÆRçVÆ—G•FW‡GW&RæFW7G&÷’‚“°¢F–ÆRæ6öÆ÷W%FW‡GW&RæFW7G&÷’‚“°¢F–ÆRæWf–FVæ6UFW‡GW&RæFW7G&÷’‚“°¢F–ÆRæF—&V7EVæ–f÷&ÒæFW7G&÷’‚“°¢F–ÆRçW'GW&%Væ–f÷&ÒæFW7G&÷’‚“°¢F–ÆRæ6öÆ÷W%Væ–f÷&ÒæFW7G&÷’‚“°¢F–ÆRç&W6VçEVæ–f÷&ÒæFW7G&÷’‚“°¢F–ÆRç&W6WEVæ–f÷&ÒæFW7G&÷’‚“°¢Ğ ¢&—fFR&W6—¦T6çf2‡v–GFƒ¢çVÖ&W"Â†V–v‡C¢çVÖ&W"“¢fö–B°¢–b‡F†—2æF—7Æ•v–GF‚ÓÓÒv–GF‚bbF†—2æF—7Æ”†V–v‡BÓÓÒ†V–v‡B’&WGW&ã°¢F†—2æF—7Æ•v–GF‚Òv–GFƒ°¢F†—2æF—7Æ”†V–v‡BÒ†V–v‡C°¢F†—2æ6çf2çv–GF‚Òv–GFƒ°¢F†—2æ6çf2æ†V–v‡BÒ†V–v‡C°¢F†—2æ6öçFW‡Bæ6öæf–wW&R‡²FWf–6S¢F†—2æFWf–6RÂf÷&ÖC¢F†—2æ6çf4f÷&ÖBÂÇ†ÖöFS¢v÷VRrÒ“°¢Ğ ¢&—fFR7FF–27–æ276W'E6†FW%fÆ–B†ÖöGVÆS¢uU6†FW$ÖöGVÆRÂÆ&VÃ¢7G&–ær“¢&öÖ—6SÇfö–Câ°¢6öç7B6ö×–ÆF–öâÒv—BÖöGVÆRævWD6ö×–ÆF–öä–æfò‚“°¢6öç7BW'&÷'2Ò6ö×–ÆF–öâæÖW76vW2æf–ÇFW"‚†ÖW76vS¢²G—S¢7G&–ærÒ’ÓâÖW76vRçG—RÓÓÒvW'&÷"r“°¢–b†W'&÷'2æÆVæwF‚ÓÓÒ’&WGW&ã°¢6öç7Bf—'7BÒW'&÷'5³Ó°¢F‡&÷ræWrW'&÷"†G¶Æ&VÇÒtu4ÂÆ–æRG¶f—'7BæÆ–æTçV×Ó¢G¶f—'7BæÖW76vWÖ“°¢Ğ§Ğ 