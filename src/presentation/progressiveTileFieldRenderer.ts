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
const COLOUR_PARAMETER_BYTES = 16;
const PRESENT_PARAMETER_BYTES = 16;
const CLEAR_PARAMETER_BYTES = 16;
const RESET_PARAMETER_BYTES = 16;
const COUNTER_VALUES = 7;
const COUNTER_BYTES = COUNTER_VALUES * Uint32Array.BYTES_PER_ELEMENT;
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
  queuedChunks: 0,
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
  counterReadback: GPUBuffer;
  resultTexture: GPUTexture;
  qualityTexture: GPUTexture;
  colourTexture: GPUTexture;
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
  private displayWidth = 1;
  private displayHeight = 1;
  private statsValue: PersistentFieldStats = EMPTY_STATS;
  private runtimeErrorListener: ((message: string) => void) | null = null;
  private adaptiveBatchTiles = INITIAL_BATCH_TILES;
  private completedChunks = 0;
  private lastBatchMs = 0;
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

    const directModule = device.createShaderModule({ code: tileDirectIterationShader });
    const perturbModule = device.createShaderModule({ code: tilePerturbationShader });
    const colourModule = device.createShaderModule({ code: tileColourShader });
    const clearModule = device.createShaderModule({ code: tileClearShader });
    const resetModule = device.createShaderModule({ code: tileResetNumericalShader });
    const presentModule = device.createShaderModule({ code: tilePresentShader });
    await Promise.all([
      this.assertShaderValid(directModule, 'tile direct iteration'),
      this.assertShaderValid(perturbModule, 'tile perturbation'),
      this.assertShaderValid(colourModule, 'tile colour'),
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
    const useLegacyPresenter = new URLSearchParams(location.search).get('presenter') === 'legacy';
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
    if (this.dead) return;
    this.latestRequest = request;
    if (!this.running) void this.pump();
  }

  get isBusy(): boolean {
    return this.running
      || this.latestRequest !== null
      || this.currentQueue.length > 0
      || this.pendingLevelOffsets.length > 0;
  }

  get stats(): PersistentFieldStats {
    const now = performance.now();
    const visibleTiles = [...this.currentVisibleKeys]
      .map(key => this.tileMap.get(key))
      .filter((tile): tile is FieldTile => Boolean(tile));
    const newestNumericalUpdate = visibleTiles.reduce(
      (latest, tile) => Math.max(latest, tile.lastNumericalUpdateAt),
      0
    );
    const oldestCreated = visibleTiles.reduce(
      (oldest, tile) => Math.min(oldest, tile.createdAt),
      now
    );
    return {
      ...this.statsValue,
      interaction: this.currentRequest?.interaction ?? this.statsValue.interaction,
      cachedTiles: this.tileMap.size,
      pendingReferences: this.referenceAtlas.pendingCount,
      referenceFailures: this.referenceAtlas.failureCount,
      numericalFreshnessMs: newestNumericalUpdate > 0 ? now - newestNumericalUpdate : 0,
      presentationHistoryMs: visibleTiles.length > 0 ? now - oldestCreated : 0
    };
  }

  present(
    targetCamera: CameraSnapshot,
    cssWidth: number,
    cssHeight: number,
    devicePixelRatio: number
  ): boolean {
    if (this.dead || cssWidth <= 0 || cssHeight <= 0 || devicePixelRatio <= 0) return false;
    const limit = this.device.limits.maxTextureDimension2D;
    const width = Math.min(limit, Math.max(1, Math.floor(cssWidth * devicePixelRatio)));
    const height = Math.min(limit, Math.max(1, Math.floor(cssHeight * devicePixelRatio)));
    this.resizeCanvas(width, height);
    const renderHeight = this.renderHeight(cssWidth, cssHeight, devicePixelRatio);
    const aspect = Math.max(1, cssWidth) / Math.max(1, cssHeight);
    const fineExponent = sampleExponentForViewport(targetCamera, renderHeigóŽ¶¶‰žËkºwµçUÈè½Õ¹Ñ•É	Õ™™•Èôô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐ½±½ÕÉÉ½ÕÀ€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹½±½ÕÉA¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èì‰Õ™™•Èè½±½ÕÉU¹¥™½É´ôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èÉ•ÍÕ±ÑQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€È°É•Í½ÕÉ”èÅÕ…±¥ÑåQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ì°É•Í½ÕÉ”è½±½ÕÉQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐ±•…ÉÉ½ÕÀ€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹±•…ÉA¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¡¥Ì¹±•…ÉU¹¥™½É´ôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èÉ•ÍÕ±ÑQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€È°É•Í½ÕÉ”èÅÕ…±¥ÑåQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ì°É•Í½ÕÉ”è½±½ÕÉQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐÉ•Í•ÑÉ½ÕÀ€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹É•Í•ÑA¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èì‰Õ™™•ÈèÉ•Í•ÑU¹¥™½É´ôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èì‰Õ™™•ÈèÍÑ…Ñ•	Õ™™•Èôô°(€€€€€€€ì‰¥¹‘¥¹œè€È°É•Í½ÕÉ”èì‰Õ™™•Èèµ•Ñ…	Õ™™•Èôô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐÁÉ•Í•¹Ñ1¥¹•…ÉÉ½ÕÀ€ôÑ¡¥Ì¹É•…Ñ•AÉ•Í•¹ÑÉ½ÕÀ (€€€€€ÁÉ•Í•¹ÑU¹¥™½É´°(€€€€€½±½ÕÉQ•áÑÕÉ”°(€€€€€ÅÕ…±¥ÑåQ•áÑÕÉ”°(€€€€€Ñ¡¥Ì¹±¥¹•…ÉM…µÁ±•È(€€€€¤ì(€€€½¹ÍÐÁÉ•Í•¹Ñ9•…É•ÍÑÉ½ÕÀ€ôÑ¡¥Ì¹É•…Ñ•AÉ•Í•¹ÑÉ½ÕÀ (€€€€€ÁÉ•Í•¹ÑU¹¥™½É´°(€€€€€½±½ÕÉQ•áÑÕÉ”°(€€€€€ÅÕ…±¥ÑåQ•áÑÕÉ”°(€€€€€Ñ¡¥Ì¹¹•…É•ÍÑM…µÁ±•È(€€€€¤ì((€€€½¹ÍÐ•¹Ñ•É5…¹¥ÑÕ‘”€ô5…Ñ ¹µ…à (€€€€€€Ä°(€€€€€5…Ñ ¹…‰Ì¡™¥á•‘Q½9Õµ‰•È¡‘•ÍÉ¥ÁÑ½È¹•¹Ñ•É`¤¤°(€€€€€5…Ñ ¹…‰Ì¡™¥á•‘Q½9Õµ‰•È¡‘•ÍÉ¥ÁÑ½È¹•¹Ñ•Éd¤¤(€€€€¤ì(€€€½¹ÍÐÍ…µÁ±•MÑ•À€ô5…Ñ ¹Á½Ü È°‘•ÍÉ¥ÁÑ½È¹Í…µÁ±•áÁ½¹•¹Ð¤ì(€€€½¹ÍÐ‘¥É•Ñ5½‘”è€Àð€Ä€ôÍ…µÁ±•MÑ•À€ø•¹Ñ•É5…¹¥ÑÕ‘”€¨5…Ñ ¹Á½Ü È°€´ÈÄ¤€ü€À€è€Äì(€€€½¹ÍÐ•¹Ñ•ÉaMÁ±¥Ð€ô™¥á•‘MÁ±¥ÑÌÈ¡‘•ÍÉ¥ÁÑ½È¹•¹Ñ•É`¤ì(€€€½¹ÍÐ•¹Ñ•ÉeMÁ±¥Ð€ô™¥á•‘MÁ±¥ÑÌÈ¡‘•ÍÉ¥ÁÑ½È¹•¹Ñ•Éd¤ì(€€€½¹ÍÐ¹•áÑaMÁ±¥Ð€ô™¥á•‘MÁ±¥ÑÌÈ¡™¥á•‘‘‘M…±• (€€€€€‘•ÍÉ¥ÁÑ½È¹•¹Ñ•É`°(€€€€€€Ä°(€€€€€‘•ÍÉ¥ÁÑ½È¹Í…µÁ±•áÁ½¹•¹Ð(€€€€¤¤ì(€€€½¹ÍÐ¹•áÑeMÁ±¥Ð€ô™¥á•‘MÁ±¥ÑÌÈ¡™¥á•‘‘‘M…±• (€€€€€‘•ÍÉ¥ÁÑ½È¹•¹Ñ•Éd°(€€€€€€Ä°(€€€€€‘•ÍÉ¥ÁÑ½È¹Í…µÁ±•áÁ½¹•¹Ð(€€€€¤¤ì(€€€½¹ÍÐÉ•ÅÕ¥É•ÍA•ÉÑÕÉ‰…Ñ¥½¸€ô€…ÍÁ±¥Ñ¡…¹•¡•¹Ñ•ÉaMÁ±¥Ð°¹•áÑaMÁ±¥Ð¤(€€€€€ñð€…ÍÁ±¥Ñ¡…¹•¡•¹Ñ•ÉeMÁ±¥Ð°¹•áÑeMÁ±¥Ð¤ì(€€€½¹ÍÐ¹½Ü€ôÁ•É™½Éµ…¹”¹¹½Ü ¤ì(€€€É•ÑÕÉ¸ì(€€€€€‘•ÍÉ¥ÁÑ½È°(€€€€€ÍÑ…Ñ•	Õ™™•È°(€€€€€µ•Ñ…	Õ™™•È°(€€€€€½Õ¹Ñ•É	Õ™™•È°(€€€€€½Õ¹Ñ•ÉI•…‘‰…¬°(€€€€€É•ÍÕ±ÑQ•áÑÕÉ”°(€€€€€ÅÕ…±¥ÑåQ•áÑÕÉ”°(€€€€€½±½ÕÉQ•áÑÕÉ”°(€€€€€‘¥É•ÑU¹¥™½É´°(€€€€€Á•ÉÑÕÉ‰U¹¥™½É´°(€€€€€½±½ÕÉU¹¥™½É´°(€€€€€ÁÉ•Í•¹ÑU¹¥™½É´°(€€€€€É•Í•ÑU¹¥™½É´°(€€€€€‘¥É•ÑÉ½ÕÀ°(€€€€€Á•ÉÑÕÉ‰É½ÕÀè¹Õ±°°(€€€€€½±½ÕÉÉ½ÕÀ°(€€€€€±•…ÉÉ½ÕÀ°(€€€€€É•Í•ÑÉ½ÕÀ°(€€€€€ÁÉ•Í•¹Ñ1¥¹•…ÉÉ½ÕÀ°(€€€€€ÁÉ•Í•¹Ñ9•…É•ÍÑÉ½ÕÀ°(€€€€€…Ñ±…ÍM±½Ð°(€€€€€¡•…±Ñ èì€¸¸¹5AQe}!1Q ô°(€€€€€¥Ñ•É…Ñ¥½¹É½¹Ñ¥•Èè€À°(€€€€€½Ù•É…•A¥á•±Ìè€À°(€€€€€É•Í½±Ù•‘A¥á•±Ìè€À°(€€€€€…ÁAÉ•Í•¹Ñ…Ñ¥½¹5½‘”è€À°(€€€€€±…ÍÑY¥Í¥‰±•Ðè¹½Ü°(€€€€€±…ÍÑ9Õµ•É¥…±UÁ‘…Ñ•Ðè€À°(€€€€€É•…Ñ•‘Ðè¹½Ü°(€€€€€Á…±•ÑÑ•A¡…Í”è9Õµ‰•È¹9…8°(€€€€€‘¥É•Ñ5½‘”°(€€€€€É•ÅÕ¥É•ÍA•ÉÑÕÉ‰…Ñ¥½¸°(€€€€€¹Õµ•É¥…±5½‘”è‘¥É•Ñ5½‘”€ôôô€À€ü€˜ÌÈµ‘¥É•Ðœ€è€‘½Õ‰±”µ™±½…Ðµ‘¥É•Ðœ°(€€€€€É•™•É•¹•MÑ…Ñ”è€¹½¹”œ°(€€€€€É•™•É•¹”è¹Õ±°°(€€€€€Á•¹‘¥¹I•™•É•¹”è¹Õ±°°(€€€€€Á•¹‘¥¹I•Í•Ðè¹Õ±°°(€€€€€É•™•É•¹•ÉÉ½Èè¹Õ±°°(€€€€€É•Á…¥ÉA…ÍÌè€À°(€€€€€É•™•É•¹•Q…É•Ðè€À(€€€ôì(€ô((€ÁÉ¥Ù…Ñ”É•…Ñ•AÉ•Í•¹ÑÉ½ÕÀ (€€€Õ¹¥™½É´èAU	Õ™™•È°(€€€½±½ÕÉQ•áÑÕÉ”èAUQ•áÑÕÉ”°(€€€ÅÕ…±¥ÑåQ•áÑÕÉ”èAUQ•áÑÕÉ”°(€€€Í…µÁ±•ÈèAUM…µÁ±•È(€€¤èAU	¥¹‘É½ÕÀì(€€€É•ÑÕÉ¸Ñ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹ÁÉ•Í•¹ÑA¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èì‰Õ™™•ÈèÕ¹¥™½É´ôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èÍ…µÁ±•Èô°(€€€€€€€ì‰¥¹‘¥¹œè€È°É•Í½ÕÉ”è½±½ÕÉQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ì°É•Í½ÕÉ”èÅÕ…±¥ÑåQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô(€€€€€t(€€€ô¤ì(€ô((€ÁÉ¥Ù…Ñ”É•…Ñ•A•ÉÑÕÉ‰É½ÕÀ¡Ñ¥±”è¥•±‘Q¥±”°É•™•É•¹”èQ¥±•ÁÕI•™•É•¹”¤èAU	¥¹‘É½ÕÀì(€€€É•ÑÕÉ¸Ñ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹Á•ÉÑÕÉ‰A¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¥±”¹Á•ÉÑÕÉ‰U¹¥™½É´ôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¥±”¹ÍÑ…Ñ•	Õ™™•Èôô°(€€€€€€€ì‰¥¹‘¥¹œè€È°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¥±”¹µ•Ñ…	Õ™™•Èôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ì°É•Í½ÕÉ”èÑ¥±”¹É•ÍÕ±ÑQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ð°É•Í½ÕÉ”èÑ¥±”¹ÅÕ…±¥ÑåQ•áÑÕÉ”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ô°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¥±”¹½Õ¹Ñ•É	Õ™™•Èôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ø°É•Í½ÕÉ”èì‰Õ™™•ÈèÉ•™•É•¹”¹‰Õ™™•Èôô(€€€€€t(€€€ô¤ì(€ô((€ÁÉ¥Ù…Ñ”…Íå¹Œ•á•ÕÑ•	…Ñ  (€€€‰…Ñ èÉ•…‘½¹±äM¡•‘Õ±•‘]½É­mt°(€€€Á…±•ÑÑ•A¡…Í”è¹Õµ‰•È(€€¤èAÉ½µ¥Í”ñÙ½¥øì(€€€™½È€¡½¹ÍÐÍ¡•‘Õ±•½˜‰…Ñ ¤ì(€€€€€½¹ÍÐÑ¥±”€ôÍ¡•‘Õ±•¹Ñ¥±”ì(€€€€€¥˜€¡Ñ¥±”¹¹Õµ•É¥…±5½‘”€ôôô€Á•ÉÑÕÉ‰…Ñ¥½¸œ¤ì(€€€€€€€Ñ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹ÝÉ¥Ñ•	Õ™™•È (€€€€€€€€€Ñ¥±”¹Á•ÉÑÕÉ‰U¹¥™½É´°(€€€€€€€€€€À°(€€€€€€€€€Ñ¡¥Ì¹É•…Ñ•A•ÉÑÕÉ‰A…É…µ•Ñ•É…Ñ„¡Í¡•‘Õ±•¤(€€€€€€€€¤ì(€€€€€ô•±Í”ì(€€€€€€€Ñ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹ÝÉ¥Ñ•	Õ™™•È (€€€€€€€€€Ñ¥±”¹‘¥É•ÑU¹¥™½É´°(€€€€€€€€€€À°(€€€€€€€€€Ñ¡¥Ì¹É•…Ñ•¥É•ÑA…É…µ•Ñ•É…Ñ„¡Í¡•‘Õ±•¤(€€€€€€€€¤ì(€€€€€ô(€€€€€Ñ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹ÝÉ¥Ñ•	Õ™™•È (€€€€€€€Ñ¥±”¹½±½ÕÉU¹¥™½É´°(€€€€€€€€À°(€€€€€€€Ñ¡¥Ì¹É•…Ñ•½±½ÕÉA…É…µ•Ñ•É…Ñ„¡Á…±•ÑÑ•A¡…Í”¤(€€€€€€¤ì(€€€ô((€€€½¹ÍÐ•¹½‘•È€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•½µµ…¹‘¹½‘•È ¤ì(€€€™½È€¡½¹ÍÐÍ¡•‘Õ±•½˜‰…Ñ ¤ì(€€€€€½¹ÍÐÑ¥±”€ôÍ¡•‘Õ±•¹Ñ¥±”ì(€€€€€¥˜€¡Í¡•‘Õ±•¹Ý½É¬¹¡Õ¹­%Ñ•É…Ñ¥½¹Ì€ø€À¤ì(€€€€€€€•¹½‘•È¹±•…É	Õ™™•È¡Ñ¥±”¹½Õ¹Ñ•É	Õ™™•È¤ì(€€€€€€€½¹ÍÐ¥Ñ•É…Ñ¥½¹A…ÍÌ€ô•¹½‘•È¹‰•¥¹½µÁÕÑ•A…ÍÌ ¤ì(€€€€€€€¥˜€¡Ñ¥±”¹¹Õµ•É¥…±5½‘”€ôôô€Á•ÉÑÕÉ‰…Ñ¥½¸œ¤ì(€€€€€€€€€¥˜€ …Ñ¥±”¹Á•ÉÑÕÉ‰É½ÕÀ¤Ñ¡É½Ü¹•ÜÉÉ½È¡5¥ÍÍ¥¹œÁ•ÉÑÕÉ‰…Ñ¥½¸‰¥¹É½ÕÀ™½È€‘íÑ¥±”¹‘•ÍÉ¥ÁÑ½È¹­•åõ€¤ì(€€€€€€€€€¥Ñ•É…Ñ¥½¹A…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹Á•ÉÑÕÉ‰A¥Á•±¥¹”¤ì(€€€€€€€€€¥Ñ•É…Ñ¥½¹A…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°Ñ¥±”¹Á•ÉÑÕÉ‰É½ÕÀ¤ì(€€€€€€€ô•±Í”ì(€€€€€€€€€¥Ñ•É…Ñ¥½¹A…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹‘¥É•ÑA¥Á•±¥¹”¤ì(€€€€€€€€€¥Ñ•É…Ñ¥½¹A…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°Ñ¥±”¹‘¥É•ÑÉ½ÕÀ¤ì(€€€€€€€ô(€€€€€€€¥Ñ•É…Ñ¥½¹A…ÍÌ¹‘¥ÍÁ…Ñ¡]½É­É½ÕÁÌ (€€€€€€€€€5…Ñ ¹•¥°¡AIM%MQ9Q}Q%1}M%i€¼€à¤°(€€€€€€€€€5…Ñ ¹•¥°¡AIM%MQ9Q}Q%1}M%i€¼€à¤(€€€€€€€€¤ì(€€€€€€€¥Ñ•É…Ñ¥½¹A…ÍÌ¹•¹ ¤ì(€€€€€ô((€€€€€½¹ÍÐ½±½ÕÉA…ÍÌ€ô•¹½‘•È¹‰•¥¹½µÁÕÑ•A…ÍÌ ¤ì(€€€€€½±½ÕÉA…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹½±½ÕÉA¥Á•±¥¹”¤ì(€€€€€½±½ÕÉA…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°Ñ¥±”¹½±½ÕÉÉ½ÕÀ¤ì(€€€€€½±½ÕÉA…ÍÌ¹‘¥ÍÁ…Ñ¡]½É­É½ÕÁÌ (€€€€€€€5…Ñ ¹•¥°¡AIM%MQ9Q}Q%1}M%i€¼€à¤°(€€€€€€€5…Ñ ¹•¥°¡AIM%MQ9Q}Q%1}M%i€¼€à¤(€€€€€€¤ì(€€€€€½±½ÕÉA…ÍÌ¹•¹ ¤ì(€€€€€¥˜€¡Ñ¡¥Ì¹…•ÁÑ•‘Ñ±…Ì€˜˜Ñ¥±”¹…Ñ±…ÍM±½Ð¤ì(€€€€€€€Ñ¡¥Ì¹…•ÁÑ•‘Ñ±…Ì¹•¹½‘•½Áä (€€€€€€€€€•¹½‘•È°Ñ¥±”¹…Ñ±…ÍM±½Ð°Ñ¥±”¹½±½ÕÉQ•áÑÕÉ”°Ñ¥±”¹ÅÕ…±¥ÑåQ•áÑÕÉ”(€€€€€€€€¤ì(€€€€€ô(€€€€€¥˜€¡Í¡•‘Õ±•¹Ý½É¬¹¡Õ¹­%Ñ•É…Ñ¥½¹Ì€ø€À¤ì(€€€€€€€•¹½‘•È¹½Áå	Õ™™•ÉQ½	Õ™™•È (€€€€€€€€€Ñ¥±”¹½Õ¹Ñ•É	Õ™™•È°(€€€€€€€€€€À°(€€€€€€€€€Ñ¥±”¹½Õ¹Ñ•ÉI•…‘‰…¬°(€€€€€€€€€€À°(€€€€€€€€€=U9QI}	eQL(€€€€€€€€¤ì(€€€€€ô(€€€ô(€€€Ñ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹ÍÕ‰µ¥Ð¡m•¹½‘•È¹™¥¹¥Í  ¥t¤ì(€€€…Ý…¥ÐÑ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹½¹MÕ‰µ¥ÑÑ•‘]½É­½¹” ¤ì((€€€…Ý…¥ÐAÉ½µ¥Í”¹…±°¡‰…Ñ ¹µ…À¡…Íå¹ŒÍ¡•‘Õ±•€ôøì(€€€€€¥˜€¡Í¡•‘Õ±•¹Ý½É¬¹¡Õ¹­%Ñ•É…Ñ¥½¹Ì€ðô€À¤É•ÑÕÉ¸ì(€€€€€½¹ÍÐÉ•…‘‰…¬€ôÍ¡•‘Õ±•¹Ñ¥±”¹½Õ¹Ñ•ÉI•…‘‰…¬ì(€€€€€…Ý…¥ÐÉ•…‘‰…¬¹µ…ÁÍå¹Œ¡AU5…Á5½‘”¹I¤ì(€€€€€½¹ÍÐÙ…±Õ•Ì€ô¹•ÜU¥¹ÐÌÉÉÉ…ä¡É•…‘‰…¬¹•Ñ5…ÁÁ•‘I…¹” ¤¤¹Í±¥” ¤ì(€€€€€É•…‘‰…¬¹Õ¹µ…À ¤ì(€€€€€Í¡•‘Õ±•¹Ñ¥±”¹¡•…±Ñ €ôì(€€€€€€€…Ñ¥Ù•A¥á•±ÌèÙ…±Õ•ÍlÁt€üü€À°(€€€€€€€•Í…Á•‘A¥á•±ÌèÙ…±Õ•ÍlÅt€üü€À°(€€€€€€€…¹…±åÑ¥%¹Ñ•É¥½ÉA¥á•±ÌèÙ…±Õ•ÍlÉt€üü€À°(€€€€€€€…ÁÁ•‘A¥á•±ÌèÙ…±Õ•ÍlÍt€üü€À°(€€€€€€€¹½¹¥¹¥Ñ•A¥á•±ÌèÙ…±Õ•ÍlÑt€üü€À°(€€€€€€€±¥Ñ¡A¥á•±ÌèÙ…±Õ•ÍlÕt€üü€À°(€€€€€€€½É‰¥Ñá¡…ÕÍÑ•‘A¥á•±ÌèÙ…±Õ•ÍlÙt€üü€À(€€€€€ôì(€€€ô¤¤ì(€ô((€ÁÉ¥Ù…Ñ”É•…Ñ•¥É•ÑA…É…µ•Ñ•É…Ñ„¡Í¡•‘Õ±•èM¡•‘Õ±•‘]½É¬¤èÉÉ…å	Õ™™•Èì(€€€½¹ÍÐ‘…Ñ„€ô¹•ÜÉÉ…å	Õ™™•È¡%IQ}AI5QI}	eQL¤ì(€€€½¹ÍÐ™±½…ÑÌ€ô¹•Ü±½…ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€½¹ÍÐÕ¹Í¥¹•€ô¹•ÜU¥¹ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€½¹ÍÐÍ¥¹•€ô¹•Ü%¹ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€½¹ÍÐÑ¥±”€ôÍ¡•‘Õ±•¹Ñ¥±”ì(€€€½¹ÍÐm•¹Ñ•Éa!¤°•¹Ñ•Éa1½t€ô™¥á•‘MÁ±¥ÑÌÈ¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•É`¤ì(€€€½¹ÍÐm•¹Ñ•Ée!¤°•¹Ñ•Ée1½t€ô™¥á•‘MÁ±¥ÑÌÈ¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•Éd¤ì(€€€™±½…ÑÍlÁt€ô•¹Ñ•Éa!¤ì(€€€™±½…ÑÍlÅt€ô•¹Ñ•Éa1¼ì(€€€™±½…ÑÍlÉt€ô•¹Ñ•Ée!¤ì(€€€™±½…ÑÍlÍt€ô•¹Ñ•Ée1¼ì(€€€Í¥¹•‘lÑt€ôÑ¥±”¹‘•ÍÉ¥ÁÑ½È¹Í…µÁ±•áÁ½¹•¹Ðì(€€€Õ¹Í¥¹•‘lÕt€ôAIM%MQ9Q}Q%1}M%iì(€€€Õ¹Í¥¹•‘lÙt€ôÍ¡•‘Õ±•¹Í¡•‘Õ±•‘¹ì(€€€Õ¹Í¥¹•‘lÝt€ôÍ¡•‘Õ±•¹Ý½É¬¹¡Õ¹­%Ñ•É…Ñ¥½¹Ìì(€€€Õ¹Í¥¹•‘lát€ôÑ¥±”¹‘¥É•Ñ5½‘”ì(€€€Õ¹Í¥¹•‘låt€ôÍ¡•‘Õ±•¹…Á5½‘”ì(€€€É•ÑÕÉ¸‘…Ñ„ì(€ô((€ÁÉ¥Ù…Ñ”É•…Ñ•A•ÉÑÕÉ‰A…É…µ•Ñ•É…Ñ„¡Í¡•‘Õ±•èM¡•‘Õ±•‘]½É¬¤èÉÉ…å	Õ™™•Èì(€€€½¹ÍÐÑ¥±”€ôÍ¡•‘Õ±•¹Ñ¥±”ì(€€€½¹ÍÐÉ•™•É•¹”€ôÑ¥±”¹É•™•É•¹”ì(€€€¥˜€ …É•™•É•¹”¤Ñ¡É½Ü¹•ÜÉÉ½È¡5¥ÍÍ¥¹œÑ¥±”É•™•É•¹”™½È€‘íÑ¥±”¹‘•ÍÉ¥ÁÑ½È¹­•åõ€¤ì(€€€½¹ÍÐ‘…Ñ„€ô¹•ÜÉÉ…å	Õ™™•È¡AIQUI	}AI5QI}	eQL¤ì(€€€½¹ÍÐ™±½…ÑÌ€ô¹•Ü±½…ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€½¹ÍÐÕ¹Í¥¹•€ô¹•ÜU¥¹ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€½¹ÍÐÍ¥¹•€ô¹•Ü%¹ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€½¹ÍÐm•¹Ñ•Éa!¤°•¹Ñ•Éa1½t€ô™¥á•‘MÁ±¥ÑÌÈ¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•É`¤ì(€€€½¹ÍÐm•¹Ñ•Ée!¤°•¹Ñ•Ée1½t€ô™¥á•‘MÁ±¥ÑÌÈ¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•Éd¤ì(€€€½¹ÍÐm‘•±Ñ…a!¤°‘•±Ñ…a1½t€ô™¥á•‘MÁ±¥ÑÌÈ¡™¥á•‘MÕˆ¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•É`°É•™•É•¹”¹•¹Ñ•É`¤¤ì(€€€½¹ÍÐm‘•±Ñ…e!¤°‘•±Ñ…e1½t€ô™¥á•‘MÁ±¥ÑÌÈ¡™¥á•‘MÕˆ¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•Éd°É•™•É•¹”¹•¹Ñ•Éd¤¤ì(€€€™±½…ÑÍlÁt€ô•¹Ñ•Éa!¤ì(€€€™±½…ÑÍlÅt€ô•¹Ñ•Éa1¼ì(€€€™±½…ÑÍlÉt€ô•¹Ñ•Ée!¤ì(€€€™±½…ÑÍlÍt€ô•¹Ñ•Ée1¼ì(€€€™±½…ÑÍlÑt€ô‘•±Ñ…a!¤ì(€€€™±½…ÑÍlÕt€ô‘•±Ñ…a1¼ì(€€€™±½…ÑÍlÙt€ô‘•±Ñ…e!¤ì(€€€™±½…ÑÍlÝt€ô‘•±Ñ…e1¼ì(€€€Í¥¹•‘lát€ôÑ¥±”¹‘•ÍÉ¥ÁÑ½È¹Í…µÁ±•áÁ½¹•¹Ðì(€€€Õ¹Í¥¹•‘låt€ôAIM%MQ9Q}Q%1}M%iì(€€€Õ¹Í¥¹•‘lÄÁt€ôÍ¡•‘Õ±•¹Í¡•‘Õ±•‘¹ì(€€€Õ¹Í¥¹•‘lÄÅt€ôÍ¡•‘Õ±•¹Ý½É¬¹¡Õ¹­%Ñ•É…Ñ¥½¹Ìì(€€€Õ¹Í¥¹•‘lÄÉt€ôÉ•™•É•¹”¹±•¹Ñ ì(€€€Õ¹Í¥¹•‘lÄÍt€ôÍ¡•‘Õ±•¹…Á5½‘”ì(€€€Õ¹Í¥¹•‘lÄÑt€ôÉ•™•É•¹”¹‰¥ÑÌì(€€€Õ¹Í¥¹•‘lÄÕt€ôÑ¥±”¹É•Á…¥ÉA…ÍÌì(€€€™±½…ÑÍlÄÙt€ô1%Q!}IQ%<ì(€€€É•ÑÕÉ¸‘…Ñ„ì(€ô((€ÁÉ¥Ù…Ñ”É•…Ñ•½±½ÕÉA…É…µ•Ñ•É…Ñ„¡Á…±•ÑÑ•A¡…Í”è¹Õµ‰•È¤èÉÉ…å	Õ™™•Èì(€€€½¹ÍÐ‘…Ñ„€ô¹•ÜÉÉ…å	Õ™™•È¡=1=UI}AI5QI}	eQL¤ì(€€€½¹ÍÐÕ¹Í¥¹•€ô¹•ÜU¥¹ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€½¹ÍÐ™±½…ÑÌ€ô¹•Ü±½…ÐÌÉÉÉ…ä¡‘…Ñ„¤ì(€€€Õ¹Í¥¹•‘lÁt€ôAIM%MQ9Q}Q%1}M%iì(€€€™±½…ÑÍlÉt€ôÁ…±•ÑÑ•A¡…Í”ì(€€€™±½…ÑÍlÍt€ôA1QQ}19Q ì(€€€É•ÑÕÉ¸‘…Ñ„ì(€ô((€ÁÉ¥Ù…Ñ”É•¹‘•É!•¥¡Ð (€€€ÍÍ]¥‘Ñ è¹Õµ‰•È°(€€€ÍÍ!•¥¡Ðè¹Õµ‰•È°(€€€‘•Ù¥•A¥á•±I…Ñ¥¼è¹Õµ‰•È(€€¤è¹Õµ‰•Èì(€€€½¹ÍÐ‘ÁÈ€ô±…µÀ¡‘•Ù¥•A¥á•±I…Ñ¥¼°€Ä°€È¤ì(€€€½¹ÍÐÉ•ÅÕ•ÍÑ•‘]¥‘Ñ €ô5…Ñ ¹µ…à Ä°ÍÍ]¥‘Ñ €¨‘ÁÈ¤ì(€€€½¹ÍÐÉ•ÅÕ•ÍÑ•‘!•¥¡Ð€ô5…Ñ ¹µ…à Ä°ÍÍ!•¥¡Ð€¨‘ÁÈ¤ì(€€€½¹ÍÐÍ…±”€ô5…Ñ ¹µ¥¸ Ä°5…Ñ ¹ÍÅÉÐ¡5a}9U5I%1}A%a1L€¼€¡É•ÅÕ•ÍÑ•‘]¥‘Ñ €¨É•ÅÕ•ÍÑ•‘!•¥¡Ð¤¤¤ì(€€€É•ÑÕÉ¸5…Ñ ¹µ…à Ä°5…Ñ ¹™±½½È¡É•ÅÕ•ÍÑ•‘!•¥¡Ð€¨Í…±”¤¤ì(€ô((€ÁÉ¥Ù…Ñ”ÅÕ…¹ÑÕµ	Õ‘•Ð¡¥¹Ñ•É…Ñ¥½¸èA•ÉÍ¥ÍÑ•¹ÑQ¥±•I•ÅÕ•ÍÑl¥¹Ñ•É…Ñ¥½¸t¤è¹Õµ‰•Èì(€€€É•ÑÕÉ¸¥¹Ñ•É…Ñ¥½¸€ôôô€µ½Ù¥¹œœ(€€€€€€ü5=Y%9}EU9QU5}5L(€€€€€€è¥¹Ñ•É…Ñ¥½¸€ôôô€Í•ÑÑ±¥¹œœ€üMQQ1%9}EU9QU5}5L€èMQQ1}EU9QU5}5Lì(€ô((€ÁÉ¥Ù…Ñ”‰…Ñ¡Q…É•Ð¡¥¹Ñ•É…Ñ¥½¸èA•ÉÍ¥ÍÑ•¹ÑQ¥±•I•ÅÕ•ÍÑl¥¹Ñ•É…Ñ¥½¸t¤è¹Õµ‰•Èì(€€€É•ÑÕÉ¸¥¹Ñ•É…Ñ¥½¸€ôôô€µ½Ù¥¹œœ(€€€€€€ü5=Y%9}	Q!}QIQ}5L(€€€€€€è¥¹Ñ•É…Ñ¥½¸€ôôô€Í•ÑÑ±¥¹œœ€üMQQ1%9}	Q!}QIQ}5L€èMQQ1}	Q!}QIQ}5Lì(€ô((€ÁÉ¥Ù…Ñ”…‘…ÁÑ	…Ñ¡M¥é” (€€€¥¹Ñ•É…Ñ¥½¸èA•ÉÍ¥ÍÑ•¹ÑQ¥±•I•ÅÕ•ÍÑl¥¹Ñ•É…Ñ¥½¸t°(€€€•±…ÁÍ•‘5Ìè¹Õµ‰•È(€€¤èÙ½¥ì(€€€½¹ÍÐÑ…É•Ð€ôÑ¡¥Ì¹‰…Ñ¡Q…É•Ð¡¥¹Ñ•É…Ñ¥½¸¤ì(€€€½¹ÍÐÉ…Ñ¥¼€ô±…µÀ¡Ñ…É•Ð€¼5…Ñ ¹µ…à À¸ÈÔ°•±…ÁÍ•‘5Ì¤°€À¸Ø°€Ä¸Ô¤ì(€€€Ñ¡¥Ì¹…‘…ÁÑ¥Ù•	…Ñ¡Q¥±•Ì€ô5…Ñ ¹É½Õ¹¡±…µÀ (€€€€€Ñ¡¥Ì¹…‘…ÁÑ¥Ù•	…Ñ¡Q¥±•Ì€¨É…Ñ¥¼°(€€€€€5%9}	Q!}Q%1L°(€€€€€5a}	Q!}Q%1L(€€€€¤¤ì(€ô((€ÁÉ¥Ù…Ñ”¡…Í½µÁ±•Ñ•¡¥±‘É•¸¡Ñ¥±”è¥•±‘Q¥±”¤è‰½½±•…¸ì(€€€½¹ÍÐ¡¥±‘áÁ½¹•¹Ð€ôÑ¥±”¹‘•ÍÉ¥ÁÑ½È¹Í…µÁ±•áÁ½¹•¹Ð€´€Äì(€€€½¹ÍÐ‰…Í•`€ôÑ¥±”¹‘•ÍÉ¥ÁÑ½È¹Ñ¥±•`€¨€É¸ì(€€€½¹ÍÐ‰…Í•d€ôÑ¥±”¹‘•ÍÉ¥ÁÑ½È¹Ñ¥±•d€¨€É¸ì(€€€™½È€¡±•Ðä€ô€Á¸ìä€ð€É¸ìä¬¬¤ì(€€€€€™½È€¡±•Ðà€ô€Á¸ìà€ð€É¸ìà¬¬¤ì(€€€€€€€½¹ÍÐ­•äèA•ÉÍ¥ÍÑ•¹ÑQ¥±•-•ä€ô€‘í¡¥±‘áÁ½¹•¹Ñôè‘ì¡‰…Í•`€¬à¤¹Ñ½MÑÉ¥¹œ ¥ôè‘ì¡‰…Í•d€¬ä¤¹Ñ½MÑÉ¥¹œ ¥õ€ì(€€€€€€€½¹ÍÐ¡¥±€ôÑ¡¥Ì¹Ñ¥±•5…À¹•Ð¡­•ä¤ì(€€€€€€€¥˜€ …¡¥±ñð¡¥±¹½Ù•É…•A¥á•±Ì€ðQ%1}A%a1}=U9P¤É•ÑÕÉ¸™…±Í”ì(€€€€€ô(€€€ô(€€€É•ÑÕÉ¸ÑÉÕ”ì(€ô((€ÁÉ¥Ù…Ñ”Ñ¥±•QÉ…¹Í™½É´ (€€€Ñ¥±”è¥•±‘Q¥±”°(€€€Ñ…É•Ñ…µ•É„è…µ•É…M¹…ÁÍ¡½Ð°(€€€…ÍÁ•Ðè¹Õµ‰•È(€€¤èìÍ…±•`è¹Õµ‰•ÈìÍ…±•dè¹Õµ‰•Èì½™™Í•Ñ`è¹Õµ‰•Èì½™™Í•Ñdè¹Õµ‰•Èôð¹Õ±°ì(€€€½¹ÍÐÍÁ…¹áÁ½¹•¹Ð€ôÑ¥±•MÁ…¹áÁ½¹•¹Ð¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹Í…µÁ±•áÁ½¹•¹Ð¤ì(€€€½¹ÍÐÍ…±•d€ôÍ…±•=Ù•Éå…‘¥Œ¡Ñ…É•Ñ…µ•É„¹Í…±”°ÍÁ…¹áÁ½¹•¹Ð¤ì(€€€½¹ÍÐÑÉ…¹Í™½É´€ôÁ…­QÉ…¹Í™½É´¡ì(€€€€€Í…±•`èÍ…±•d€¨…ÍÁ•Ð°(€€€€€Í…±•d°(€€€€€½™™Í•Ñ`è™¥á•‘¥™™•É•¹•=Ù•Éå…‘¥Œ¡Ñ…É•Ñ…µ•É„¹•¹Ñ•É`°Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•É`°ÍÁ…¹áÁ½¹•¹Ð¤°(€€€€€½™™Í•Ñdè™¥á•‘¥™™•É•¹•=Ù•Éå…‘¥Œ¡Ñ…É•Ñ…µ•É„¹•¹Ñ•Éd°Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹•¹Ñ•Éd°ÍÁ…¹áÁ½¹•¹Ð¤(€€€ô¤ì(€€€É•ÑÕÉ¸ÑÉ…¹Í™½Éµ%Í¥¹¥Ñ”¡ÑÉ…¹Í™½É´¤€üÑÉ…¹Í™½É´€è¹Õ±°ì(€ô((€ÁÉ¥Ù…Ñ”•Ù¥Ñ½±‘Q¥±•Ì ¤èÙ½¥ì(€€€½¹ÍÐ…¡•Q…É•Ð€ô5…Ñ ¹µ…à (€€€€€5a}!}Q%1L°(€€€€€Ñ¡¥Ì¹ÕÉÉ•¹ÑY¥Í¥‰±•-•åÌ¹Í¥é”€¬!}!%MQ=Ie}Q%1}IMIY(€€€€¤ì(€€€¥˜€¡Ñ¡¥Ì¹Ñ¥±•5…À¹Í¥é”€ðô…¡•Q…É•Ð¤É•ÑÕÉ¸ì(€€€½¹ÍÐ…¹‘¥‘…Ñ•Ì€ôl¸¸¹Ñ¡¥Ì¹Ñ¥±•5…À¹Ù…±Õ•Ì ¥t(€€€€€€¹™¥±Ñ•È¡Ñ¥±”€ôø€…Ñ¡¥Ì¹ÕÉÉ•¹ÑY¥Í¥‰±•-•åÌ¹¡…Ì¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹­•ä¤¤(€€€€€€¹Í½ÉÐ ¡±•™Ð°É¥¡Ð¤€ôø±•™Ð¹±…ÍÑY¥Í¥‰±•Ð€´É¥¡Ð¹±…ÍÑY¥Í¥‰±•Ð¤ì(€€€Ý¡¥±”€¡Ñ¡¥Ì¹Ñ¥±•5…À¹Í¥é”€ø…¡•Q…É•Ð€˜˜…¹‘¥‘…Ñ•Ì¹±•¹Ñ €ø€À¤ì(€€€€€½¹ÍÐÑ¥±”€ô…¹‘¥‘…Ñ•Ì¹Í¡¥™Ð ¤ì(€€€€€¥˜€ …Ñ¥±”¤‰É•…¬ì(€€€€€Ñ¡¥Ì¹Ñ¥±•5…À¹‘•±•Ñ”¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹­•ä¤ì(€€€€€Ñ¡¥Ì¹‘•ÍÑÉ½åQ¥±”¡Ñ¥±”¤ì(€€€ô(€ô((€ÁÉ¥Ù…Ñ”•Ù¥Ñ=¹•½±‘Q¥±” ¤è‰½½±•…¸ì(€€€½¹ÍÐÑ¥±”€ôl¸¸¹Ñ¡¥Ì¹Ñ¥±•5…À¹Ù…±Õ•Ì ¥t(€€€€€€¹™¥±Ñ•È¡…¹‘¥‘…Ñ”€ôø€…Ñ¡¥Ì¹ÕÉÉ•¹ÑY¥Í¥‰±•-•åÌ¹¡…Ì¡…¹‘¥‘…Ñ”¹‘•ÍÉ¥ÁÑ½È¹­•ä¤¤(€€€€€€¹Í½ÉÐ ¡±•™Ð°É¥¡Ð¤€ôø±•™Ð¹±…ÍÑY¥Í¥‰±•Ð€´É¥¡Ð¹±…ÍÑY¥Í¥‰±•Ð¥lÁtì(€€€¥˜€ …Ñ¥±”¤É•ÑÕÉ¸™…±Í”ì(€€€Ñ¡¥Ì¹Ñ¥±•5…À¹‘•±•Ñ”¡Ñ¥±”¹‘•ÍÉ¥ÁÑ½È¹­•ä¤ì(€€€Ñ¡¥Ì¹‘•ÍÑÉ½åQ¥±”¡Ñ¥±”¤ì(€€€É•ÑÕÉ¸ÑÉÕ”ì(€ô((€ÁÉ¥Ù…Ñ”‘•ÍÑÉ½åQ¥±”¡Ñ¥±”è¥•±‘Q¥±”¤èÙ½¥ì(€€€¥˜€¡Ñ¡¥Ì¹…•ÁÑ•‘Ñ±…Ì€˜˜Ñ¥±”¹…Ñ±…ÍM±½Ð¤Ñ¡¥Ì¹…•ÁÑ•‘Ñ±…Ì¹É•±•…Í”¡Ñ¥±”¹…Ñ±…ÍM±½Ð¤ì(€€€Ñ¥±”¹ÍÑ…Ñ•	Õ™™•È¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹µ•Ñ…	Õ™™•È¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹½Õ¹Ñ•É	Õ™™•È¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹½Õ¹Ñ•ÉI•…‘‰…¬¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹É•ÍÕ±ÑQ•áÑÕÉ”¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹ÅÕ…±¥ÑåQ•áÑÕÉ”¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹½±½ÕÉQ•áÑÕÉ”¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹‘¥É•ÑU¹¥™½É´¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹Á•ÉÑÕÉ‰U¹¥™½É´¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹½±½ÕÉU¹¥™½É´¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹ÁÉ•Í•¹ÑU¹¥™½É´¹‘•ÍÑÉ½ä ¤ì(€€€Ñ¥±”¹É•Í•ÑU¹¥™½É´¹‘•ÍÑÉ½ä ¤ì(€ô((€ÁÉ¥Ù…Ñ”É•Í¥é•…¹Ù…Ì¡Ý¥‘Ñ è¹Õµ‰•È°¡•¥¡Ðè¹Õµ‰•È¤èÙ½¥ì(€€€¥˜€¡Ñ¡¥Ì¹‘¥ÍÁ±…å]¥‘Ñ €ôôôÝ¥‘Ñ €˜˜Ñ¡¥Ì¹‘¥ÍÁ±…å!•¥¡Ð€ôôô¡•¥¡Ð¤É•ÑÕÉ¸ì(€€€Ñ¡¥Ì¹‘¥ÍÁ±…å]¥‘Ñ €ôÝ¥‘Ñ ì(€€€Ñ¡¥Ì¹‘¥ÍÁ±…å!•¥¡Ð€ô¡•¥¡Ðì(€€€Ñ¡¥Ì¹…¹Ù…Ì¹Ý¥‘Ñ €ôÝ¥‘Ñ ì(€€€Ñ¡¥Ì¹…¹Ù…Ì¹¡•¥¡Ð€ô¡•¥¡Ðì(€€€Ñ¡¥Ì¹½¹Ñ•áÐ¹½¹™¥ÕÉ”¡ì‘•Ù¥”èÑ¡¥Ì¹‘•Ù¥”°™½Éµ…ÐèÑ¡¥Ì¹…¹Ù…Í½Éµ…Ð°…±Á¡…5½‘”è€½Á…ÅÕ”œô¤ì(€ô((€ÁÉ¥Ù…Ñ”ÍÑ…Ñ¥Œ…Íå¹Œ…ÍÍ•ÉÑM¡…‘•ÉY…±¥¡µ½‘Õ±”èAUM¡…‘•É5½‘Õ±”°±…‰•°èÍÑÉ¥¹œ¤èAÉ½µ¥Í”ñÙ½¥øì(€€€½¹ÍÐ½µÁ¥±…Ñ¥½¸€ô…Ý…¥Ðµ½‘Õ±”¹•Ñ½µÁ¥±…Ñ¥½¹%¹™¼ ¤ì(€€€½¹ÍÐ•ÉÉ½ÉÌ€ô½µÁ¥±…Ñ¥½¸¹µ•ÍÍ…•Ì¹™¥±Ñ•È ¡µ•ÍÍ…”èìÑåÁ”èÍÑÉ¥¹œô¤€ôøµ•ÍÍ…”¹ÑåÁ”€ôôô€•ÉÉ½Èœ¤ì(€€€¥˜€¡•ÉÉ½ÉÌ¹±•¹Ñ €ôôô€À¤É•ÑÕÉ¸ì(€€€½¹ÍÐ™¥ÉÍÐ€ô•ÉÉ½ÉÍlÁtì(€€€Ñ¡É½Ü¹•ÜÉÉ½È¡€‘í±…‰•±ô]M0±¥¹”€‘í™¥ÉÍÐ¹±¥¹•9Õµôè€‘í™¥ÉÍÐ¹µ•ÍÍ…•õ€¤ì(€ô)ô