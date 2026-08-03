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
  atlasNeedsClear: boolean;
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
    this.latestRequest = request;
    const gateWaiters = this.testGateWaiters.splice(0);
    for (const resolve of gateWaiters) resolve();
    if (!this.running) void this.pump();
  }

  setSuspended(suspended: boolean): void {
    this.suspended = suspended;
    if (!suspended) return;
    this.latestRequest = null;
    this.currentQueue = [];
    this.pendingLevelOffsets = [];
    const gateWaiters = this.testGateWaiters.splice(0);
    for (const resolve of gateWaiters) resolve();
    const batchWaiters = this.testBatchWaiters.splice(0);
    for (const waiter of batchWaiters) waiter.reject(new Error('Renderer suspended before the requested test batches completed'));
    const pauseWaiters = this.testPauseWaiters.splice(0);
    for (const resolve of pauseWaiters) resolve();
  }

  setTestSchedulerPaused(paused: boolean): Promise<void> {
    this.testSchedulerPaused = paused;
    if (paused) {
      if (!this.testBatchExecuting) return Promise.resolve();
      return new Promise(resolve => this.testPauseWaiters.push(resolve));
    }
    this.testBatchPermits = 0;
    const waiters = this.testGateWaiters.splice(0);
    for (const resolve of waiters) resolve();
    return Promise.resolve();
  }

  releaseTestBatches(count: number): Promise<{
    batchRevision: number;
    requestId: number;
    requestBatchCount: number;
  }> {
    if (!Number.isFinite(count) || count < 1) {
      return Promise.reject(new Error('Batch release count must be a positive finite integer'));
    }
    const permitted = Math.floor(count);
    const requestId = this.latestRequest?.requestId ?? this.currentRequest?.requestId;
    if (requestId === undefined) return Promise.reject(new Error('No numerical request is available for the test gate'));
    const targetCount = (this.testRequestBatchCounts.get(requestId) ?? 0) + permitted;
    this.testBatchPermits += permitted;
    const waiters = this.testGateWaiters.splice(0);
    for (const resolve of waiters) resolve();
    return new Promise((resolve, reject) => this.testBatchWaiters.push({
      requestId, targetCount, resolve, reject
    }));
  }

  get completedTestBatchRevision(): number {
    return this.testBatchRevision;
  }

  get currentTestRequestId(): number {
    return this.latestRequest?.requestId ?? this.currentRequest?.requestId ?? 0;
  }

  nextContinuityFrame(afterFrame: number) {
    return this.atlasPresenter?.nextContinuityFrame(afterFrame)
      ?? Promise.reject(new Error('Continuity frames require the atlas presenter'));
  }

  get isBusy(): boolean {
    return this.running
      || this.latestRequest !== null
      || this.currentQueue.length > 0
      || this.pendingBatches.length > 0
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
    const requestedWidth = Math.max(1, cssWidth * devicePixelRatio);
    const requestedHeight = Math.max(1, cssHeight * devicePixelRatio);
    const displayScale = Math.min(1, limit / requestedWidth, limit / requestedHeight);
    const width = Math.max(1, Math.floor(requestedWidth * displayScale));
    const height = Math.max(1, Math.floor(requestedHeight * displayScale));
    this.resizeCanvas(width, height);
    const renderHeight = this.renderHeight(cssWidth, cssHeight, devicePixelRatio);
    const aspect = Math.max(1, cssWidth) / Math.max(1, cssHeight);
    const fineExponent = sampleExponentForViewport(targetCamera, renderHeight, 0);
    const settledPresentation = (this.currentRequest?.interaction ?? 'settled') === 'settled';
    const drawTiles: FieldTile[] = [];
    const seen = new Set<PersistentTileKey>();

    for (const levelOffset of [4, 3, 2, 1, 0]) {
      for (const descriptor of visibleTileDescriptors(
        targetCamera,
        aspect,
        renderHeight,
        0.5,
        0…7254 tokens truncated…age: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
    });
    const directUniform = this.device.createBuffer({
      size: DIRECT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const perturbUniform = this.device.createBuffer({
      size: PERTURB_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const colourUniform = this.device.createBuffer({
      size: COLOUR_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const presentUniform = this.device.createBuffer({
      size: PRESENT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const resetUniform = this.device.createBuffer({
      size: RESET_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const directGroup = this.device.createBindGroup({
      layout: this.directPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: directUniform } },
        { binding: 1, resource: { buffer: stateBuffer } },
        { binding: 2, resource: { buffer: metaBuffer } },
        { binding: 3, resource: resultTexture.createView() },
        { binding: 4, resource: qualityTexture.createView() },
        { binding: 5, resource: { buffer: counterBuffer } }
      ]
    });
    const colourGroup = this.device.createBindGroup({
      layout: this.colourPipeline.getBindGroupLayout(0),
      entries: this.acceptedAtlas && atlasSlot
        ? [
          { binding: 0, resource: { buffer: colourUniform } },
          { binding: 1, resource: resultTexture.createView() },
          { binding: 2, resource: qualityTexture.createView() },
          { binding: 3, resource: this.acceptedAtlas.colour.createView() },
          { binding: 4, resource: this.acceptedAtlas.quality.createView() },
          { binding: 5, resource: this.acceptedAtlas.evidence.createView() }
        ]
        : [
          { binding: 0, resource: { buffer: colourUniform } },
          { binding: 1, resource: resultTexture.createView() },
          { binding: 2, resource: qualityTexture.createView() },
          { binding: 3, resource: colourTexture.createView() },
          { binding: 4, resource: evidenceTexture.createView() }
        ]
    });
    const clearGroup = this.device.createBindGroup({
      layout: this.clearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.clearUniform } },
        { binding: 1, resource: resultTexture.createView() },
        { binding: 2, resource: qualityTexture.createView() },
        { binding: 3, resource: colourTexture.createView() }
      ]
    });
    const resetGroup = this.device.createBindGroup({
      layout: this.resetPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: resetUniform } },
        { binding: 1, resource: { buffer: stateBuffer } },
        { binding: 2, resource: { buffer: metaBuffer } }
      ]
    });
    const presentLinearGroup = this.createPresentGroup(
      presentUniform,
      colourTexture,
      qualityTexture,
      this.linearSampler
    );
    const presentNearestGroup = this.createPresentGroup(
      presentUniform,
      colourTexture,
      qualityTexture,
      this.nearestSampler
    );

    const centerMagnitude = Math.max(
      1,
      Math.abs(fixedToNumber(descriptor.centerX)),
      Math.abs(fixedToNumber(descriptor.centerY))
    );
    const sampleStep = Math.pow(2, descriptor.sampleExponent);
    const directMode: 0 | 1 = sampleStep > centerMagnitude * Math.pow(2, -21) ? 0 : 1;
    const centerXSplit = fixedSplitF32(descriptor.centerX);
    const centerYSplit = fixedSplitF32(descriptor.centerY);
    const nextXSplit = fixedSplitF32(fixedAddScaled(
      descriptor.centerX,
      1,
      descriptor.sampleExponent
    ));
    const nextYSplit = fixedSplitF32(fixedAddScaled(
      descriptor.centerY,
      1,
      descriptor.sampleExponent
    ));
    const requiresPerturbation = !splitChanged(centerXSplit, nextXSplit)
      || !splitChanged(centerYSplit, nextYSplit);
    const now = performance.now();
    return {
      descriptor,
      stateBuffer,
      metaBuffer,
      counterBuffer,
      resultTexture,
      qualityTexture,
      colourTexture,
      evidenceTexture,
      directUniform,
      perturbUniform,
      colourUniform,
      presentUniform,
      resetUniform,
      directGroup,
      perturbGroup: null,
      colourGroup,
      clearGroup,
      resetGroup,
      presentLinearGroup,
      presentNearestGroup,
      atlasSlot,
      atlasNeedsClear: atlasSlot !== null,
      health: { ...EMPTY_HEALTH },
      iterationFrontier: 0,
      coveragePixels: 0,
      resolvedPixels: 0,
      capPresentationMode: 0,
      lastVisibleAt: now,
      lastNumericalUpdateAt: 0,
      createdAt: now,
      palettePhase: Number.NaN,
      directMode,
      requiresPerturbation,
      numericalMode: directMode === 0 ? 'f32-direct' : 'double-float-direct',
      referenceState: 'none',
      reference: null,
      pendingReference: null,
      pendingReset: null,
      referenceError: null,
      repairPass: 0,
      referenceTarget: 0
    };
  }

  private createPresentGroup(
    uniform: GPUBuffer,
    colourTexture: GPUTexture,
    qualityTexture: GPUTexture,
    sampler: GPUSampler
  ): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: colourTexture.createView() },
        { binding: 3, resource: qualityTexture.createView() }
      ]
    });
  }

  private createPerturbGroup(tile: FieldTile, reference: TileGpuReference): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.perturbPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tile.perturbUniform } },
        { binding: 1, resource: { buffer: tile.stateBuffer } },
        { binding: 2, resource: { buffer: tile.metaBuffer } },
        { binding: 3, resource: tile.resultTexture.createView() },
        { binding: 4, resource: tile.qualityTexture.createView() },
        { binding: 5, resource: { buffer: tile.counterBuffer } },
        { binding: 6, resource: { buffer: reference.buffer } }
      ]
    });
  }

  private submitBatch(
    batch: readonly ScheduledWork[],
    request: PersistentTileRequest
  ): PendingBatch {
    const keys = new Set(batch.map(scheduled => scheduled.tile.descriptor.key));
    if (keys.size !== batch.length) throw new Error('A tile was scheduled twice in one GPU batch');
    for (const pending of this.pendingBatches) {
      if (pending.batch.some(scheduled => keys.has(scheduled.tile.descriptor.key))) {
        throw new Error('A tile already has an in-flight GPU mutation');
      }
    }
    const slot = this.readbackSlots.find(candidate => !candidate.busy);
    if (!slot) throw new Error('Counter readback ring exhausted');
    slot.busy = true;
    for (const scheduled of batch) {
      const tile = scheduled.tile;
      if (tile.numericalMode === 'perturbation') {
        this.device.queue.writeBuffer(
          tile.perturbUniform,
          0,
          this.createPerturbParameterData(scheduled)
        );
      } else {
        this.device.queue.writeBuffer(
          tile.directUniform,
          0,
          this.createDirectParameterData(scheduled)
        );
      }
      this.device.queue.writeBuffer(
        tile.colourUniform,
        0,
        this.createColourParameterData(scheduled, request.palettePhase)
      );
    }

    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(slot.buffer);
    for (let batchIndex = 0; batchIndex < batch.length; batchIndex++) {
      const scheduled = batch[batchIndex];
      const tile = scheduled.tile;
      if (this.requiresNumericalDispatch(scheduled)) {
        encoder.clearBuffer(tile.counterBuffer);
        const iterationPass = encoder.beginComputePass();
        if (tile.numericalMode === 'perturbation') {
          if (!tile.perturbGroup) throw new Error(`Missing perturbation bind group for ${tile.descriptor.key}`);
          iterationPass.setPipeline(this.perturbPipeline);
          iterationPass.setBindGroup(0, tile.perturbGroup);
        } else {
          iterationPass.setPipeline(this.directPipeline);
          iterationPass.setBindGroup(0, tile.directGroup);
        }
        iterationPass.dispatchWorkgroups(
          Math.ceil(PERSISTENT_TILE_SIZE / 8),
          Math.ceil(PERSISTENT_TILE_SIZE / 8)
        );
        iterationPass.end();
      }

      const colourPass = encoder.beginComputePass();
      colourPass.setPipeline(this.colourPipeline);
      colourPass.setBindGroup(0, tile.colourGroup);
      colourPass.dispatchWorkgroups(
        Math.ceil(PERSISTENT_TILE_SIZE / 8),
        Math.ceil(PERSISTENT_TILE_SIZE / 8)
      );
      colourPass.end();
      if (this.requiresNumericalDispatch(scheduled)) {
        encoder.copyBufferToBuffer(
          tile.counterBuffer,
          0,
          slot.buffer,
          batchIndex * COUNTER_READBACK_STRIDE,
          COUNTER_BYTES
        );
      }
    }
    const submittedAt = performance.now();
    this.device.queue.submit([encoder.finish()]);
    for (const scheduled of batch) {
      if (scheduled.tile.atlasSlot) scheduled.tile.atlasNeedsClear = false;
    }
    const completed = slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const mapped = new Uint32Array(slot.buffer.getMappedRange());
      const health = batch.flatMap((scheduled, batchIndex) => {
        if (!this.requiresNumericalDispatch(scheduled)) return [];
        const offset = batchIndex * (COUNTER_READBACK_STRIDE / Uint32Array.BYTES_PER_ELEMENT);
        return [{
          scheduled,
          value: {
            activePixels: mapped[offset] ?? 0,
            escapedPixels: mapped[offset + 1] ?? 0,
            analyticInteriorPixels: mapped[offset + 2] ?? 0,
            cappedPixels: mapped[offset + 3] ?? 0,
            nonFinitePixels: mapped[offset + 4] ?? 0,
            glitchPixels: mapped[offset + 5] ?? 0,
            orbitExhaustedPixels: mapped[offset + 6] ?? 0
          }
        }];
      });
      slot.buffer.unmap();
      return { completedAt: performance.now(), health };
    });
    this.submittedChunks += batch.length;
    this.atlasPublications += this.acceptedAtlas ? batch.length : 0;
    this.avoidedAtlasCopies += this.acceptedAtlas ? batch.length * 3 : 0;
    return { batch, request, submittedAt, slot, completed };
  }

  private requiresNumericalDispatch(scheduled: ScheduledWork): boolean {
    return scheduled.work.chunkIterations > 0
      || scheduled.capMode > scheduled.tile.capPresentationMode;
  }

  private createDirectParameterData(scheduled: ScheduledWork): ArrayBuffer {
    const data = new ArrayBuffer(DIRECT_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    const signed = new Int32Array(data);
    const tile = scheduled.tile;
    const [centerXHi, centerXLo] = fixedSplitF32(tile.descriptor.centerX);
    const [centerYHi, centerYLo] = fixedSplitF32(tile.descriptor.centerY);
    floats[0] = centerXHi;
    floats[1] = centerXLo;
    floats[2] = centerYHi;
    floats[3] = centerYLo;
    signed[4] = tile.descriptor.sampleExponent;
    unsigned[5] = PERSISTENT_TILE_SIZE;
    unsigned[6] = scheduled.scheduledEnd;
    unsigned[7] = scheduled.work.chunkIterations;
    unsigned[8] = tile.directMode;
    unsigned[9] = scheduled.capMode;
    return data;
  }

  private createPerturbParameterData(scheduled: ScheduledWork): ArrayBuffer {
    const tile = scheduled.tile;
    const reference = tile.reference;
    if (!reference) throw new Error(`Missing tile reference for ${tile.descriptor.key}`);
    const data = new ArrayBuffer(PERTURB_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    const signed = new Int32Array(data);
    const [centerXHi, centerXLo] = fixedSplitF32(tile.descriptor.centerX);
    const [centerYHi, centerYLo] = fixedSplitF32(tile.descriptor.centerY);
    const [deltaXHi, deltaXLo] = fixedSplitF32(fixedSub(tile.descriptor.centerX, reference.centerX));
    const [deltaYHi, deltaYLo] = fixedSplitF32(fixedSub(tile.descriptor.centerY, reference.centerY));
    floats[0] = centerXHi;
    floats[1] = centerXLo;
    floats[2] = centerYHi;
    floats[3] = centerYLo;
    floats[4] = deltaXHi;
    floats[5] = deltaXLo;
    floats[6] = deltaYHi;
    floats[7] = deltaYLo;
    signed[8] = tile.descriptor.sampleExponent;
    unsigned[9] = PERSISTENT_TILE_SIZE;
    unsigned[10] = scheduled.scheduledEnd;
    unsigned[11] = scheduled.work.chunkIterations;
    unsigned[12] = reference.length;
    unsigned[13] = scheduled.capMode;
    unsigned[14] = reference.bits;
    unsigned[15] = tile.repairPass;
    floats[16] = GLITCH_RATIO;
    return data;
  }

  private createColourParameterData(
    scheduled: ScheduledWork,
    palettePhase: number
  ): ArrayBuffer {
    const data = new ArrayBuffer(COLOUR_PARAMETER_BYTES);
    const unsigned = new Uint32Array(data);
    const floats = new Float32Array(data);
    unsigned[0] = PERSISTENT_TILE_SIZE;
    if (this.acceptedAtlas && scheduled.tile.atlasSlot) {
      unsigned[1] = Math.abs(scheduled.tile.palettePhase - palettePhase) > 1e-6 ? 1 : 0;
      unsigned[2] = scheduled.tile.iterationFrontier;
      const forceCapPublication = scheduled.capMode > scheduled.tile.capPresentationMode;
      unsigned[3] = (forceCapPublication ? 1 : 0) | (scheduled.tile.atlasNeedsClear ? 2 : 0);
      unsigned[4] = scheduled.tile.atlasSlot.x;
      unsigned[5] = scheduled.tile.atlasSlot.y;
      floats[6] = palettePhase;
      floats[7] = PALETTE_LENGTH;
    } else {
      floats[2] = palettePhase;
      floats[3] = PALETTE_LENGTH;
    }
    return data;
  }

  private renderHeight(
    cssWidth: number,
    cssHeight: number,
    devicePixelRatio: number
  ): number {
    const dpr = clamp(devicePixelRatio, 1, 2);
    const requestedWidth = Math.max(1, cssWidth * dpr);
    const requestedHeight = Math.max(1, cssHeight * dpr);
    const scale = Math.min(1, Math.sqrt(MAX_NUMERICAL_PIXELS / (requestedWidth * requestedHeight)));
    return Math.max(1, Math.floor(requestedHeight * scale));
  }

  private async awaitTestBatchPermit(): Promise<boolean> {
    while (this.testSchedulerPaused
      && this.testBatchPermits <= 0
      && !this.dead
      && !this.suspended
      && !this.latestRequest) {
      await new Promise<void>(resolve => this.testGateWaiters.push(resolve));
    }
    if (this.testSchedulerPaused && this.testBatchPermits > 0) {
      this.testBatchPermits--;
      return true;
    }
    return false;
  }

  private noteTestBatchCompleted(requestId: number): void {
    this.testBatchRevision++;
    const requestBatchCount = (this.testRequestBatchCounts.get(requestId) ?? 0) + 1;
    this.testRequestBatchCounts.set(requestId, requestBatchCount);
    const ready = this.testBatchWaiters.filter(
      waiter => waiter.requestId === requestId && waiter.targetCount <= requestBatchCount
    );
    this.testBatchWaiters = this.testBatchWaiters.filter(waiter => !ready.includes(waiter));
    for (const waiter of ready) waiter.resolve({
      batchRevision: this.testBatchRevision,
      requestId,
      requestBatchCount
    });
  }

  private quantumBudget(interaction: PersistentTileRequest['interaction']): number {
    return interaction === 'moving'
      ? MOVING_QUANTUM_MS
      : interaction === 'settling' ? SETTLING_QUANTUM_MS : SETTLED_QUANTUM_MS;
  }

  private batchTarget(interaction: PersistentTileRequest['interaction']): number {
    return interaction === 'moving'
      ? MOVING_BATCH_TARGET_MS
      : interaction === 'settling' ? SETTLING_BATCH_TARGET_MS : SETTLED_BATCH_TARGET_MS;
  }

  private adaptBatchSize(
    interaction: PersistentTileRequest['interaction'],
    elapsedMs: number
  ): void {
    const target = this.batchTarget(interaction);
    const ratio = clamp(target / Math.max(0.25, elapsedMs), 0.6, 1.5);
    this.adaptiveBatchTiles = Math.round(clamp(
      this.adaptiveBatchTiles * ratio,
      MIN_BATCH_TILES,
      MAX_BATCH_TILES
    ));
  }

  private hasCompleteChildren(tile: FieldTile): boolean {
    const childExponent = tile.descriptor.sampleExponent - 1;
    const baseX = tile.descriptor.tileX * 2n;
    const baseY = tile.descriptor.tileY * 2n;
    for (let y = 0n; y < 2n; y++) {
      for (let x = 0n; x < 2n; x++) {
        const key: PersistentTileKey = `${childExponent}:${(baseX + x).toString()}:${(baseY + y).toString()}`;
        const child = this.tileMap.get(key);
        if (!child || child.coveragePixels < TILE_PIXEL_COUNT) return false;
      }
    }
    return true;
  }

  private tileTransform(
    tile: FieldTile,
    targetCamera: CameraSnapshot,
    aspect: number
  ): { scaleX: number; scaleY: number; offsetX: number; offsetY: number } | null {
    const spanExponent = tileSpanExponent(tile.descriptor.sampleExponent);
    const scaleY = scaleOverDyadic(targetCamera.scale, spanExponent);
    const transform = packTransform({
      scaleX: scaleY * aspect,
      scaleY,
      offsetX: fixedDifferenceOverDyadic(targetCamera.centerX, tile.descriptor.centerX, spanExponent),
      offsetY: fixedDifferenceOverDyadic(targetCamera.centerY, tile.descriptor.centerY, spanExponent)
    });
    return transformIsFinite(transform) ? transform : null;
  }

  private evictColdTiles(): void {
    const cacheTarget = Math.max(
      MAX_CACHED_TILES,
      this.currentVisibleKeys.size + CACHE_HISTORY_TILE_RESERVE
    );
    if (this.tileMap.size <= cacheTarget) return;
    const candidates = [...this.tileMap.values()]
      .filter(tile => !this.currentVisibleKeys.has(tile.descriptor.key))
      .sort((left, right) => left.lastVisibleAt - right.lastVisibleAt);
    while (this.tileMap.size > cacheTarget && candidates.length > 0) {
      const tile = candidates.shift();
      if (!tile) break;
      this.tileMap.delete(tile.descriptor.key);
      this.destroyTile(tile);
    }
  }

  private evictOneColdTile(): boolean {
    const tile = [...this.tileMap.values()]
      .filter(candidate => !this.currentVisibleKeys.has(candidate.descriptor.key))
      .sort((left, right) => left.lastVisibleAt - right.lastVisibleAt)[0];
    if (!tile) return false;
    this.tileMap.delete(tile.descriptor.key);
    this.destroyTile(tile);
    return true;
  }

  private destroyTile(tile: FieldTile): void {
    if (this.acceptedAtlas && tile.atlasSlot) this.acceptedAtlas.release(tile.atlasSlot);
    tile.stateBuffer.destroy();
    tile.metaBuffer.destroy();
    tile.counterBuffer.destroy();
    tile.resultTexture.destroy();
    tile.qualityTexture.destroy();
    tile.colourTexture.destroy();
    tile.evidenceTexture.destroy();
    tile.directUniform.destroy();
    tile.perturbUniform.destroy();
    tile.colourUniform.destroy();
    tile.presentUniform.destroy();
    tile.resetUniform.destroy();
  }

  private resizeCanvas(width: number, height: number): void {
    if (this.displayWidth === width && this.displayHeight === height) return;
    this.displayWidth = width;
    this.displayHeight = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.context.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' });
  }

  private static async assertShaderValid(module: GPUShaderModule, label: string): Promise<void> {
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message: { type: string }) => message.type === 'error');
    if (errors.length === 0) return;
    const first = errors[0];
    throw new Error(`${label} WGSL line ${first.lineNum}: ${first.message}`);
  }
}
