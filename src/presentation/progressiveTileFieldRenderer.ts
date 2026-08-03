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
  atlasSlot: AtlasSlot;
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
  private readonly acceptedAtlas: AcceptedTileAtlas;
  private readonly atlasPresenter: AtlasHistoryPresenter;
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
    acceptedAtlas: AcceptedTileAtlas,
    atlasPresenter: AtlasHistoryPresenter,
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
    this.useLegacyPresenter = new URLSearchParams(location.search).get('presenter') === 'legacy';
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
    const acceptedAtlas = new AcceptedTileAtlas(device);
    const atlasPresenter = await AtlasHistoryPresenter.create(device, context, canvasFormat);
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
    if (this.dead || this.tileMap.size === 0) return false;
    const width = Math.max(1, Math.floor(cssWidth * devicePixelRatio));
    const height = Math.max(1, Math.floor(cssHeight * devicePixelRatio));
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
        0.5,
        levelOffset,
        1
      )) {
        const tile = this.tileMap.get(descriptor.key);
        if (!tile || seen.has(tile.descriptor.key) || tile.coveragePixels <= 0) continue;
        seen.add(tile.descriptor.key);
        drawTiles.push(tile);
      }
    }
    if (drawTiles.length === 0) return false;
    drawTiles.sort((left, right) => right.descriptor.sampleExponent - left.descriptor.sampleExponent);

    const renderable: Array<{ tile: FieldTile; group: GPUBindGroup; transform: ReturnType<TileFieldRenderer['tileTransform']> }> = [];
    for (const tile of drawTiles) {
      if (this.hasCompleteChildren(tile)) continue;
      const transform = this.tileTransform(tile, targetCamera, aspect);
      if (!transform) continue;
      this.device.queue.writeBuffer(tile.presentUniform, 0, new Float32Array([
        transform.scaleX,
        transform.scaleY,
        transform.offsetX,
        transform.offsetY
      ]));
      renderable.push({
        tile,
        transform,
        group: settledPresentation && tile.descriptor.sampleExponent <= fineExponent
          ? tile.presentNearestGroup
          : tile.presentLinearGroup
      });
    }
    if (renderable.length === 0) return false;

    if (!this.useLegacyPresenter) {
      const instances: AtlasInstance[] = renderable.flatMap(item => item.transform
        ? [{ transform: item.transform, slot: item.tile.atlasSlot }]
        : []);
      const authoritative = settledPresentation
        && this.currentVisibleKeys.size > 0
        && [...this.currentVisibleKeys].every(key => {
          const tile = this.tileMap.get(key);
          return Boolean(tile && tile.coveragePixels >= TILE_PIXEL_COUNT);
        });
      return this.atlasPresenter.present(
        targetCamera, aspect, width, height, this.acceptedAtlas, instances, authoritative
      );
    }

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.008, g: 0.01, b: 0.014, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    pass.setPipeline(this.presentPipeline);
    for (const item of renderable) {
      pass.setBindGroup(0, item.group);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  get presentationMode(): 'atlas-history' | 'legacy' {
    return this.useLegacyPresenter ? 'legacy' : 'atlas-history';
  }

  get presentationDiagnostics() {
    return { ...this.atlasPresenter.diagnostics, validationErrors: [...this.deviceErrors] };
  }

  forceDeviceLossForTest(): void {
    if (!this.dead) this.device.destroy();
  }

  dispose(): void {
    if (this.dead) return;
    this.dead = true;
    this.latestRequest = null;
    this.currentQueue = [];
    this.pendingLevelOffsets = [];
    this.referenceAtlas.dispose();
    for (const tile of this.tileMap.values()) this.destroyTile(tile);
    this.tileMap.clear();
    this.atlasPresenter.destroy();
    this.acceptedAtlas.destroy();
    this.clearUniform.destroy();
    this.context.unconfigure();
  }

  private async pump(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (
        !this.dead && (this.latestRequest
        || this.currentQueue.length > 0
        || this.pendingLevelOffsets.length > 0)
      ) {
        if (this.latestRequest) {
          const request = this.latestRequest;
          this.latestRequest = null;
          await this.prepareRequest(request);
        }
        const request = this.currentRequest;
        if (!request) break;

        if (this.currentQueue.length === 0) {
          const admitted = await this.admitNextSpatialLevel(request);
          if (this.latestRequest) continue;
          if (!admitted) break;
        }
        if (this.currentQueue.length === 0) continue;

        const quantumStarted = performance.now();
        const quantumBudget = this.quantumBudget(request.interaction);
        do {
          if (this.latestRequest || this.currentQueue.length === 0) break;
          this.currentQueue.sort((left, right) => left.work.priority - right.work.priority);
          const batch = this.currentQueue.splice(0, this.adaptiveBatchTiles);
          const started = performance.now();
          await this.executeBatch(batch, request.palettePhase);
          this.lastBatchMs = Math.max(0.1, performance.now() - started);
          this.completedChunks += batch.length;
          this.adaptBatchSize(request.interaction, this.lastBatchMs);
          this.finishBatch(batch, request);
          this.updateStats();
        } while (performance.now() - quantumStarted < quantumBudget);

        this.evictColdTiles();
        if (
          !this.latestRequest
          && (this.currentQueue.length > 0 || this.pendingLevelOffsets.length > 0)
        ) {
          await schedulerYield();
        }
      }
    } catch (error) {
      console.error('Progressive tile field scheduler failed', error);
      const message = error instanceof Error ? error.message : String(error);
      this.runtimeErrorListener?.(message);
    } finally {
      this.running = false;
      if (
        this.latestRequest
        || this.currentQueue.length > 0
        || this.pendingLevelOffsets.length > 0
      ) {
        void this.pump();
      }
    }
  }

  private async prepareRequest(request: PersistentTileRequest): Promise<void> {
    this.currentRequest = request;
    this.completedChunks = 0;
    this.lastBatchMs = 0;
    this.currentPlan = [];
    this.currentQueue = [];
    this.currentVisibleKeys = new Set<PersistentTileKey>();
    this.currentRenderHeight = this.renderHeight(
      request.cssWidth,
      request.cssHeight,
      request.devicePixelRatio
    );
    this.currentAspect = Math.max(1, request.cssWidth) / Math.max(1, request.cssHeight);
    this.pendingLevelOffsets = [2, 1, 0];
    this.maximumPlannedLevel = 2;
    await this.activatePendingReferences();
    await this.admitNextSpatialLevel(request);
    this.updateStats();
  }

  private async admitNextSpatialLevel(request: PersistentTileRequest): Promise<boolean> {
    while (this.pendingLevelOffsets.length > 0 && !this.latestRequest) {
      const levelOffset = this.pendingLevelOffsets.shift();
      if (levelOffset === undefined) break;
      const levelPlan = this.planDescriptorsForLevel(request, levelOffset);
      const ready = await this.ensureTiles(levelPlan.map(item => item.descriptor));
      if (!ready || this.latestRequest) return false;

      await this.activatePendingReferences();
      this.currentPlan.push(...levelPlan);
      for (const item of levelPlan) this.currentVisibleKeys.add(item.descriptor.key);
      this.ensureReferencePolicy(request, levelPlan);
      const levelQueue = this.initialWorkQueue(request, levelPlan);
      this.currentQueue.push(...levelQueue);
      this.updateStats();
      if (levelQueue.length > 0) return true;
    }
    return this.currentQueue.length > 0;
  }

  private planDescriptorsForLevel(
    request: PersistentTileRequest,
    levelOffset: number
  ): PlannedDescriptor[] {
    const margin = levelOffset > 0 ? 1 : 0;
    return visibleTileDescriptors(
      request.camera,
      this.currentAspect,
      this.currentRenderHeight,
      request.focusX,
      request.focusY,
      levelOffset,
      margin
    ).map(descriptor => ({ descriptor, levelOffset }));
  }

  private initialWorkQueue(
    request: PersistentTileRequest,
    plan: readonly PlannedDescriptor[]
  ): ScheduledWork[] {
    const queue: ScheduledWork[] = [];
    for (const planned of plan) {
      const tile = this.tileMap.get(planned.descriptor.key);
      if (!tile) continue;
      tile.lastVisibleAt = performance.now();
      const finalTarget = this.effectiveFinalTarget(tile, request);
      const chunkIterations = request.interaction === 'moving'
        ? 64
        : request.interaction === 'settling' ? 96 : 128;
      const needsIterations = tile.iterationFrontier < finalTarget;
      const needsRecolour = Math.abs(tile.palettePhase - request.palettePhase) > 1e-6;
      const scheduledEnd = needsIterations
        ? Math.min(finalTarget, tile.iterationFrontier + chunkIterations)
        : tile.iterationFrontier;
      const scheduled = this.makeScheduledWork(
        request,
        tile,
        planned.levelOffset,
        this.maximumPlannedLevel,
        finalTarget,
        scheduledEnd,
        needsIterations ? chunkIterations : 0
      );
      const needsAcceptanceUpdate = tile.capPresentationMode !== scheduled.capMode;
      if (!needsIterations && !needsRecolour && !needsAcceptanceUpdate) continue;
      queue.push(scheduled);
    }
    return queue;
  }

  private makeScheduledWork(
    request: PersistentTileRequest,
    tile: FieldTile,
    levelOffset: number,
    maximumLevel: number,
    finalTarget: number,
    scheduledEnd: number,
    chunkIterations: number
  ): ScheduledWork {
    const finalChunk = scheduledEnd >= finalTarget;
    const numericallyFinal = finalTarget >= request.targetIterations;
    const retainsAuthoritativeCap = tile.capPresentationMode === 2
      && tile.iterationFrontier >= request.targetIterations;
    const capMode: CapPresentationMode = retainsAuthoritativeCap
      ? 2
      : finalChunk && numericallyFinal && request.interaction === 'settled'
        ? 2
        : (levelOffset > 0 || request.interaction === 'moving') ? 1 : 0;
    const spatialPriority = maximumLevel - levelOffset;
    const coveragePenalty = tile.coveragePixels > 0 ? 2 : 0;
    const iterationTier = Math.floor(tile.iterationFrontier / Math.max(1, chunkIterations || 128));
    const work: PersistentTileWork = {
      requestId: request.requestId,
      key: tile.descriptor.key,
      targetIterations: finalTarget,
      chunkIterations,
      acceptIterationCap: capMode > 0,
      priority: spatialPriority
        + tile.descriptor.distanceFromFocus * 0.15
        + coveragePenalty
        + iterationTier * 0.35
        + (tile.numericalMode === 'perturbation' ? -0.1 : 0)
    };
    return { work, tile, scheduledEnd, finalTarget, levelOffset, capMode };
  }

  private finishBatch(batch: readonly ScheduledWork[], request: PersistentTileRequest): void {
    for (const scheduled of batch) {
      const tile = scheduled.tile;
      tile.iterationFrontier = Math.max(tile.iterationFrontier, scheduled.scheduledEnd);
      if (scheduled.work.chunkIterations > 0) tile.lastNumericalUpdateAt = performance.now();
      tile.lastVisibleAt = performance.now();
      tile.palettePhase = request.palettePhase;
      tile.capPresentationMode = scheduled.capMode;
      const acceptedCoverage = tile.health.escapedPixels
        + tile.health.analyticInteriorPixels
        + (scheduled.capMode > 0 ? tile.health.cappedPixels : 0);
      tile.coveragePixels = Math.max(tile.coveragePixels, acceptedCoverage);
      const finalAcceptedCap = scheduled.capMode === 2
        && scheduled.scheduledEnd >= scheduled.finalTarget;
      const resolvedCoverage = tile.health.escapedPixels
        + tile.health.analyticInteriorPixels
        + (finalAcceptedCap ? tile.health.cappedPixels : 0);
      tile.resolvedPixels = Math.max(tile.resolvedPixels, resolvedCoverage);

      this.maybeRequestRepair(tile, request);
      const continuingPixels = tile.health.activePixels + tile.health.cappedPixels;
      if (
        scheduled.work.chunkIterations > 0
        && continuingPixels > 0
        && scheduled.scheduledEnd < scheduled.finalTarget
        && !this.latestRequest
      ) {
        const nextEnd = Math.min(
          scheduled.finalTarget,
          scheduled.scheduledEnd + scheduled.work.chunkIterations
        );
        this.currentQueue.push(this.makeScheduledWork(
          request,
          tile,
          scheduled.levelOffset,
          this.maximumPlannedLevel,
          scheduled.finalTarget,
          nextEnd,
          scheduled.work.chunkIterations
        ));
      }
    }
  }

  private updateStats(): void {
    const request = this.currentRequest;
    if (!request) return;
    const visible = this.currentPlan
      .map(item => this.tileMap.get(item.descriptor.key))
      .filter((tile): tile is FieldTile => Boolean(tile));
    this.statsValue = {
      requestId: request.requestId,
      interaction: request.interaction,
      visibleTiles: visible.length,
      cachedTiles: this.tileMap.size,
      activeTiles: visible.filter(tile => tile.resolvedPixels < TILE_PIXEL_COUNT).length,
      convergedTiles: visible.filter(tile => tile.resolvedPixels >= TILE_PIXEL_COUNT).length,
      directTiles: visible.filter(tile => tile.numericalMode !== 'perturbation').length,
      perturbationTiles: visible.filter(tile => tile.numericalMode === 'perturbation').length,
      pendingReferences: this.referenceAtlas.pendingCount,
      repairTiles: visible.filter(tile => tile.repairPass > 0).length,
      referenceFailures: this.referenceAtlas.failureCount,
      completedChunks: this.completedChunks,
      queuedChunks: this.currentQueue.length,
      lastBatchMs: this.lastBatchMs,
      numericalFreshnessMs: 0,
      presentationHistoryMs: 0,
      sampleExponent: sampleExponentForViewport(
        request.camera,
        this.currentRenderHeight,
        0
      ),
      tileSize: PERSISTENT_TILE_SIZE
    };
  }

  private ensureReferencePolicy(
    request: PersistentTileRequest,
    plan: readonly PlannedDescriptor[]
  ): void {
    const referenceTarget = request.targetIterations;
    for (const planned of plan) {
      const tile = this.tileMap.get(planned.descriptor.key);
      if (!tile || !this.tileNeedsPerturbation(tile)) continue;
      if (tile.referenceState === 'queued') continue;

      const failurePixels = tile.health.glitchPixels
        + tile.health.orbitExhaustedPixels
        + tile.health.nonFinitePixels;
      const unresolvedPixels = TILE_PIXEL_COUNT - tile.resolvedPixels;
      if (tile.numericalMode === 'perturbation' && tile.reference) {
        if (failurePixels > 0) continue;
        const needsLongerReference = tile.reference.requestedIterations < referenceTarget
          && unresolvedPixels > 0;
        if (!needsLongerReference) continue;
      }
      if (tile.referenceState === 'failed' && tile.referenceTarget >= referenceTarget) continue;

      const reusable = this.referenceAtlas.findReusable(tile.descriptor.key, referenceTarget);
      if (reusable && reusable !== tile.reference) {
        tile.pendingReference = reusable;
        tile.pendingReset = tile.numericalMode === 'perturbation' ? 'unresolved' : 'all';
        tile.referenceState = 'ready';
        tile.referenceTarget = referenceTarget;
        this.requestCurrentAgain();
        continue;
      }
      this.queueReference(tile, tile.repairPass, referenceTarget);
    }
  }

  private tileNeedsPerturbation(tile: FieldTile): boolean {
    if (tile.numericalMode === 'perturbation') return true;
    if (tile.health.nonFinitePixels > 0) return true;
    return tile.requiresPerturbation;
  }

  private queueReference(
    tile: FieldTile,
    repairPass: number,
    referenceTarget: number
  ): void {
    if (repairPass > MAX_REFERENCE_REPAIR_PASSES) {
      tile.referenceState = 'failed';
      tile.referenceTarget = referenceTarget;
      return;
    }
    tile.referenceState = 'queued';
    tile.referenceTarget = referenceTarget;
    tile.referenceError = null;
    void this.referenceAtlas.request(
      tile.descriptor,
      referenceTarget,
      tile.descriptor.distanceFromFocus + repairPass * 0.05,
      repairPass
    ).then(reference => {
      if (this.tileMap.get(tile.descriptor.key) !== tile) return;
      tile.pendingReference = reference;
      tile.pendingReset = tile.numericalMode === 'perturbation' ? 'unresolved' : 'all';
      tile.referenceState = 'ready';
      tile.referenceError = null;
      tile.repairPass = repairPass;
      tile.referenceTarget = referenceTarget;
      this.requestCurrentAgain();
    }).catch(error => {
      if (this.tileMap.get(tile.descriptor.key) !== tile) return;
      tile.referenceError = error instanceof Error ? error.message : String(error);
      tile.referenceState = 'failed';
      tile.referenceTarget = referenceTarget;
    });
  }

  private maybeRequestRepair(tile: FieldTile, request: PersistentTileRequest): void {
    if (tile.numericalMode !== 'perturbation') return;
    const unresolvedFailures = tile.health.glitchPixels
      + tile.health.orbitExhaustedPixels
      + tile.health.nonFinitePixels;
    if (unresolvedFailures <= 0 || tile.referenceState === 'queued') return;
    if (tile.repairPass >= MAX_REFERENCE_REPAIR_PASSES) {
      tile.referenceState = 'failed';
      return;
    }
    this.queueReference(tile, tile.repairPass + 1, request.targetIterations);
  }

  private requestCurrentAgain(): void {
    if (!this.currentRequest || this.latestRequest) return;
    this.latestRequest = this.currentRequest;
    if (!this.running) void this.pump();
  }

  private async activatePendingReferences(): Promise<void> {
    const activations = [...this.tileMap.values()]
      .filter(tile => tile.pendingReference && tile.pendingReset);
    if (activations.length === 0) return;

    for (const tile of activations) {
      const reference = tile.pendingReference;
      if (!reference) continue;
      tile.reference = reference;
      tile.pendingReference = null;
      tile.numericalMode = 'perturbation';
      tile.referenceState = 'ready';
      tile.perturbGroup = this.createPerturbGroup(tile, reference);
    }

    const encoder = this.device.createCommandEncoder();
    for (const tile of activations) {
      const preserveAccepted = tile.pendingReset === 'unresolved';
      this.device.queue.writeBuffer(
        tile.resetUniform,
        0,
        new Uint32Array([PERSISTENT_TILE_SIZE, preserveAccepted ? 1 : 0, 0, 0])
      );
      encoder.clearBuffer(tile.counterBuffer);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.resetPipeline);
      pass.setBindGroup(0, tile.resetGroup);
      pass.dispatchWorkgroups(
        Math.ceil(PERSISTENT_TILE_SIZE / 8),
        Math.ceil(PERSISTENT_TILE_SIZE / 8)
      );
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    for (const tile of activations) {
      const preserveAccepted = tile.pendingReset === 'unresolved';
      tile.pendingReset = null;
      tile.iterationFrontier = 0;
      tile.health = {
        ...EMPTY_HEALTH,
        activePixels: Math.max(0, TILE_PIXEL_COUNT
          - (preserveAccepted ? tile.resolvedPixels : 0))
      };
      if (!preserveAccepted) tile.resolvedPixels = 0;
      tile.capPresentationMode = tile.coveragePixels > 0 ? 1 : 0;
      tile.lastNumericalUpdateAt = performance.now();
    }
  }

  private effectiveFinalTarget(tile: FieldTile, request: PersistentTileRequest): number {
    if (tile.numericalMode === 'perturbation') return request.targetIterations;
    if (this.tileNeedsPerturbation(tile)) {
      return Math.min(request.targetIterations, DIRECT_SAFETY_ITERATIONS);
    }
    return request.targetIterations;
  }

  private async ensureTiles(
    descriptors: readonly PersistentTileDescriptor[]
  ): Promise<boolean> {
    const missing = descriptors.filter(descriptor => !this.tileMap.has(descriptor.key));
    for (let start = 0; start < missing.length; start += RESOURCE_CREATION_BATCH_TILES) {
      if (this.latestRequest) return false;
      const batchDescriptors = missing.slice(start, start + RESOURCE_CREATION_BATCH_TILES);
      const created: FieldTile[] = [];
      this.device.pushErrorScope('validation');
      for (const descriptor of batchDescriptors) {
        if (this.tileMap.has(descriptor.key)) continue;
        const tile = this.createTile(descriptor);
        this.tileMap.set(descriptor.key, tile);
        created.push(tile);
      }
      const validationError = await this.device.popErrorScope();
      if (validationError) {
        for (const tile of created) {
          this.tileMap.delete(tile.descriptor.key);
          this.destroyTile(tile);
        }
        throw new Error(`Tile resource layout validation failed: ${validationError.message}`);
      }
      if (created.length > 0) {
        const encoder = this.device.createCommandEncoder();
        for (const tile of created) {
          encoder.clearBuffer(tile.stateBuffer);
          encoder.clearBuffer(tile.metaBuffer);
          encoder.clearBuffer(tile.counterBuffer);
          const pass = encoder.beginComputePass();
          pass.setPipeline(this.clearPipeline);
          pass.setBindGroup(0, tile.clearGroup);
          pass.dispatchWorkgroups(
            Math.ceil(PERSISTENT_TILE_SIZE / 8),
            Math.ceil(PERSISTENT_TILE_SIZE / 8)
          );
          pass.end();
        }
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
      }
      if (start + RESOURCE_CREATION_BATCH_TILES < missing.length) await schedulerYield();
    }
    return !this.latestRequest;
  }

  private createTile(descriptor: PersistentTileDescriptor): FieldTile {
    if (this.acceptedAtlas.availableSlots === 0 && !this.evictOneColdTile()) {
      throw new Error('Accepted tile atlas exhausted with no retireable tile');
    }
    const atlasSlot = this.acceptedAtlas.allocate();
    const stateBuffer = this.device.createBuffer({
      size: TILE_PIXEL_COUNT * STATE_BYTES_PER_PIXEL,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const metaBuffer = this.device.createBuffer({
      size: TILE_PIXEL_COUNT * META_BYTES_PER_PIXEL,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const counterBuffer = this.device.createBuffer({
      size: COUNTER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    const counterReadback = this.device.createBuffer({
      size: COUNTER_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    const resultTexture = this.device.createTexture({
      size: [PERSISTENT_TILE_SIZE, PERSISTENT_TILE_SIZE],
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    });
    const qualityTexture = this.device.createTexture({
      size: [PERSISTENT_TILE_SIZE, PERSISTENT_TILE_SIZE],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    });
    const colourTexture = this.device.createTexture({
      size: [PERSISTENT_TILE_SIZE, PERSISTENT_TILE_SIZE],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
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
      entries: [
        { binding: 0, resource: { buffer: colourUniform } },
        { binding: 1, resource: resultTexture.createView() },
        { binding: 2, resource: qualityTexture.createView() },
        { binding: 3, resource: colourTexture.createView() }
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
      counterReadback,
      resultTexture,
      qualityTexture,
      colourTexture,
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

  private async executeBatch(
    batch: readonly ScheduledWork[],
    palettePhase: number
  ): Promise<void> {
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
        this.createColourParameterData(palettePhase)
      );
    }

    const encoder = this.device.createCommandEncoder();
    for (const scheduled of batch) {
      const tile = scheduled.tile;
      if (scheduled.work.chunkIterations > 0) {
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
      this.acceptedAtlas.encodeCopy(
        encoder, tile.atlasSlot, tile.colourTexture, tile.qualityTexture
      );
      if (scheduled.work.chunkIterations > 0) {
        encoder.copyBufferToBuffer(
          tile.counterBuffer,
          0,
          tile.counterReadback,
          0,
          COUNTER_BYTES
        );
      }
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await Promise.all(batch.map(async scheduled => {
      if (scheduled.work.chunkIterations <= 0) return;
      const readback = scheduled.tile.counterReadback;
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(readback.getMappedRange()).slice();
      readback.unmap();
      scheduled.tile.health = {
        activePixels: values[0] ?? 0,
        escapedPixels: values[1] ?? 0,
        analyticInteriorPixels: values[2] ?? 0,
        cappedPixels: values[3] ?? 0,
        nonFinitePixels: values[4] ?? 0,
        glitchPixels: values[5] ?? 0,
        orbitExhaustedPixels: values[6] ?? 0
      };
    }));
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

  private createColourParameterData(palettePhase: number): ArrayBuffer {
    const data = new ArrayBuffer(COLOUR_PARAMETER_BYTES);
    const unsigned = new Uint32Array(data);
    const floats = new Float32Array(data);
    unsigned[0] = PERSISTENT_TILE_SIZE;
    floats[2] = palettePhase;
    floats[3] = PALETTE_LENGTH;
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
    this.acceptedAtlas.release(tile.atlasSlot);
    tile.stateBuffer.destroy();
    tile.metaBuffer.destroy();
    tile.counterBuffer.destroy();
    tile.counterReadback.destroy();
    tile.resultTexture.destroy();
    tile.qualityTexture.destroy();
    tile.colourTexture.destroy();
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
