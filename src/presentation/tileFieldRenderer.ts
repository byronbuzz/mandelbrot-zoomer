import {
  fixedDifferenceToNumber,
  fixedSplitF32,
  fixedSub,
  fixedToNumber
} from '../bigFixed';
import { scaleToNumber } from '../binaryScale';
import type { CameraSnapshot } from '../camera/types';
import {
  tileClearShader,
  tileColourShader,
  tileDirectIterationShader,
  tilePerturbationShader,
  tilePresentShader,
  tileResetNumericalShader
} from '../numerical/tileFieldShaders';
import {
  TileReferenceAtlas,
  type TileGpuReference
} from '../references/tileReferenceAtlas';
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
const MAX_BATCH_TILES = 4;
const MAX_CACHED_TILES = 96;
const MOVING_PIXEL_BUDGET = 720_000;
const SETTLING_PIXEL_BUDGET = 960_000;
const SETTLED_PIXEL_BUDGET = 1_200_000;
const MOVING_DIRECT_ITERATIONS = 256;
const SETTLING_DIRECT_ITERATIONS = 768;
const DIRECT_SAFETY_ITERATIONS = 256;
const PERTURBATION_POLICY_ITERATIONS = 384;
const MAX_REFERENCE_REPAIR_PASSES = 4;
const GLITCH_RATIO = 1e-6;
const PALETTE_LENGTH = 64;

const EMPTY_HEALTH: PersistentTileHealth = {
  activePixels: PERSISTENT_TILE_SIZE * PERSISTENT_TILE_SIZE,
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
  presentGroup: GPUBindGroup;
  health: PersistentTileHealth;
  iterationFrontier: number;
  acceptedPixels: number;
  acceptIterationCap: boolean;
  lastVisibleAt: number;
  lastNumericalUpdateAt: number;
  createdAt: number;
  palettePhase: number;
  directMode: 0 | 1;
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
}>;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

export class TileFieldRenderer {
  private readonly tileMap = new Map<PersistentTileKey, FieldTile>();
  private readonly directPipeline: GPUComputePipeline;
  private readonly perturbPipeline: GPUComputePipeline;
  private readonly colourPipeline: GPUComputePipeline;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly resetPipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly clearUniform: GPUBuffer;
  private readonly referenceAtlas: TileReferenceAtlas;
  private latestRequest: PersistentTileRequest | null = null;
  private currentRequest: PersistentTileRequest | null = null;
  private currentVisibleKeys = new Set<PersistentTileKey>();
  private running = false;
  private displayWidth = 1;
  private displayHeight = 1;
  private statsValue: PersistentFieldStats = EMPTY_STATS;
  private runtimeErrorListener: ((message: string) => void) | null = null;

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
    readonly adapterLabel: string
  ) {
    this.directPipeline = directPipeline;
    this.perturbPipeline = perturbPipeline;
    this.colourPipeline = colourPipeline;
    this.clearPipeline = clearPipeline;
    this.resetPipeline = resetPipeline;
    this.presentPipeline = presentPipeline;
    this.clearUniform = device.createBuffer({
      size: CLEAR_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(
      this.clearUniform,
      0,
      new Uint32Array([PERSISTENT_TILE_SIZE, 0, 0, 0])
    );
    this.sampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
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
      label
    );
  }

  onRuntimeError(listener: (message: string) => void): void {
    this.runtimeErrorListener = listener;
  }

  onDeviceError(listener: (message: string) => void): void {
    this.device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
      listener(`WebGPU error: ${event.error.message}`);
    });
    void this.device.lost.then((reason: { message?: string; reason?: string }) => {
      listener(`GPU device lost: ${reason.message || reason.reason}`);
    });
  }

  request(request: PersistentTileRequest): void {
    this.latestRequest = request;
    if (!this.running) void this.pump();
  }

  get isBusy(): boolean {
    return this.running || this.latestRequest !== null;
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
      pendingReferences: [...this.tileMap.values()]
        .filter(tile => tile.referenceState === 'queued').length,
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
    if (this.tileMap.size === 0) return false;
    const width = Math.max(1, Math.floor(cssWidth * devicePixelRatio));
    const height = Math.max(1, Math.floor(cssHeight * devicePixelRatio));
    this.resizeCanvas(width, height);
    const interaction = this.currentRequest?.interaction ?? 'settled';
    const renderHeight = this.renderHeight(cssWidth, cssHeight, devicePixelRatio, interaction);
    const aspect = Math.max(1, cssWidth) / Math.max(1, cssHeight);
    const drawTiles: FieldTile[] = [];
    const seen = new Set<PersistentTileKey>();

    for (const levelOffset of [3, 2, 1, 0, -1]) {
      for (const descriptor of visibleTileDescriptors(
        targetCamera,
        aspect,
        renderHeight,
        0.5,
        0.5,
        levelOffset
      )) {
        const tile = this.tileMap.get(descriptor.key);
        if (!tile || seen.has(tile.descriptor.key) || tile.acceptedPixels <= 0) continue;
        seen.add(tile.descriptor.key);
        drawTiles.push(tile);
      }
    }
    if (drawTiles.length === 0) return false;
    drawTiles.sort((left, right) => right.descriptor.sampleExponent - left.descriptor.sampleExponent);

    const renderable: FieldTile[] = [];
    for (const tile of drawTiles) {
      const transform = this.tileTransform(tile, targetCamera, aspect);
      if (!transform) continue;
      this.device.queue.writeBuffer(tile.presentUniform, 0, new Float32Array([
        transform.scaleX,
        transform.scaleY,
        transform.offsetX,
        transform.offsetY
      ]));
      renderable.push(tile);
    }
    if (renderable.length === 0) return false;

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
    for (const tile of renderable) {
      pass.setBindGroup(0, tile.presentGroup);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  private async pump(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.latestRequest) {
        const request = this.latestRequest;
        this.latestRequest = null;
        this.currentRequest = request;
        const renderHeight = this.renderHeight(
          request.cssWidth,
          request.cssHeight,
          request.devicePixelRatio,
          request.interaction
        );
        const aspect = Math.max(1, request.cssWidth) / Math.max(1, request.cssHeight);
        const levelOffset = request.interaction === 'moving' ? 1 : 0;
        const descriptors = visibleTileDescriptors(
          request.camera,
          aspect,
          renderHeight,
          request.focusX,
          request.focusY,
          levelOffset
        );
        await this.ensureTiles(descriptors);
        await this.activatePendingReferences();
        this.currentVisibleKeys = this.protectedVisibleKeys(request, renderHeight, aspect, descriptors);
        this.ensureReferencePolicy(request, descriptors);
        const queue = this.initialWorkQueue(request, descriptors);
        let completedChunks = 0;
        let lastBatchMs = 0;

        this.updateStats(request, descriptors, queue, completedChunks, lastBatchMs);
        while (queue.length > 0) {
          if (this.latestRequest) break;
          queue.sort((left, right) => left.work.priority - right.work.priority);
          const batch = queue.splice(0, MAX_BATCH_TILES);
          const started = performance.now();
          await this.executeBatch(batch, request.palettePhase);
          lastBatchMs = Math.max(0.1, performance.now() - started);
          completedChunks += batch.length;

          for (const scheduled of batch) {
            const tile = scheduled.tile;
            tile.iterationFrontier = Math.max(tile.iterationFrontier, scheduled.scheduledEnd);
            if (scheduled.work.chunkIterations > 0) tile.lastNumericalUpdateAt = performance.now();
            tile.lastVisibleAt = performance.now();
            tile.palettePhase = request.palettePhase;
            tile.acceptIterationCap = scheduled.work.acceptIterationCap;
            tile.acceptedPixels = tile.health.escapedPixels
              + tile.health.analyticInteriorPixels
              + (tile.acceptIterationCap ? tile.health.cappedPixels : 0);

            this.maybeRequestRepair(tile, request);
            if (
              scheduled.work.chunkIterations > 0
              && tile.health.activePixels > 0
              && tile.iterationFrontier < scheduled.work.targetIterations
              && !this.latestRequest
            ) {
              queue.push({
                work: {
                  ...scheduled.work,
                  priority: scheduled.work.priority + 0.125
                },
                tile,
                scheduledEnd: Math.min(
                  scheduled.work.targetIterations,
                  tile.iterationFrontier + scheduled.work.chunkIterations
                )
              });
            }
          }
          this.updateStats(request, descriptors, queue, completedChunks, lastBatchMs);
        }
        this.evictColdTiles();
      }
    } catch (error) {
      console.error('Tile field scheduler failed', error);
      const message = error instanceof Error ? error.message : String(error);
      this.runtimeErrorListener?.(message);
    } finally {
      this.running = false;
      if (this.latestRequest) void this.pump();
    }
  }

  private updateStats(
    request: PersistentTileRequest,
    descriptors: readonly PersistentTileDescriptor[],
    queue: readonly ScheduledWork[],
    completedChunks: number,
    lastBatchMs: number
  ): void {
    const visible = descriptors
      .map(descriptor => this.tileMap.get(descriptor.key))
      .filter((tile): tile is FieldTile => Boolean(tile));
    const tilePixelCount = PERSISTENT_TILE_SIZE * PERSISTENT_TILE_SIZE;
    this.statsValue = {
      requestId: request.requestId,
      interaction: request.interaction,
      visibleTiles: visible.length,
      cachedTiles: this.tileMap.size,
      activeTiles: visible.filter(tile => tile.acceptedPixels < tilePixelCount).length,
      convergedTiles: visible.filter(tile => tile.acceptedPixels >= tilePixelCount).length,
      directTiles: visible.filter(tile => tile.numericalMode !== 'perturbation').length,
      perturbationTiles: visible.filter(tile => tile.numericalMode === 'perturbation').length,
      pendingReferences: visible.filter(tile => tile.referenceState === 'queued').length,
      repairTiles: visible.filter(tile => tile.repairPass > 0).length,
      referenceFailures: this.referenceAtlas.failureCount,
      completedChunks,
      queuedChunks: queue.length,
      lastBatchMs,
      numericalFreshnessMs: 0,
      presentationHistoryMs: 0,
      sampleExponent: descriptors[0]?.sampleExponent ?? 0,
      tileSize: PERSISTENT_TILE_SIZE
    };
  }

  private ensureReferencePolicy(
    request: PersistentTileRequest,
    descriptors: readonly PersistentTileDescriptor[]
  ): void {
    const referenceTarget = this.referenceTargetForRequest(request);
    for (const descriptor of descriptors) {
      const tile = this.tileMap.get(descriptor.key);
      if (!tile || !this.tileNeedsPerturbation(tile, request)) continue;
      if (tile.referenceState === 'queued') continue;

      const failurePixels = tile.health.glitchPixels
        + tile.health.orbitExhaustedPixels
        + tile.health.nonFinitePixels;
      const unresolvedPixels = tile.health.activePixels
        + tile.health.cappedPixels
        + failurePixels;

      if (tile.numericalMode === 'perturbation' && tile.reference) {
        // GPU-detected failures are repaired by maybeRequestRepair after the batch.
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

      this.queueReference(tile, request, tile.repairPass, referenceTarget);
    }
  }

  private referenceTargetForRequest(request: PersistentTileRequest): number {
    if (request.interaction === 'moving') {
      return Math.min(request.targetIterations, MOVING_DIRECT_ITERATIONS);
    }
    if (request.interaction === 'settling') {
      return Math.min(request.targetIterations, SETTLING_DIRECT_ITERATIONS);
    }
    return request.targetIterations;
  }

  private tileNeedsPerturbation(tile: FieldTile, request: PersistentTileRequest): boolean {
    if (tile.numericalMode === 'perturbation') return true;
    if (tile.directMode === 0) return false;
    if (tile.health.nonFinitePixels > 0) return true;
    return request.targetIterations >= PERTURBATION_POLICY_ITERATIONS;
  }

  private queueReference(
    tile: FieldTile,
    request: PersistentTileRequest,
    repairPass: number,
    referenceTarget = this.referenceTargetForRequest(request)
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
    const referenceTarget = Math.max(tile.referenceTarget, this.referenceTargetForRequest(request));
    if (tile.referenceState === 'failed' && tile.referenceTarget >= referenceTarget) return;
    if (tile.repairPass >= MAX_REFERENCE_REPAIR_PASSES) {
      tile.referenceState = 'failed';
      tile.referenceTarget = referenceTarget;
      return;
    }
    this.queueReference(tile, request, tile.repairPass + 1, referenceTarget);
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
        activePixels: Math.max(0, PERSISTENT_TILE_SIZE * PERSISTENT_TILE_SIZE
          - (preserveAccepted ? tile.acceptedPixels : 0))
      };
      if (!preserveAccepted) tile.acceptedPixels = 0;
      tile.acceptIterationCap = false;
      tile.lastNumericalUpdateAt = performance.now();
    }
  }

  private initialWorkQueue(
    request: PersistentTileRequest,
    descriptors: readonly PersistentTileDescriptor[]
  ): ScheduledWork[] {
    const queue: ScheduledWork[] = [];
    for (const descriptor of descriptors) {
      const tile = this.tileMap.get(descriptor.key);
      if (!tile) continue;
      tile.lastVisibleAt = performance.now();
      const targetIterations = this.effectiveTarget(tile, request);
      const chunkIterations = request.interaction === 'moving'
        ? 64
        : request.interaction === 'settling' ? 96 : 128;
      const acceptIterationCap = this.acceptIterationCap(tile, request);
      const needsIterations = tile.iterationFrontier < targetIterations;
      const needsRecolour = Math.abs(tile.palettePhase - request.palettePhase) > 1e-6;
      const needsAcceptanceUpdate = tile.acceptIterationCap !== acceptIterationCap;
      if (!needsIterations && !needsRecolour && !needsAcceptanceUpdate) continue;
      const work: PersistentTileWork = {
        requestId: request.requestId,
        key: descriptor.key,
        targetIterations,
        chunkIterations: needsIterations ? chunkIterations : 0,
        acceptIterationCap,
        priority: descriptor.distanceFromFocus
          + (tile.acceptedPixels > 0 ? 0.25 : 0)
          + (tile.numericalMode === 'perturbation' ? -0.1 : 0)
      };
      queue.push({
        work,
        tile,
        scheduledEnd: needsIterations
          ? Math.min(targetIterations, tile.iterationFrontier + chunkIterations)
          : tile.iterationFrontier
      });
    }
    return queue;
  }

  private effectiveTarget(tile: FieldTile, request: PersistentTileRequest): number {
    const interactionTarget = request.interaction === 'moving'
      ? Math.min(request.targetIterations, MOVING_DIRECT_ITERATIONS)
      : request.interaction === 'settling'
        ? Math.min(request.targetIterations, SETTLING_DIRECT_ITERATIONS)
        : request.targetIterations;
    if (tile.numericalMode === 'perturbation') return interactionTarget;
    if (this.tileNeedsPerturbation(tile, request)) {
      return Math.min(interactionTarget, DIRECT_SAFETY_ITERATIONS);
    }
    return interactionTarget;
  }

  private acceptIterationCap(tile: FieldTile, request: PersistentTileRequest): boolean {
    if (request.interaction !== 'settled') return false;
    if (tile.numericalMode === 'perturbation') return true;
    return !this.tileNeedsPerturbation(tile, request);
  }

  private async ensureTiles(descriptors: readonly PersistentTileDescriptor[]): Promise<void> {
    const created: FieldTile[] = [];
    for (const descriptor of descriptors) {
      if (this.tileMap.has(descriptor.key)) continue;
      const tile = this.createTile(descriptor);
      this.tileMap.set(descriptor.key, tile);
      created.push(tile);
    }
    if (created.length === 0) return;

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

  private createTile(descriptor: PersistentTileDescriptor): FieldTile {
    const pixelCount = PERSISTENT_TILE_SIZE * PERSISTENT_TILE_SIZE;
    const stateBuffer = this.device.createBuffer({
      size: pixelCount * STATE_BYTES_PER_PIXEL,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const metaBuffer = this.device.createBuffer({
      size: pixelCount * META_BYTES_PER_PIXEL,
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
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    const qualityTexture = this.device.createTexture({
      size: [PERSISTENT_TILE_SIZE, PERSISTENT_TILE_SIZE],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    const colourTexture = this.device.createTexture({
      size: [PERSISTENT_TILE_SIZE, PERSISTENT_TILE_SIZE],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
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
        { binding: 2, resource: { buffer: metaBuffer } },
        { binding: 3, resource: resultTexture.createView() },
        { binding: 4, resource: qualityTexture.createView() }
      ]
    });
    const presentGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: presentUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: colourTexture.createView() },
        { binding: 3, resource: qualityTexture.createView() }
      ]
    });

    const centerMagnitude = Math.max(
      1,
      Math.abs(fixedToNumber(descriptor.centerX)),
      Math.abs(fixedToNumber(descriptor.centerY))
    );
    const sampleStep = Math.pow(2, descriptor.sampleExponent);
    const directMode: 0 | 1 = sampleStep > centerMagnitude * Math.pow(2, -21) ? 0 : 1;
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
      presentGroup,
      health: EMPTY_HEALTH,
      iterationFrontier: 0,
      acceptedPixels: 0,
      acceptIterationCap: false,
      lastVisibleAt: now,
      lastNumericalUpdateAt: 0,
      createdAt: now,
      palettePhase: Number.NaN,
      directMode,
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
    unsigned[6] = scheduled.work.targetIterations;
    unsigned[7] = scheduled.work.chunkIterations;
    unsigned[8] = tile.directMode;
    unsigned[9] = scheduled.work.acceptIterationCap ? 1 : 0;
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
    unsigned[10] = scheduled.work.targetIterations;
    unsigned[11] = scheduled.work.chunkIterations;
    unsigned[12] = reference.length;
    unsigned[13] = scheduled.work.acceptIterationCap ? 1 : 0;
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

  private protectedVisibleKeys(
    request: PersistentTileRequest,
    renderHeight: number,
    aspect: number,
    primary: readonly PersistentTileDescriptor[]
  ): Set<PersistentTileKey> {
    const keys = new Set(primary.map(descriptor => descriptor.key));
    for (const offset of [1, 2, 3]) {
      for (const descriptor of visibleTileDescriptors(
        request.camera,
        aspect,
        renderHeight,
        request.focusX,
        request.focusY,
        offset
      )) {
        if (this.tileMap.has(descriptor.key)) keys.add(descriptor.key);
      }
    }
    return keys;
  }

  private renderHeight(
    cssWidth: number,
    cssHeight: number,
    devicePixelRatio: number,
    interaction: PersistentTileRequest['interaction']
  ): number {
    const dpr = clamp(devicePixelRatio, 1, 2);
    const requestedWidth = Math.max(1, cssWidth * dpr);
    const requestedHeight = Math.max(1, cssHeight * dpr);
    const pixelBudget = interaction === 'moving'
      ? MOVING_PIXEL_BUDGET
      : interaction === 'settling' ? SETTLING_PIXEL_BUDGET : SETTLED_PIXEL_BUDGET;
    const scale = Math.min(1, Math.sqrt(pixelBudget / (requestedWidth * requestedHeight)));
    return Math.max(1, Math.floor(requestedHeight * scale));
  }

  private tileTransform(
    tile: FieldTile,
    targetCamera: CameraSnapshot,
    aspect: number
  ): { scaleX: number; scaleY: number; offsetX: number; offsetY: number } | null {
    const tileSpan = Math.pow(2, tileSpanExponent(tile.descriptor.sampleExponent));
    const viewportHeight = scaleToNumber(targetCamera.scale);
    const transform = {
      scaleX: viewportHeight * aspect / tileSpan,
      scaleY: viewportHeight / tileSpan,
      offsetX: fixedDifferenceToNumber(targetCamera.centerX, tile.descriptor.centerX) / tileSpan,
      offsetY: fixedDifferenceToNumber(targetCamera.centerY, tile.descriptor.centerY) / tileSpan
    };
    return Object.values(transform).every(value => Number.isFinite(value) && Math.abs(value) <= 3.3e38)
      ? transform
      : null;
  }

  private evictColdTiles(): void {
    if (this.tileMap.size <= MAX_CACHED_TILES) return;
    const candidates = [...this.tileMap.values()]
      .filter(tile => !this.currentVisibleKeys.has(tile.descriptor.key))
      .sort((left, right) => left.lastVisibleAt - right.lastVisibleAt);
    while (this.tileMap.size > MAX_CACHED_TILES && candidates.length > 0) {
      const tile = candidates.shift();
      if (!tile) break;
      this.tileMap.delete(tile.descriptor.key);
      this.destroyTile(tile);
    }
  }

  private destroyTile(tile: FieldTile): void {
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
