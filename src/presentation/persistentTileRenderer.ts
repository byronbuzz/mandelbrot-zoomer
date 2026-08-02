import {
  fixedDifferenceToNumber,
  fixedSplitF32,
  fixedToNumber
} from '../bigFixed';
import { scaleToNumber } from '../binaryScale';
import type { CameraSnapshot } from '../camera/types';
import {
  persistentTileClearShader,
  persistentTileColourShader,
  persistentTileIterationShader,
  persistentTilePresentShader
} from '../numerical/persistentTileShaders';
import {
  PERSISTENT_TILE_SIZE,
  type PersistentFieldStats,
  type PersistentTileDescriptor,
  type PersistentTileHealth,
  type PersistentTileKey,
  type PersistentTileRequest,
  type PersistentTileWork
} from '../tiles/persistentTileTypes';
import {
  sampleExponentForViewport,
  tileSpanExponent,
  visibleTileDescriptors
} from '../tiles/worldTilePlanner';

const TILE_PARAMETER_BYTES = 64;
const PRESENT_PARAMETER_BYTES = 16;
const CLEAR_PARAMETER_BYTES = 16;
const COUNTER_VALUES = 5;
const COUNTER_BYTES = COUNTER_VALUES * Uint32Array.BYTES_PER_ELEMENT;
const STATE_BYTES_PER_PIXEL = 16;
const META_BYTES_PER_PIXEL = 8;
const MAX_BATCH_TILES = 4;
const MAX_CACHED_TILES = 96;
const MOVING_PIXEL_BUDGET = 720_000;
const SETTLING_PIXEL_BUDGET = 960_000;
const SETTLED_PIXEL_BUDGET = 1_200_000;
const PALETTE_LENGTH = 64;

const EMPTY_HEALTH: PersistentTileHealth = {
  activePixels: PERSISTENT_TILE_SIZE * PERSISTENT_TILE_SIZE,
  escapedPixels: 0,
  analyticInteriorPixels: 0,
  cappedPixels: 0,
  nonFinitePixels: 0
};

const EMPTY_STATS: PersistentFieldStats = {
  requestId: 0,
  interaction: 'settled',
  visibleTiles: 0,
  cachedTiles: 0,
  activeTiles: 0,
  convergedTiles: 0,
  completedChunks: 0,
  queuedChunks: 0,
  lastBatchMs: 0,
  numericalFreshnessMs: 0,
  presentationHistoryMs: 0,
  sampleExponent: 0,
  tileSize: PERSISTENT_TILE_SIZE
};

type PersistentGpuTile = {
  descriptor: PersistentTileDescriptor;
  stateBuffer: GPUBuffer;
  metaBuffer: GPUBuffer;
  counterBuffer: GPUBuffer;
  counterReadback: GPUBuffer;
  resultTexture: GPUTexture;
  qualityTexture: GPUTexture;
  colourTexture: GPUTexture;
  iterationUniform: GPUBuffer;
  presentUniform: GPUBuffer;
  iterationGroup: GPUBindGroup;
  colourGroup: GPUBindGroup;
  clearGroup: GPUBindGroup;
  presentGroup: GPUBindGroup;
  health: PersistentTileHealth;
  iterationFrontier: number;
  acceptedPixels: number;
  lastVisibleAt: number;
  lastNumericalUpdateAt: number;
  createdAt: number;
  palettePhase: number;
  mode: 0 | 1;
};

type ScheduledTileWork = Readonly<{
  work: PersistentTileWork;
  tile: PersistentGpuTile;
  scheduledEnd: number;
}>;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

export class PersistentTileRenderer {
  private readonly tileMap = new Map<PersistentTileKey, PersistentGpuTile>();
  private readonly iterationPipeline: GPUComputePipeline;
  private readonly colourPipeline: GPUComputePipeline;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly clearUniform: GPUBuffer;
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
    iterationPipeline: GPUComputePipeline,
    colourPipeline: GPUComputePipeline,
    clearPipeline: GPUComputePipeline,
    presentPipeline: GPURenderPipeline,
    readonly adapterLabel: string
  ) {
    this.iterationPipeline = iterationPipeline;
    this.colourPipeline = colourPipeline;
    this.clearPipeline = clearPipeline;
    this.presentPipeline = presentPipeline;
    this.clearUniform = device.createBuffer({
      size: CLEAR_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(
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
  }

  static async create(canvas: HTMLCanvasElement): Promise<PersistentTileRenderer> {
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

    const iterationModule = device.createShaderModule({ code: persistentTileIterationShader });
    const colourModule = device.createShaderModule({ code: persistentTileColourShader });
    const clearModule = device.createShaderModule({ code: persistentTileClearShader });
    const presentModule = device.createShaderModule({ code: persistentTilePresentShader });
    await Promise.all([
      this.assertShaderValid(iterationModule, 'persistent tile iteration'),
      this.assertShaderValid(colourModule, 'persistent tile colour'),
      this.assertShaderValid(clearModule, 'persistent tile clear'),
      this.assertShaderValid(presentModule, 'persistent tile presentation')
    ]);

    const [iterationPipeline, colourPipeline, clearPipeline, presentPipeline] = await Promise.all([
      device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: iterationModule, entryPoint: 'main' }
      }),
      device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: colourModule, entryPoint: 'main' }
      }),
      device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: clearModule, entryPoint: 'main' }
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
    return new PersistentTileRenderer(
      canvas,
      device,
      context,
      canvasFormat,
      iterationPipeline,
      colourPipeline,
      clearPipeline,
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
      .filter((tile): tile is PersistentGpuTile => Boolean(tile));
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
      cachedTiles: this.tileMap.size,
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
    const renderHeight = this.renderHeight(cssWidth, cssHeight, devicePixelRatio, 'settled');
    const aspect = Math.max(1, cssWidth) / Math.max(1, cssHeight);
    const desiredExponent = sampleExponentForViewport(targetCamera, renderHeight, 0);
    const drawTiles: PersistentGpuTile[] = [];
    const seen = new Set<PersistentTileKey>();

    for (const levelOffset of [2, 1, 0, -1]) {
      const descriptors = visibleTileDescriptors(
        targetCamera,
        aspect,
        renderHeight,
        0.5,
        0.5,
        levelOffset
      );
      for (const descriptor of descriptors) {
        const tile = this.tileMap.get(descriptor.key);
        if (!tile || seen.has(tile.descriptor.key) || tile.acceptedPixels <= 0) continue;
        seen.add(tile.descriptor.key);
        drawTiles.push(tile);
      }
    }

    if (drawTiles.length === 0) return false;
    drawTiles.sort((left, right) => right.descriptor.sampleExponent - left.descriptor.sampleExponent);
    for (const tile of drawTiles) {
      const transform = this.tileTransform(tile, targetCamera, aspect);
      if (!transform) continue;
      this.device.queue.writeBuffer(
        tile.presentUniform,
        0,
        new Float32Array([
          transform.scaleX,
          transform.scaleY,
          transform.offsetX,
          transform.offsetY
        ])
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
    for (const tile of drawTiles) {
      if (!this.tileTransform(tile, targetCamera, aspect)) continue;
      pass.setBindGroup(0, tile.presentGroup);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.statsValue = {
      ...this.statsValue,
      sampleExponent: desiredExponent
    };
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
        this.currentVisibleKeys = new Set(descriptors.map(descriptor => descriptor.key));
        await this.ensureTiles(descriptors);
        const queue = this.initialWorkQueue(request, descriptors);
        let completedChunks = 0;
        let lastBatchMs = 0;

        this.statsValue = {
          requestId: request.requestId,
          interaction: request.interaction,
          visibleTiles: descriptors.length,
          cachedTiles: this.tileMap.size,
          activeTiles: queue.length,
          convergedTiles: descriptors.length - queue.length,
          completedChunks: 0,
          queuedChunks: queue.length,
          lastBatchMs: 0,
          numericalFreshnessMs: 0,
          presentationHistoryMs: 0,
          sampleExponent: descriptors[0]?.sampleExponent ?? 0,
          tileSize: PERSISTENT_TILE_SIZE
        };

        while (queue.length > 0) {
          if (this.hasNewerRequest(request.requestId)) break;
          queue.sort((left, right) => left.work.priority - right.work.priority);
          const batch = queue.splice(0, MAX_BATCH_TILES);
          const started = performance.now();
          await this.executeBatch(batch, request.palettePhase);
          lastBatchMs = Math.max(0.1, performance.now() - started);
          completedChunks += batch.length;

          for (const scheduled of batch) {
            const tile = scheduled.tile;
            tile.iterationFrontier = Math.max(tile.iterationFrontier, scheduled.scheduledEnd);
            tile.lastNumericalUpdateAt = performance.now();
            tile.lastVisibleAt = tile.lastNumericalUpdateAt;
            tile.palettePhase = request.palettePhase;
            tile.acceptedPixels = tile.health.escapedPixels
              + tile.health.analyticInteriorPixels
              + tile.health.nonFinitePixels
              + (scheduled.work.acceptIterationCap ? tile.health.cappedPixels : 0);

            if (
              tile.health.activePixels > 0
              && tile.iterationFrontier < scheduled.work.targetIterations
              && !this.hasNewerRequest(request.requestId)
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

          const activeTiles = queue.length;
          this.statsValue = {
            ...this.statsValue,
            activeTiles,
            convergedTiles: Math.max(0, descriptors.length - activeTiles),
            completedChunks,
            queuedChunks: queue.length,
            lastBatchMs
          };
        }

        this.evictColdTiles();
      }
    } catch (error) {
      console.error('Persistent tile scheduler failed', error);
      const message = error instanceof Error ? error.message : String(error);
      this.runtimeErrorListener?.(message);
    } finally {
      this.running = false;
      if (this.latestRequest) void this.pump();
    }
  }

  private initialWorkQueue(
    request: PersistentTileRequest,
    descriptors: readonly PersistentTileDescriptor[]
  ): ScheduledTileWork[] {
    const targetIterations = request.interaction === 'moving'
      ? Math.min(request.targetIterations, 256)
      : request.interaction === 'settling'
        ? Math.min(request.targetIterations, 768)
        : request.targetIterations;
    const chunkIterations = request.interaction === 'moving'
      ? 64
      : request.interaction === 'settling' ? 96 : 128;
    const acceptIterationCap = request.interaction === 'settled';
    const queue: ScheduledTileWork[] = [];

    for (const descriptor of descriptors) {
      const tile = this.tileMap.get(descriptor.key);
      if (!tile) continue;
      tile.lastVisibleAt = performance.now();
      const needsIterations = tile.iterationFrontier < targetIterations;
      const needsRecolour = Math.abs(tile.palettePhase - request.palettePhase) > 1e-6;
      if (!needsIterations && !needsRecolour) continue;
      const work: PersistentTileWork = {
        requestId: request.requestId,
        key: descriptor.key,
        targetIterations,
        chunkIterations: needsIterations ? chunkIterations : 0,
        acceptIterationCap,
        priority: descriptor.distanceFromFocus
          + (tile.acceptedPixels > 0 ? 0.3 : 0)
          + Math.min(2, Math.max(0, performance.now() - tile.lastNumericalUpdateAt) / 5000)
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

  private async ensureTiles(descriptors: readonly PersistentTileDescriptor[]): Promise<void> {
    const created: PersistentGpuTile[] = [];
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

  private createTile(descriptor: PersistentTileDescriptor): PersistentGpuTile {
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
    const iterationUniform = this.device.createBuffer({
      size: TILE_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const presentUniform = this.device.createBuffer({
      size: PRESENT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const iterationGroup = this.device.createBindGroup({
      layout: this.iterationPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: iterationUniform } },
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
        { binding: 0, resource: { buffer: iterationUniform } },
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
    const presentGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: presentUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: colourTexture.createView() }
      ]
    });

    const centerMagnitude = Math.max(
      1,
      Math.abs(fixedToNumber(descriptor.centerX)),
      Math.abs(fixedToNumber(descriptor.centerY))
    );
    const sampleStep = Math.pow(2, descriptor.sampleExponent);
    const mode: 0 | 1 = sampleStep > centerMagnitude * Math.pow(2, -21) ? 0 : 1;
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
      iterationUniform,
      presentUniform,
      iterationGroup,
      colourGroup,
      clearGroup,
      presentGroup,
      health: EMPTY_HEALTH,
      iterationFrontier: 0,
      acceptedPixels: 0,
      lastVisibleAt: now,
      lastNumericalUpdateAt: 0,
      createdAt: now,
      palettePhase: Number.NaN,
      mode
    };
  }

  private async executeBatch(
    batch: readonly ScheduledTileWork[],
    palettePhase: number
  ): Promise<void> {
    for (const scheduled of batch) {
      this.device.queue.writeBuffer(
        scheduled.tile.iterationUniform,
        0,
        this.createTileParameterData(scheduled, palettePhase)
      );
    }

    const encoder = this.device.createCommandEncoder();
    for (const scheduled of batch) {
      const tile = scheduled.tile;
      encoder.clearBuffer(tile.counterBuffer);
      if (scheduled.work.chunkIterations > 0) {
        const iterationPass = encoder.beginComputePass();
        iterationPass.setPipeline(this.iterationPipeline);
        iterationPass.setBindGroup(0, tile.iterationGroup);
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
      encoder.copyBufferToBuffer(
        tile.counterBuffer,
        0,
        tile.counterReadback,
        0,
        COUNTER_BYTES
      );
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await Promise.all(batch.map(async scheduled => {
      const readback = scheduled.tile.counterReadback;
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(readback.getMappedRange()).slice();
      readback.unmap();
      scheduled.tile.health = {
        activePixels: values[0] ?? 0,
        escapedPixels: values[1] ?? 0,
        analyticInteriorPixels: values[2] ?? 0,
        cappedPixels: values[3] ?? 0,
        nonFinitePixels: values[4] ?? 0
      };
    }));
  }

  private createTileParameterData(
    scheduled: ScheduledTileWork,
    palettePhase: number
  ): ArrayBuffer {
    const data = new ArrayBuffer(TILE_PARAMETER_BYTES);
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
    unsigned[8] = tile.mode;
    unsigned[9] = scheduled.work.acceptIterationCap ? 1 : 0;
    floats[10] = palettePhase;
    floats[11] = PALETTE_LENGTH;
    return data;
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
    tile: PersistentGpuTile,
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

  private destroyTile(tile: PersistentGpuTile): void {
    tile.stateBuffer.destroy();
    tile.metaBuffer.destroy();
    tile.counterBuffer.destroy();
    tile.counterReadback.destroy();
    tile.resultTexture.destroy();
    tile.qualityTexture.destroy();
    tile.colourTexture.destroy();
    tile.iterationUniform.destroy();
    tile.presentUniform.destroy();
  }

  private hasNewerRequest(requestId: number): boolean {
    return Boolean(this.latestRequest && this.latestRequest.requestId > requestId);
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
