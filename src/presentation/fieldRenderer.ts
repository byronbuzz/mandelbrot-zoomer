import { fixedDifferenceToNumber, fixedSplitF32, fixedToNumber } from '../bigFixed';
import { scaleToNumber } from '../binaryScale';
import type { CameraSnapshot } from '../camera/types';
import {
  clearFieldShader,
  colourShader,
  directOrbitShader,
  presentShader,
  seedShader
} from '../numerical/shaders';
import { AdaptiveNavigationBudget } from '../scheduler/adaptiveBudget';
import { planFieldJobs } from '../tiles/jobPlanner';
import type {
  FieldJob,
  FieldPhase,
  FieldStats,
  NavigationProfile,
  PrecisionMode,
  RenderRequest
} from '../tiles/types';

const RENDER_PARAMETER_BYTES = 96;
const SEED_PARAMETER_BYTES = 32;
const CLEAR_PARAMETER_BYTES = 16;
const PRESENT_PARAMETER_BYTES = 16;
const RESULT_FORMAT: GPUTextureFormat = 'rgba32float';
const COLOUR_FORMAT: GPUTextureFormat = 'rgba8unorm';
const MAX_BATCH_JOBS = 4;
const MAX_POOL_RESOURCES = 4;
const MOVING_MAX_PIXELS = 2_600_000;
const SETTLED_MAX_PIXELS = 5_200_000;
const PALETTE_LENGTH = 64;

const EMPTY_STATS: FieldStats = {
  requestId: 0,
  interaction: 'settled',
  phase: 'seed',
  completedJobs: 0,
  totalJobs: 0,
  lastBatchMs: 0,
  batchJobs: 0,
  publishedJobs: 0,
  precision: 'f32-direct',
  renderWidth: 1,
  renderHeight: 1,
  navigationResolution: 1,
  navigationIterations: 0,
  navigationBlockSize: 4,
  fieldAgeMs: 0,
  anchorGeneration: 0
};

type FieldResources = Readonly<{
  key: string;
  resultTexture: GPUTexture;
  colourTexture: GPUTexture;
  qualityTexture: GPUTexture;
}>;

type Field = {
  request: RenderRequest;
  width: number;
  height: number;
  precision: PrecisionMode;
  resources: FieldResources;
  orbitGroups: GPUBindGroup[];
  colourGroups: GPUBindGroup[];
  presentGroup: GPUBindGroup;
  createdAt: number;
  publishedJobs: number;
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

export class FieldRenderer {
  private readonly renderUniforms: GPUBuffer[];
  private readonly seedUniform: GPUBuffer;
  private readonly clearUniform: GPUBuffer;
  private readonly presentUniform: GPUBuffer;
  private readonly linearSampler: GPUSampler;
  private readonly orbitPipeline: GPUComputePipeline;
  private readonly colourPipeline: GPUComputePipeline;
  private readonly seedPipeline: GPUComputePipeline;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly navigationBudget = new AdaptiveNavigationBudget();
  private readonly resourcePool = new Map<string, FieldResources[]>();
  private activeField: Field | null = null;
  private latestRequest: RenderRequest | null = null;
  private running = false;
  private displayWidth = 1;
  private displayHeight = 1;
  private statsValue: FieldStats = EMPTY_STATS;
  private runtimeErrorListener: ((message: string) => void) | null = null;

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly canvasFormat: GPUTextureFormat,
    orbitPipeline: GPUComputePipeline,
    colourPipeline: GPUComputePipeline,
    seedPipeline: GPUComputePipeline,
    clearPipeline: GPUComputePipeline,
    presentPipeline: GPURenderPipeline,
    readonly adapterLabel: string
  ) {
    this.orbitPipeline = orbitPipeline;
    this.colourPipeline = colourPipeline;
    this.seedPipeline = seedPipeline;
    this.clearPipeline = clearPipeline;
    this.presentPipeline = presentPipeline;
    this.renderUniforms = Array.from({ length: MAX_BATCH_JOBS }, () => device.createBuffer({
      size: RENDER_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    }));
    this.seedUniform = device.createBuffer({
      size: SEED_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.clearUniform = device.createBuffer({
      size: CLEAR_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.presentUniform = device.createBuffer({
      size: PRESENT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.linearSampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
  }

  static async create(canvas: HTMLCanvasElement): Promise<FieldRenderer> {
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

    const orbitModule = device.createShaderModule({ code: directOrbitShader });
    const colourModule = device.createShaderModule({ code: colourShader });
    const seedModule = device.createShaderModule({ code: seedShader });
    const clearModule = device.createShaderModule({ code: clearFieldShader });
    const presentModule = device.createShaderModule({ code: presentShader });
    await Promise.all([
      this.assertShaderValid(orbitModule, 'direct orbit'),
      this.assertShaderValid(colourModule, 'colour'),
      this.assertShaderValid(seedModule, 'field reprojection'),
      this.assertShaderValid(clearModule, 'field clear'),
      this.assertShaderValid(presentModule, 'presentation')
    ]);

    const [orbitPipeline, colourPipeline, seedPipeline, clearPipeline, presentPipeline] = await Promise.all([
      device.createComputePipelineAsync({ layout: 'auto', compute: { module: orbitModule, entryPoint: 'main' } }),
      device.createComputePipelineAsync({ layout: 'auto', compute: { module: colourModule, entryPoint: 'main' } }),
      device.createComputePipelineAsync({ layout: 'auto', compute: { module: seedModule, entryPoint: 'main' } }),
      device.createComputePipelineAsync({ layout: 'auto', compute: { module: clearModule, entryPoint: 'main' } }),
      device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: presentModule, entryPoint: 'vertexMain' },
        fragment: {
          module: presentModule,
          entryPoint: 'fragmentMain',
          targets: [{ format: canvasFormat }]
        },
        primitive: { topology: 'triangle-list' }
      })
    ]);

    const adapterLabel = adapter.info.vendor || adapter.info.description || 'GPU';
    return new FieldRenderer(
      canvas,
      device,
      context,
      canvasFormat,
      orbitPipeline,
      colourPipeline,
      seedPipeline,
      clearPipeline,
      presentPipeline,
      adapterLabel
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

  request(request: RenderRequest): void {
    this.latestRequest = request;
    if (!this.running) void this.pump();
  }

  present(targetCamera: CameraSnapshot, cssWidth: number, cssHeight: number, devicePixelRatio: number): boolean {
    const field = this.activeField;
    if (!field) return false;
    const width = Math.max(1, Math.floor(cssWidth * devicePixelRatio));
    const height = Math.max(1, Math.floor(cssHeight * devicePixelRatio));
    this.resizeCanvas(width, height);
    const transform = this.cameraTransform(
      field.request.camera,
      targetCamera,
      field.request.cssWidth,
      field.request.cssHeight,
      cssWidth,
      cssHeight
    );
    if (!transform) return false;
    this.device.queue.writeBuffer(this.presentUniform, 0, new Float32Array([
      transform.scaleX,
      transform.scaleY,
      transform.offsetX,
      transform.offsetY
    ]));

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
    pass.setBindGroup(0, field.presentGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  get stats(): FieldStats {
    const field = this.activeField;
    return {
      ...this.statsValue,
      fieldAgeMs: field ? Math.max(0, performance.now() - field.createdAt) : 0
    };
  }

  get isBusy(): boolean { return this.running || this.latestRequest !== null; }

  private async pump(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.latestRequest) {
        const request = this.latestRequest;
        this.latestRequest = null;
        const navigation = this.navigationBudget.profile(request.targetIterations);
        const dimensions = this.fieldDimensions(request, navigation);
        let field = this.activeField;
        if (!field || !this.canReuse(field, request, dimensions.width, dimensions.height)) {
          const nextField = await this.createSeededField(request, dimensions.width, dimensions.height);
          const superseded = this.hasNewerRequest(request.requestId);
          if (superseded && this.activeField) {
            this.releaseField(nextField);
            continue;
          }
          const previous = this.activeField;
          this.activeField = nextField;
          field = nextField;
          if (previous) this.retireField(previous);
        } else {
          field.request = request;
        }

        const jobs = planFieldJobs(request, field.width, field.height, navigation);
        this.statsValue = {
          requestId: request.requestId,
          interaction: request.interaction,
          phase: jobs[0]?.phase ?? 'complete',
          completedJobs: 0,
          totalJobs: jobs.length,
          lastBatchMs: 0,
          batchJobs: 0,
          publishedJobs: field.publishedJobs,
          precision: field.precision,
          renderWidth: field.width,
          renderHeight: field.height,
          navigationResolution: navigation.resolutionScale,
          navigationIterations: navigation.iterations,
          navigationBlockSize: navigation.blockSize,
          fieldAgeMs: 0,
          anchorGeneration: request.camera.generation
        };

        let nextJob = 0;
        while (nextJob < jobs.length) {
          if (this.hasNewerRequest(request.requestId)) break;
          const batchCount = this.batchCount(jobs[nextJob]);
          const batch = jobs.slice(nextJob, nextJob + batchCount);
          const started = performance.now();
          await this.executeBatch(field, batch);
          const batchMs = Math.max(0.1, performance.now() - started);
          nextJob += batch.length;
          field.publishedJobs += batch.length;
          if (request.interaction === 'moving') {
            this.navigationBudget.observe(batchMs, request.targetIterations);
          }
          const phase: FieldPhase = nextJob >= jobs.length ? 'complete' : jobs[nextJob]?.phase ?? 'complete';
          this.statsValue = {
            ...this.statsValue,
            phase,
            completedJobs: nextJob,
            lastBatchMs: batchMs,
            batchJobs: batch.length,
            publishedJobs: field.publishedJobs
          };
        }
      }
    } catch (error) {
      console.error('WebGPU field scheduler failed', error);
      const message = error instanceof Error ? error.message : String(error);
      this.runtimeErrorListener?.(message);
    } finally {
      this.running = false;
      if (this.latestRequest) void this.pump();
    }
  }

  private async createSeededField(request: RenderRequest, width: number, height: number): Promise<Field> {
    const resources = this.acquireResources(width, height);
    const precision = this.precisionFor(request.camera, height);
    const orbitGroups = this.renderUniforms.map(buffer => this.device.createBindGroup({
      layout: this.orbitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: resources.resultTexture.createView() }
      ]
    }));
    const colourGroups = this.renderUniforms.map(buffer => this.device.createBindGroup({
      layout: this.colourPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: resources.resultTexture.createView() },
        { binding: 2, resource: resources.colourTexture.createView() },
        { binding: 3, resource: resources.qualityTexture.createView() }
      ]
    }));
    const presentGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.presentUniform } },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: resources.colourTexture.createView() }
      ]
    });
    const field: Field = {
      request,
      width,
      height,
      precision,
      resources,
      orbitGroups,
      colourGroups,
      presentGroup,
      createdAt: performance.now(),
      publishedJobs: 0
    };
    await this.seedField(field, this.activeField);
    return field;
  }

  private async seedField(destination: Field, source: Field | null): Promise<void> {
    const encoder = this.device.createCommandEncoder();
    if (source) {
      const transform = this.cameraTransform(
        source.request.camera,
        destination.request.camera,
        source.request.cssWidth,
        source.request.cssHeight,
        destination.request.cssWidth,
        destination.request.cssHeight
      );
      if (transform) {
        const data = new ArrayBuffer(SEED_PARAMETER_BYTES);
        const unsigned = new Uint32Array(data);
        const floats = new Float32Array(data);
        unsigned[0] = destination.width;
        unsigned[1] = destination.height;
        floats[4] = transform.scaleX;
        floats[5] = transform.scaleY;
        floats[6] = transform.offsetX;
        floats[7] = transform.offsetY;
        this.device.queue.writeBuffer(this.seedUniform, 0, data);
        const bindGroup = this.device.createBindGroup({
          layout: this.seedPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.seedUniform } },
            { binding: 1, resource: this.linearSampler },
            { binding: 2, resource: source.resources.colourTexture.createView() },
            { binding: 3, resource: destination.resources.colourTexture.createView() },
            { binding: 4, resource: destination.resources.qualityTexture.createView() }
          ]
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.seedPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(destination.width / 8), Math.ceil(destination.height / 8));
        pass.end();
      } else {
        this.encodeClear(encoder, destination);
      }
    } else {
      this.encodeClear(encoder, destination);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  private encodeClear(encoder: GPUCommandEncoder, destination: Field): void {
    const data = new Uint32Array([destination.width, destination.height, 0, 0]);
    this.device.queue.writeBuffer(this.clearUniform, 0, data);
    const bindGroup = this.device.createBindGroup({
      layout: this.clearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.clearUniform } },
        { binding: 1, resource: destination.resources.colourTexture.createView() },
        { binding: 2, resource: destination.resources.qualityTexture.createView() }
      ]
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.clearPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(destination.width / 8), Math.ceil(destination.height / 8));
    pass.end();
  }

  private async executeBatch(field: Field, jobs: readonly FieldJob[]): Promise<void> {
    for (let index = 0; index < jobs.length; index++) {
      const job = jobs[index];
      if (!job) continue;
      this.device.queue.writeBuffer(
        this.renderUniforms[index],
        0,
        this.createRenderData(field, job)
      );
    }

    const encoder = this.device.createCommandEncoder();
    for (let index = 0; index < jobs.length; index++) {
      const job = jobs[index];
      if (!job) continue;
      const orbitPass = encoder.beginComputePass();
      orbitPass.setPipeline(this.orbitPipeline);
      orbitPass.setBindGroup(0, field.orbitGroups[index]);
      orbitPass.dispatchWorkgroups(
        Math.ceil(Math.ceil(job.width / job.blockSize) / 8),
        Math.ceil(Math.ceil(job.height / job.blockSize) / 8)
      );
      orbitPass.end();

      const colourPass = encoder.beginComputePass();
      colourPass.setPipeline(this.colourPipeline);
      colourPass.setBindGroup(0, field.colourGroups[index]);
      colourPass.dispatchWorkgroups(Math.ceil(job.width / 8), Math.ceil(job.height / 8));
      colourPass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  private createRenderData(field: Field, job: FieldJob): ArrayBuffer {
    const data = new ArrayBuffer(RENDER_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    const signed = new Int32Array(data);
    const [centerXHi, centerXLo] = fixedSplitF32(field.request.camera.centerX);
    const [centerYHi, centerYLo] = fixedSplitF32(field.request.camera.centerY);
    floats[0] = centerXHi;
    floats[1] = centerXLo;
    floats[2] = centerYHi;
    floats[3] = centerYLo;
    floats[4] = field.request.camera.scale.mantissa;
    floats[5] = field.width / field.height;
    unsigned[6] = field.width;
    unsigned[7] = field.height;
    unsigned[8] = job.iterations;
    unsigned[9] = job.blockSize;
    unsigned[10] = job.x;
    unsigned[11] = job.y;
    unsigned[12] = job.width;
    unsigned[13] = job.height;
    unsigned[14] = field.precision === 'double-float-direct' ? 1 : 0;
    signed[15] = field.request.camera.scale.exponent;
    floats[16] = field.request.palettePhase;
    floats[17] = PALETTE_LENGTH;
    unsigned[18] = job.acceptIterationCap ? 1 : 0;
    return data;
  }

  private fieldDimensions(request: RenderRequest, navigation: NavigationProfile): { width: number; height: number } {
    const requestedDpr = clamp(request.devicePixelRatio, 1, 2);
    const interactionScale = request.interaction === 'moving'
      ? navigation.resolutionScale
      : request.interaction === 'settling' ? 0.88 : 1;
    let width = Math.max(1, Math.floor(request.cssWidth * requestedDpr * interactionScale));
    let height = Math.max(1, Math.floor(request.cssHeight * requestedDpr * interactionScale));
    const maxDimension = Number(this.device.limits.maxTextureDimension2D);
    const dimensionScale = Math.min(1, maxDimension / width, maxDimension / height);
    width = Math.max(1, Math.floor(width * dimensionScale));
    height = Math.max(1, Math.floor(height * dimensionScale));
    const maxPixels = request.interaction === 'settled' ? SETTLED_MAX_PIXELS : MOVING_MAX_PIXELS;
    const pixelScale = Math.min(1, Math.sqrt(maxPixels / Math.max(1, width * height)));
    return {
      width: Math.max(1, Math.floor(width * pixelScale)),
      height: Math.max(1, Math.floor(height * pixelScale))
    };
  }

  private precisionFor(camera: CameraSnapshot, renderHeight: number): PrecisionMode {
    const centerMagnitude = Math.max(
      1,
      Math.abs(fixedToNumber(camera.centerX)),
      Math.abs(fixedToNumber(camera.centerY))
    );
    const pixelStep = scaleToNumber(camera.scale) / Math.max(1, renderHeight);
    const conservativeF32Resolution = centerMagnitude * Math.pow(2, -21);
    return pixelStep > conservativeF32Resolution ? 'f32-direct' : 'double-float-direct';
  }

  private cameraTransform(
    source: CameraSnapshot,
    target: CameraSnapshot,
    sourceCssWidth: number,
    sourceCssHeight: number,
    targetCssWidth: number,
    targetCssHeight: number
  ): { scaleX: number; scaleY: number; offsetX: number; offsetY: number } | null {
    const sourceScale = Math.max(scaleToNumber(source.scale), Number.MIN_VALUE);
    const scaleRatio = target.scale.mantissa / source.scale.mantissa
      * Math.pow(2, target.scale.exponent - source.scale.exponent);
    const sourceAspect = Math.max(1, sourceCssWidth) / Math.max(1, sourceCssHeight);
    const targetAspect = Math.max(1, targetCssWidth) / Math.max(1, targetCssHeight);
    const transform = {
      scaleX: scaleRatio * targetAspect / sourceAspect,
      scaleY: scaleRatio,
      offsetX: fixedDifferenceToNumber(target.centerX, source.centerX) / sourceScale / sourceAspect,
      offsetY: fixedDifferenceToNumber(target.centerY, source.centerY) / sourceScale
    };
    return Object.values(transform).every(value => Number.isFinite(value) && Math.abs(value) <= 3.3e38)
      ? transform
      : null;
  }

  private canReuse(field: Field, request: RenderRequest, width: number, height: number): boolean {
    return field.request.camera.generation === request.camera.generation
      && field.request.palettePhase === request.palettePhase
      && field.width === width
      && field.height === height;
  }

  private batchCount(job: FieldJob | undefined): number {
    if (!job) return 1;
    if (job.blockSize === 1 && job.iterations >= 1500) return 1;
    if (job.blockSize <= 2 && job.iterations >= 750) return 2;
    return MAX_BATCH_JOBS;
  }

  private acquireResources(width: number, height: number): FieldResources {
    const key = `${width}x${height}`;
    const available = this.resourcePool.get(key);
    const reused = available?.pop();
    if (reused) return reused;
    return {
      key,
      resultTexture: this.device.createTexture({
        size: [width, height],
        format: RESULT_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
      }),
      colourTexture: this.device.createTexture({
        size: [width, height],
        format: COLOUR_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
      }),
      qualityTexture: this.device.createTexture({
        size: [width, height],
        format: COLOUR_FORMAT,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
      })
    };
  }

  private retireField(field: Field): void {
    void this.device.queue.onSubmittedWorkDone()
      .then(() => this.releaseField(field))
      .catch((error: unknown) => console.error('Unable to retire field resources', error));
  }

  private releaseField(field: Field): void {
    const resources = field.resources;
    const pooled = this.pooledResourceCount();
    if (pooled >= MAX_POOL_RESOURCES) {
      resources.resultTexture.destroy();
      resources.colourTexture.destroy();
      resources.qualityTexture.destroy();
      return;
    }
    const available = this.resourcePool.get(resources.key) ?? [];
    if (available.length >= 2) {
      resources.resultTexture.destroy();
      resources.colourTexture.destroy();
      resources.qualityTexture.destroy();
      return;
    }
    available.push(resources);
    this.resourcePool.set(resources.key, available);
  }

  private pooledResourceCount(): number {
    let count = 0;
    for (const resources of this.resourcePool.values()) count += resources.length;
    return count;
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
