import { fixedDifferenceToNumber, fixedSplitF32 } from '../bigFixed';
import { scaleToNumber } from '../binaryScale';
import type { CameraSnapshot } from '../v4/types';
import { progressiveColourShader, progressiveOrbitShader, progressivePresentShader } from './shaders';
import { ProgressiveTileScheduler } from './tileScheduler';
import type {
  ProgressiveRendererStats,
  ProgressiveSurfaceSnapshot,
  ProgressiveTileJob
} from './types';

const ORBIT_PARAMETER_BYTES = 64;
const COLOUR_PARAMETER_BYTES = 32;
const PRESENT_PARAMETER_BYTES = 64;
const RESULT_FORMAT: GPUTextureFormat = 'rgba32float';
const COLOUR_FORMAT: GPUTextureFormat = 'rgba8unorm';
const PALETTE_LENGTH = 64;

type Surface = {
  snapshot: ProgressiveSurfaceSnapshot;
  width: number;
  height: number;
  resultTexture: GPUTexture;
  colourTexture: GPUTexture;
};

export class ProgressiveRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly canvasFormat: GPUTextureFormat;
  private readonly orbitPipeline: GPUComputePipeline;
  private readonly colourPipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly orbitUniform: GPUBuffer;
  private readonly colourUniform: GPUBuffer;
  private readonly presentUniform: GPUBuffer;
  private readonly sampler: GPUSampler;
  private readonly scheduler = new ProgressiveTileScheduler();
  private current: Surface | null = null;
  private stable: Surface | null = null;
  private displayWidth = 1;
  private displayHeight = 1;
  private lastTileMs = 0;
  private lastBlockSize = 0;
  private completedJobs = 0;
  private totalJobs = 0;
  private readonly tileTimestamps: number[] = [];

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    canvasFormat: GPUTextureFormat,
    orbitPipeline: GPUComputePipeline,
    colourPipeline: GPUComputePipeline,
    presentPipeline: GPURenderPipeline,
    readonly adapterLabel: string
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.canvasFormat = canvasFormat;
    this.orbitPipeline = orbitPipeline;
    this.colourPipeline = colourPipeline;
    this.presentPipeline = presentPipeline;
    this.orbitUniform = device.createBuffer({
      size: ORBIT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.colourUniform = device.createBuffer({
      size: COLOUR_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.presentUniform = device.createBuffer({
      size: PRESENT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.sampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
  }

  static async create(canvas: HTMLCanvasElement): Promise<ProgressiveRenderer> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is unavailable. Use a current Chromium browser with hardware acceleration enabled.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) throw new Error('Unable to create a WebGPU canvas context');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });

    const orbitModule = device.createShaderModule({ code: progressiveOrbitShader });
    await this.assertShaderValid(orbitModule, 'V6 orbit');
    const colourModule = device.createShaderModule({ code: progressiveColourShader });
    await this.assertShaderValid(colourModule, 'V6 colour');
    const presentModule = device.createShaderModule({ code: progressivePresentShader });
    await this.assertShaderValid(presentModule, 'V6 presentation');

    const orbitPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: orbitModule, entryPoint: 'main' }
    });
    const colourPipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: colourModule, entryPoint: 'main' }
    });
    const presentPipeline = await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vertexMain' },
      fragment: {
        module: presentModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: canvasFormat }]
      },
      primitive: { topology: 'triangle-list' }
    });

    const adapterLabel = adapter.info.vendor || adapter.info.description || 'GPU';
    return new ProgressiveRenderer(
      canvas,
      device,
      context,
      canvasFormat,
      orbitPipeline,
      colourPipeline,
      presentPipeline,
      adapterLabel
    );
  }

  onDeviceError(listener: (message: string) => void): void {
    this.device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
      listener(`WebGPU error: ${event.error.message}`);
    });
    void this.device.lost.then(reason => {
      listener(`GPU device lost: ${reason.message || reason.reason}`);
    });
  }

  startAnchor(snapshot: ProgressiveSurfaceSnapshot): void {
    const width = Math.max(1, Math.floor(snapshot.cssWidth * snapshot.devicePixelRatio));
    const height = Math.max(1, Math.floor(snapshot.cssHeight * snapshot.devicePixelRatio));
    this.resizeCanvas(width, height);

    const retired = this.stable;
    this.stable = this.current;
    this.current = this.createSurface(snapshot, width, height);
    if (retired) this.retireSurface(retired);

    this.scheduler.reset(
      snapshot.generation,
      width,
      height,
      snapshot.iterations,
      snapshot.focusX,
      snapshot.focusY,
      snapshot.motionPressure
    );
    this.completedJobs = 0;
    this.totalJobs = this.scheduler.totalJobs;
    this.lastBlockSize = 0;
  }

  async step(): Promise<boolean> {
    const surface = this.current;
    if (!surface) return false;
    const job = this.scheduler.next();
    if (!job || job.generation !== surface.snapshot.generation) return false;

    const orbitData = this.createOrbitData(surface, job);
    const colourData = this.createColourData(surface, job);
    this.device.queue.writeBuffer(this.orbitUniform, 0, orbitData);
    this.device.queue.writeBuffer(this.colourUniform, 0, colourData);

    const orbitBindGroup = this.device.createBindGroup({
      layout: this.orbitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.orbitUniform } },
        { binding: 1, resource: surface.resultTexture.createView() }
      ]
    });
    const colourBindGroup = this.device.createBindGroup({
      layout: this.colourPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.colourUniform } },
        { binding: 1, resource: surface.resultTexture.createView() },
        { binding: 2, resource: surface.colourTexture.createView() }
      ]
    });

    const blockColumns = Math.ceil(job.width / job.blockSize);
    const blockRows = Math.ceil(job.height / job.blockSize);
    const encoder = this.device.createCommandEncoder();
    const orbitPass = encoder.beginComputePass();
    orbitPass.setPipeline(this.orbitPipeline);
    orbitPass.setBindGroup(0, orbitBindGroup);
    orbitPass.dispatchWorkgroups(Math.ceil(blockColumns / 8), Math.ceil(blockRows / 8));
    orbitPass.end();

    const colourPass = encoder.beginComputePass();
    colourPass.setPipeline(this.colourPipeline);
    colourPass.setBindGroup(0, colourBindGroup);
    colourPass.dispatchWorkgroups(Math.ceil(job.width / 8), Math.ceil(job.height / 8));
    colourPass.end();

    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const elapsed = Math.max(0.1, performance.now() - started);

    if (this.current?.snapshot.generation === job.generation) {
      this.scheduler.markCompleted();
      this.completedJobs = this.scheduler.completedJobs;
      this.lastTileMs = elapsed;
      this.lastBlockSize = job.blockSize;
      this.noteTileCompletion();
    }
    return true;
  }

  present(targetCamera: CameraSnapshot, cssWidth: number, cssHeight: number, devicePixelRatio: number): boolean {
    const current = this.current;
    if (!current) return false;
    const displayWidth = Math.max(1, Math.floor(cssWidth * devicePixelRatio));
    const displayHeight = Math.max(1, Math.floor(cssHeight * devicePixelRatio));
    this.resizeCanvas(displayWidth, displayHeight);

    const stable = this.stable;
    const currentTransform = this.cameraTransform(current.snapshot.camera, targetCamera, current.snapshot.cssWidth, current.snapshot.cssHeight, cssWidth, cssHeight);
    const stableTransform = stable
      ? this.cameraTransform(stable.snapshot.camera, targetCamera, stable.snapshot.cssWidth, stable.snapshot.cssHeight, cssWidth, cssHeight)
      : currentTransform;
    if (!currentTransform || !stableTransform) return false;

    const parameters = new ArrayBuffer(PRESENT_PARAMETER_BYTES);
    const floats = new Float32Array(parameters);
    const unsigned = new Uint32Array(parameters);
    floats[0] = currentTransform.scaleX;
    floats[1] = currentTransform.scaleY;
    floats[2] = currentTransform.offsetX;
    floats[3] = currentTransform.offsetY;
    floats[4] = stableTransform.scaleX;
    floats[5] = stableTransform.scaleY;
    floats[6] = stableTransform.offsetX;
    floats[7] = stableTransform.offsetY;
    unsigned[8] = stable ? 1 : 0;
    this.device.queue.writeBuffer(this.presentUniform, 0, parameters);

    const bindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.presentUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: current.colourTexture.createView() },
        { binding: 3, resource: (stable?.colourTexture ?? current.colourTexture).createView() }
      ]
    });
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
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  recolour(palettePhase: number): void {
    const surfaces = [this.stable, this.current].filter((surface): surface is Surface => Boolean(surface));
    for (const surface of surfaces) {
      surface.snapshot = { ...surface.snapshot, palettePhase };
      const job: ProgressiveTileJob = {
        generation: surface.snapshot.generation,
        x: 0,
        y: 0,
        width: surface.width,
        height: surface.height,
        blockSize: 1,
        iterations: surface.snapshot.iterations,
        priority: 0
      };
      const colourData = this.createColourData(surface, job);
      this.device.queue.writeBuffer(this.colourUniform, 0, colourData);
      const bindGroup = this.device.createBindGroup({
        layout: this.colourPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.colourUniform } },
          { binding: 1, resource: surface.resultTexture.createView() },
          { binding: 2, resource: surface.colourTexture.createView() }
        ]
      });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.colourPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(surface.width / 8), Math.ceil(surface.height / 8));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }
  }

  get hasWork(): boolean { return this.scheduler.hasWork; }
  get anchorGeneration(): number { return this.current?.snapshot.generation ?? 0; }
  get precisionLabel(): string {
    const exponent = this.current?.snapshot.camera.scale.exponent ?? 0;
    return exponent <= -12 ? 'double-float direct' : 'f32 direct';
  }

  get stats(): ProgressiveRendererStats {
    this.trimTileTimestamps(performance.now());
    return {
      phase: this.scheduler.phase,
      pendingJobs: this.scheduler.pendingJobs,
      completedJobs: this.completedJobs,
      totalJobs: this.totalJobs,
      lastBlockSize: this.lastBlockSize,
      lastTileMs: this.lastTileMs,
      tileRate: this.tileTimestamps.length,
      anchorGeneration: this.anchorGeneration,
      analyticInteriorEnabled: true
    };
  }

  private createSurface(snapshot: ProgressiveSurfaceSnapshot, width: number, height: number): Surface {
    const resultTexture = this.device.createTexture({
      size: [width, height],
      format: RESULT_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    const colourTexture = this.device.createTexture({
      size: [width, height],
      format: COLOUR_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    return { snapshot, width, height, resultTexture, colourTexture };
  }

  private createOrbitData(surface: Surface, job: ProgressiveTileJob): ArrayBuffer {
    const data = new ArrayBuffer(ORBIT_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    const signed = new Int32Array(data);
    const [centerXHi, centerXLo] = fixedSplitF32(surface.snapshot.camera.centerX);
    const [centerYHi, centerYLo] = fixedSplitF32(surface.snapshot.camera.centerY);
    floats[0] = centerXHi;
    floats[1] = centerXLo;
    floats[2] = centerYHi;
    floats[3] = centerYLo;
    floats[4] = surface.snapshot.camera.scale.mantissa;
    floats[5] = surface.width / surface.height;
    unsigned[6] = surface.width;
    unsigned[7] = surface.height;
    unsigned[8] = job.iterations;
    unsigned[9] = job.x;
    unsigned[10] = job.y;
    unsigned[11] = job.width;
    unsigned[12] = job.height;
    unsigned[13] = job.blockSize;
    unsigned[14] = surface.snapshot.camera.scale.exponent <= -12 ? 1 : 0;
    signed[15] = surface.snapshot.camera.scale.exponent;
    return data;
  }

  private createColourData(surface: Surface, job: ProgressiveTileJob): ArrayBuffer {
    const data = new ArrayBuffer(COLOUR_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    unsigned[0] = surface.width;
    unsigned[1] = surface.height;
    unsigned[2] = job.x;
    unsigned[3] = job.y;
    unsigned[4] = job.width;
    unsigned[5] = job.height;
    floats[6] = surface.snapshot.palettePhase;
    floats[7] = PALETTE_LENGTH;
    return data;
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
    const offsetX = fixedDifferenceToNumber(target.centerX, source.centerX) / sourceScale / sourceAspect;
    const offsetY = fixedDifferenceToNumber(target.centerY, source.centerY) / sourceScale;
    const transform = {
      scaleX: scaleRatio * targetAspect / sourceAspect,
      scaleY: scaleRatio,
      offsetX,
      offsetY
    };
    return Object.values(transform).every(Number.isFinite) ? transform : null;
  }

  private resizeCanvas(width: number, height: number): void {
    if (this.displayWidth === width && this.displayHeight === height) return;
    this.displayWidth = width;
    this.displayHeight = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.context.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' });
  }

  private retireSurface(surface: Surface): void {
    void this.device.queue.onSubmittedWorkDone().then(() => {
      if (surface !== this.current && surface !== this.stable) {
        surface.resultTexture.destroy();
        surface.colourTexture.destroy();
      }
    }).catch(error => console.error('Unable to retire V6 surface', error));
  }

  private noteTileCompletion(): void {
    const now = performance.now();
    this.tileTimestamps.push(now);
    this.trimTileTimestamps(now);
  }

  private trimTileTimestamps(now: number): void {
    const cutoff = now - 1000;
    while (this.tileTimestamps.length > 0 && (this.tileTimestamps[0] ?? now) < cutoff) {
      this.tileTimestamps.shift();
    }
  }

  private static async assertShaderValid(module: GPUShaderModule, label: string): Promise<void> {
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length === 0) return;
    const first = errors[0];
    throw new Error(`${label} WGSL line ${first.lineNum}: ${first.message}`);
  }
}
