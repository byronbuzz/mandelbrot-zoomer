import { fixedDifferenceToNumber, fixedSplitF32, fixedSub } from '../bigFixed';
import { scaleToNumber } from '../binaryScale';
import type { CameraSnapshot } from '../v4/types';
import { ReferenceService } from './referenceService';
import type { DeepReference } from './referenceService';
import { progressiveColourShader, progressiveOrbitShader, progressivePresentShader } from './shaders';
import { ProgressiveTileScheduler } from './tileScheduler';
import type {
  ProgressiveRendererStats,
  ProgressiveSurfaceSnapshot,
  ProgressiveTileJob,
  V6NumericalMode,
  V6ReferenceState
} from './types';

const ORBIT_PARAMETER_BYTES = 96;
const COLOUR_PARAMETER_BYTES = 32;
const PRESENT_PARAMETER_BYTES = 64;
const TELEMETRY_BYTES = 16;
const RESULT_FORMAT: GPUTextureFormat = 'rgba32float';
const COLOUR_FORMAT: GPUTextureFormat = 'rgba8unorm';
const PALETTE_LENGTH = 64;
const PERTURBATION_SCALE_EXPONENT = -13;
const STATE_BYTES_PER_PIXEL = 16;
const META_BYTES_PER_PIXEL = 8;

type OrbitTelemetrySnapshot = Readonly<{
  rebaseEvents: number;
  fallbackPixels: number;
  nonFiniteEvents: number;
  orbitExhaustions: number;
}>;

const EMPTY_TELEMETRY: OrbitTelemetrySnapshot = {
  rebaseEvents: 0,
  fallbackPixels: 0,
  nonFiniteEvents: 0,
  orbitExhaustions: 0
};

function align4(value: number): number {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

type Surface = {
  snapshot: ProgressiveSurfaceSnapshot;
  width: number;
  height: number;
  resultTexture: GPUTexture;
  colourTexture: GPUTexture;
  numericalMode: V6NumericalMode;
  referenceState: V6ReferenceState;
  referenceError: string | null;
  reference: DeepReference | null;
  referencePromise: Promise<DeepReference> | null;
  stateBuffer: GPUBuffer | null;
  metaBuffer: GPUBuffer | null;
  telemetryBuffer: GPUBuffer | null;
  telemetry: OrbitTelemetrySnapshot;
  telemetryCaptured: boolean;
  preferCurrent: boolean;
  publishedJobs: number;
};

export class ProgressiveRenderer {
  private readonly orbitUniform: GPUBuffer;
  private readonly colourUniform: GPUBuffer;
  private readonly presentUniform: GPUBuffer;
  private readonly sampler: GPUSampler;
  private readonly dummyState: GPUBuffer;
  private readonly dummyMeta: GPUBuffer;
  private readonly dummyReference: GPUBuffer;
  private readonly dummyTelemetry: GPUBuffer;
  private readonly scheduler = new ProgressiveTileScheduler();
  private readonly referenceService: ReferenceService;
  private current: Surface | null = null;
  private stable: Surface | null = null;
  private displayWidth = 1;
  private displayHeight = 1;
  private lastTileMs = 0;
  private lastBlockSize = 0;
  private lastIterationLimit = 0;
  private completedJobs = 0;
  private totalJobs = 0;
  private readonly tileTimestamps: number[] = [];

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly canvasFormat: GPUTextureFormat,
    private readonly orbitPipeline: GPUComputePipeline,
    private readonly colourPipeline: GPUComputePipeline,
    private readonly presentPipeline: GPURenderPipeline,
    readonly adapterLabel: string
  ) {
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
    this.dummyState = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    this.dummyMeta = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    this.dummyReference = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE });
    this.dummyTelemetry = device.createBuffer({ size: TELEMETRY_BYTES, usage: GPUBufferUsage.STORAGE });
    this.sampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
    this.referenceService = new ReferenceService(device);
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
    void this.device.lost.then((reason: { message?: string; reason?: string }) => {
      listener(`GPU device lost: ${reason.message || reason.reason}`);
    });
  }

  startAnchor(snapshot: ProgressiveSurfaceSnapshot): void {
    const deep = this.needsPerturbation(snapshot.camera);
    const deepMotionPreview = deep && snapshot.motionPressure > 0.2;
    this.referenceService.cancel();

    const dimensions = this.surfaceDimensions(snapshot, deep && !deepMotionPreview);
    const next = this.createSurface(
      snapshot,
      dimensions.width,
      dimensions.height,
      deep && !deepMotionPreview,
      deepMotionPreview
    );
    this.replaceCurrent(next);

    const surface = this.current;
    if (!surface) return;
    if (deepMotionPreview) {
      this.scheduler.reset(
        snapshot.generation,
        surface.width,
        surface.height,
        snapshot.iterations,
        snapshot.focusX,
        snapshot.focusY,
        snapshot.motionPressure,
        undefined,
        true
      );
    } else if (deep) {
      surface.referencePromise = this.referenceService.request(snapshot, surface.width, surface.height);
      this.scheduler.reset(
        snapshot.generation,
        surface.width,
        surface.height,
        snapshot.iterations,
        snapshot.focusX,
        snapshot.focusY,
        snapshot.motionPressure,
        undefined,
        true
      );
    } else {
      this.scheduler.reset(
        snapshot.generation,
        surface.width,
        surface.height,
        snapshot.iterations,
        snapshot.focusX,
        snapshot.focusY,
        snapshot.motionPressure
      );
    }
    this.resetProgress();
    this.totalJobs = this.scheduler.totalJobs;
  }

  async step(): Promise<boolean> {
    let surface = this.current;
    if (!surface) return false;
    if (surface.numericalMode === 'perturbation-pending') {
      const pendingReference = surface.referencePromise;
      if (!pendingReference) {
        this.activateDirectFallback(surface, 'reference promise missing');
      } else {
        try {
          const reference = await pendingReference;
          if (surface !== this.current) {
            reference.buffer.destroy();
            return false;
          }
          this.activatePerturbation(surface, reference);
        } catch (error) {
          if (surface !== this.current) return false;
          const message = error instanceof Error ? error.message : String(error);
          this.activateDirectFallback(surface, message);
          if (message !== 'Reference request superseded') {
            console.error('Unable to generate V6 perturbation reference; using direct fallback', error);
          }
        }
      }
      surface = this.current;
      if (!surface) return false;
    }

    const job = this.scheduler.next();
    if (!job || job.generation !== surface.snapshot.generation) return false;
    const finalScheduledJob = this.scheduler.pendingJobs === 0;
    this.device.queue.writeBuffer(this.orbitUniform, 0, this.createOrbitData(surface, job));
    this.device.queue.writeBuffer(this.colourUniform, 0, this.createColourData(surface, job));

    const orbitBindGroup = this.device.createBindGroup({
      layout: this.orbitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.orbitUniform } },
        { binding: 1, resource: surface.resultTexture.createView() },
        { binding: 2, resource: { buffer: surface.stateBuffer ?? this.dummyState } },
        { binding: 3, resource: { buffer: surface.metaBuffer ?? this.dummyMeta } },
        { binding: 4, resource: { buffer: surface.reference?.buffer ?? this.dummyReference } },
        { binding: 5, resource: { buffer: surface.telemetryBuffer ?? this.dummyTelemetry } }
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

    const encoder = this.device.createCommandEncoder();
    const orbitPass = encoder.beginComputePass();
    orbitPass.setPipeline(this.orbitPipeline);
    orbitPass.setBindGroup(0, orbitBindGroup);
    orbitPass.dispatchWorkgroups(
      Math.ceil(Math.ceil(job.width / job.blockSize) / 8),
      Math.ceil(Math.ceil(job.height / job.blockSize) / 8)
    );
    orbitPass.end();

    const colourPass = encoder.beginComputePass();
    colourPass.setPipeline(this.colourPipeline);
    colourPass.setBindGroup(0, colourBindGroup);
    colourPass.dispatchWorkgroups(Math.ceil(job.width / 8), Math.ceil(job.height / 8));
    colourPass.end();

    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    if (this.current?.snapshot.generation === job.generation) {
      this.scheduler.markCompleted();
      this.completedJobs = this.scheduler.completedJobs;
      this.lastTileMs = Math.max(0.1, performance.now() - started);
      this.lastBlockSize = job.blockSize;
      this.lastIterationLimit = job.iterations;
      surface.publishedJobs++;
      this.noteTileCompletion();
      if (finalScheduledJob) await this.captureTelemetry(surface);
    }
    return true;
  }

  present(targetCamera: CameraSnapshot, cssWidth: number, cssHeight: number, devicePixelRatio: number): boolean {
    const current = this.current;
    if (!current) return false;
    this.resizeCanvas(
      Math.max(1, Math.floor(cssWidth * devicePixelRatio)),
      Math.max(1, Math.floor(cssHeight * devicePixelRatio))
    );

    const stable = this.stable;
    const currentTransform = this.cameraTransform(
      current.snapshot.camera,
      targetCamera,
      current.snapshot.cssWidth,
      current.snapshot.cssHeight,
      cssWidth,
      cssHeight
    );
    const stableTransform = stable
      ? this.cameraTransform(
          stable.snapshot.camera,
          targetCamera,
          stable.snapshot.cssWidth,
          stable.snapshot.cssHeight,
          cssWidth,
          cssHeight
        )
      : currentTransform;
    if (!currentTransform || !stableTransform) return false;

    const data = new ArrayBuffer(PRESENT_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    floats.set([
      currentTransform.scaleX,
      currentTransform.scaleY,
      currentTransform.offsetX,
      currentTransform.offsetY,
      stableTransform.scaleX,
      stableTransform.scaleY,
      stableTransform.offsetX,
      stableTransform.offsetY
    ]);
    unsigned[8] = stable ? 1 : 0;
    unsigned[9] = current.preferCurrent ? 1 : 0;
    this.device.queue.writeBuffer(this.presentUniform, 0, data);

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
      this.device.queue.writeBuffer(this.colourUniform, 0, this.createColourData(surface, job));
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
    const surface = this.current;
    if (!surface) return 'waiting';
    if (surface.referenceState === 'deferred') return 'deep motion preview · double-float direct';
    if (surface.numericalMode === 'perturbation-pending') return 'perturbation reference generating';
    if (surface.numericalMode === 'perturbation') {
      const reference = surface.reference;
      const telemetry = surface.telemetry;
      const repair = telemetry.fallbackPixels > 0 ? ` · ${telemetry.fallbackPixels} fallback px` : '';
      return `scaled perturbation · ${reference?.bits ?? 0}-bit ref · ${reference?.length ?? 0} orbit · ${telemetry.rebaseEvents} rebases${repair}`;
    }
    if (surface.referenceState === 'failed') return 'double-float direct fallback';
    return surface.numericalMode === 'double-float-direct' ? 'double-float direct' : 'f32 direct';
  }

  get stats(): ProgressiveRendererStats {
    this.trimTileTimestamps(performance.now());
    const surface = this.current;
    const telemetry = surface?.telemetry ?? EMPTY_TELEMETRY;
    return {
      phase: this.scheduler.phase,
      pendingJobs: this.scheduler.pendingJobs,
      completedJobs: this.completedJobs,
      totalJobs: this.totalJobs,
      lastBlockSize: this.lastBlockSize,
      lastIterationLimit: this.lastIterationLimit,
      lastTileMs: this.lastTileMs,
      tileRate: this.tileTimestamps.length,
      anchorGeneration: this.anchorGeneration,
      analyticInteriorEnabled: true,
      numericalMode: surface?.numericalMode ?? 'f32-direct',
      referenceState: surface?.referenceState ?? 'inactive',
      referenceOrbitLength: surface?.reference?.length ?? 0,
      referenceBits: surface?.reference?.bits ?? 0,
      referenceGenerationMs: surface?.reference?.generationMs ?? 0,
      referenceError: surface?.referenceError ?? null,
      previewActive: surface?.preferCurrent ?? false,
      publishedJobs: surface?.publishedJobs ?? 0,
      rebaseEvents: telemetry.rebaseEvents,
      fallbackPixels: telemetry.fallbackPixels,
      nonFiniteEvents: telemetry.nonFiniteEvents,
      orbitExhaustions: telemetry.orbitExhaustions
    };
  }

  private createSurface(
    snapshot: ProgressiveSurfaceSnapshot,
    width: number,
    height: number,
    perturbationPending: boolean,
    preferCurrent: boolean
  ): Surface {
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
    const directMode: V6NumericalMode = snapshot.camera.scale.exponent <= -12
      ? 'double-float-direct'
      : 'f32-direct';
    return {
      snapshot,
      width,
      height,
      resultTexture,
      colourTexture,
      numericalMode: perturbationPending ? 'perturbation-pending' : directMode,
      referenceState: perturbationPending ? 'generating' : preferCurrent ? 'deferred' : 'inactive',
      referenceError: null,
      reference: null,
      referencePromise: null,
      stateBuffer: null,
      metaBuffer: null,
      telemetryBuffer: null,
      telemetry: EMPTY_TELEMETRY,
      telemetryCaptured: false,
      preferCurrent,
      publishedJobs: 0
    };
  }

  private replaceCurrent(next: Surface): void {
    const previous = this.current;
    if (previous) {
      if (previous.publishedJobs > 0) {
        this.releaseRecurrence(previous);
        const retiredStable = this.stable;
        this.stable = previous;
        if (retiredStable && retiredStable !== previous) this.retireSurface(retiredStable);
      } else {
        this.retireSurface(previous);
      }
    }
    this.current = next;
  }

  private activatePerturbation(surface: Surface, reference: DeepReference): void {
    const pixelCount = surface.width * surface.height;
    const stateBuffer = this.device.createBuffer({
      size: align4(pixelCount * STATE_BYTES_PER_PIXEL),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const metaBuffer = this.device.createBuffer({
      size: align4(pixelCount * META_BYTES_PER_PIXEL),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const telemetryBuffer = this.device.createBuffer({
      size: TELEMETRY_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    const encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(stateBuffer);
    encoder.clearBuffer(metaBuffer);
    encoder.clearBuffer(telemetryBuffer);
    this.device.queue.submit([encoder.finish()]);

    surface.stateBuffer = stateBuffer;
    surface.metaBuffer = metaBuffer;
    surface.telemetryBuffer = telemetryBuffer;
    surface.telemetry = EMPTY_TELEMETRY;
    surface.telemetryCaptured = false;
    surface.reference = reference;
    surface.referencePromise = null;
    surface.referenceState = 'ready';
    surface.referenceError = null;
    surface.numericalMode = 'perturbation';
    surface.preferCurrent = false;
  }

  private activateDirectFallback(surface: Surface, reason: string): void {
    surface.referencePromise = null;
    surface.referenceState = 'failed';
    surface.referenceError = reason;
    surface.numericalMode = 'double-float-direct';
    surface.preferCurrent = true;
    this.scheduler.reset(
      surface.snapshot.generation,
      surface.width,
      surface.height,
      surface.snapshot.iterations,
      surface.snapshot.focusX,
      surface.snapshot.focusY,
      0
    );
    this.resetProgress();
    this.totalJobs = this.scheduler.totalJobs;
  }

  private createOrbitData(surface: Surface, job: ProgressiveTileJob): ArrayBuffer {
    const data = new ArrayBuffer(ORBIT_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    const signed = new Int32Array(data);
    const [centerXHi, centerXLo] = fixedSplitF32(surface.snapshot.camera.centerX);
    const [centerYHi, centerYLo] = fixedSplitF32(surface.snapshot.camera.centerY);
    floats.set([centerXHi, centerXLo, centerYHi, centerYLo]);
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
    unsigned[14] = surface.numericalMode === 'perturbation'
      ? 2
      : surface.snapshot.camera.scale.exponent <= -12 ? 1 : 0;
    signed[15] = surface.snapshot.camera.scale.exponent;
    if (surface.reference) {
      const deltaX = fixedSplitF32(fixedSub(surface.snapshot.camera.centerX, surface.reference.centerX));
      const deltaY = fixedSplitF32(fixedSub(surface.snapshot.camera.centerY, surface.reference.centerY));
      floats.set([deltaX[0], deltaX[1], deltaY[0], deltaY[1]], 16);
      unsigned[20] = surface.reference.length;
    }
    unsigned[21] = surface.snapshot.iterations;
    return data;
  }

  private createColourData(surface: Surface, job: ProgressiveTileJob): ArrayBuffer {
    const data = new ArrayBuffer(COLOUR_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    unsigned.set([surface.width, surface.height, job.x, job.y, job.width, job.height]);
    floats[6] = surface.snapshot.palettePhase;
    floats[7] = PALETTE_LENGTH;
    return data;
  }

  private surfaceDimensions(
    snapshot: ProgressiveSurfaceSnapshot,
    perturbation: boolean
  ): { width: number; height: number } {
    const cssWidth = Math.max(1, snapshot.cssWidth);
    const cssHeight = Math.max(1, snapshot.cssHeight);
    let pixelRatio = snapshot.devicePixelRatio;
    if (perturbation) {
      const limit = Number(this.device.limits.maxStorageBufferBindingSize) * 0.9;
      const maxPixels = Math.max(
        1,
        Math.min(
          Math.floor(limit / STATE_BYTES_PER_PIXEL),
          Math.floor(limit / META_BYTES_PER_PIXEL)
        )
      );
      pixelRatio = Math.min(pixelRatio, Math.sqrt(maxPixels / (cssWidth * cssHeight)));
    }
    return {
      width: Math.max(1, Math.floor(cssWidth * Math.max(pixelRatio, 1 / cssWidth))),
      height: Math.max(1, Math.floor(cssHeight * Math.max(pixelRatio, 1 / cssHeight)))
    };
  }

  private needsPerturbation(camera: CameraSnapshot): boolean {
    return camera.scale.exponent <= PERTURBATION_SCALE_EXPONENT;
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
    return Object.values(transform).every(Number.isFinite) ? transform : null;
  }

  private async captureTelemetry(surface: Surface): Promise<void> {
    const source = surface.telemetryBuffer;
    if (!source || surface.telemetryCaptured) return;
    surface.telemetryCaptured = true;
    const readback = this.device.createBuffer({
      size: TELEMETRY_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, readback, 0, TELEMETRY_BYTES);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(readback.getMappedRange()).slice();
      readback.unmap();
      surface.telemetry = {
        rebaseEvents: values[0] ?? 0,
        fallbackPixels: values[1] ?? 0,
        nonFiniteEvents: values[2] ?? 0,
        orbitExhaustions: values[3] ?? 0
      };
    } catch (error) {
      surface.telemetryCaptured = false;
      console.error('Unable to read V6 perturbation telemetry', error);
    } finally {
      readback.destroy();
    }
  }

  private resizeCanvas(width: number, height: number): void {
    if (this.displayWidth === width && this.displayHeight === height) return;
    this.displayWidth = width;
    this.displayHeight = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.context.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' });
  }

  private releaseRecurrence(surface: Surface): void {
    const state = surface.stateBuffer;
    const meta = surface.metaBuffer;
    const telemetry = surface.telemetryBuffer;
    surface.stateBuffer = null;
    surface.metaBuffer = null;
    surface.telemetryBuffer = null;
    if (!state && !meta && !telemetry) return;
    void this.device.queue.onSubmittedWorkDone().then(() => {
      state?.destroy();
      meta?.destroy();
      telemetry?.destroy();
    }).catch((error: unknown) => console.error('Unable to release V6 recurrence state', error));
  }

  private retireSurface(surface: Surface): void {
    this.releaseRecurrence(surface);
    const reference = surface.reference?.buffer;
    surface.reference = null;
    surface.referencePromise = null;
    void this.device.queue.onSubmittedWorkDone().then(() => {
      if (surface !== this.current && surface !== this.stable) {
        surface.resultTexture.destroy();
        surface.colourTexture.destroy();
        reference?.destroy();
      }
    }).catch((error: unknown) => console.error('Unable to retire V6 surface', error));
  }

  private resetProgress(): void {
    this.completedJobs = 0;
    this.totalJobs = 0;
    this.lastBlockSize = 0;
    this.lastIterationLimit = 0;
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
    const errors = compilation.messages.filter((message: { type: string }) => message.type === 'error');
    if (errors.length === 0) return;
    const first = errors[0];
    throw new Error(`${label} WGSL line ${first.lineNum}: ${first.message}`);
  }
}
