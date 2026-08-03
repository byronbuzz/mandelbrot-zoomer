import {
  admitReprojection,
  reprojectionTransform,
  sameView,
  type PresentationView
} from './geometry';
import { overlayShader, presentShader, reprojectShader } from './shaders';

const REPROJECT_UNIFORM_BYTES = 32;
const VIEW_UNIFORM_BYTES = 32;
const TILE_COLUMNS = 8;
const TILE_ROWS = 6;
const TILE_COUNT = TILE_COLUMNS * TILE_ROWS;
const TILE_BATCH = 6;

type ResourceSet = {
  anchor: GPUTexture;
  candidate: GPUTexture;
  width: number;
  height: number;
  epoch: number;
};

export type Phase1Diagnostics = Readonly<{
  state: 'initializing' | 'ready' | 'lost' | 'recovering' | 'failed' | 'suspended';
  deviceEpoch: number;
  resourceEpoch: number;
  recoveryCount: number;
  validationErrors: number;
  frames: number;
  submissions: number;
  historyFrames: number;
  fallbackFrames: number;
  anchorPromotions: number;
  acceptedTiles: number;
  tileCount: number;
  lastFrameCpuMs: number;
  frameCpuP95Ms: number;
  worstReprojectionErrorTexels: number;
  width: number;
  height: number;
}>;

function percentile95(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * 0.95))] ?? 0;
}

export class Phase1PresentationKernel {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private reprojectPipeline!: GPURenderPipeline;
  private overlayPipeline!: GPURenderPipeline;
  private presentPipeline!: GPURenderPipeline;
  private sampler!: GPUSampler;
  private reprojectUniform!: GPUBuffer;
  private viewUniform!: GPUBuffer;
  private tileBuffer!: GPUBuffer;
  private resources: ResourceSet | null = null;
  private resizeSource: ResourceSet | null = null;
  private anchorView: PresentationView | null = null;
  private currentView: PresentationView;
  private stateValue: Phase1Diagnostics['state'] = 'initializing';
  private deviceEpoch = 1;
  private resourceEpoch = 0;
  private recoveryCount = 0;
  private validationErrors = 0;
  private frames = 0;
  private submissions = 0;
  private historyFrames = 0;
  private fallbackFrames = 0;
  private anchorPromotions = 0;
  private acceptedTiles = 0;
  private lastFrameCpuMs = 0;
  private frameCpuSamples: number[] = [];
  private worstReprojectionErrorTexels = 0;
  private destroyed = false;
  private errorListener: ((message: string) => void) | null = null;

  private constructor(
    private readonly canvas: HTMLCanvasElement,
    adapter: GPUAdapter,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    initialView: PresentationView
  ) {
    this.adapter = adapter;
    this.device = device;
    this.context = context;
    this.format = format;
    this.currentView = initialView;
  }

  static async create(
    canvas: HTMLCanvasElement,
    initialView: PresentationView
  ): Promise<Phase1PresentationKernel> {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) throw new Error('Unable to create WebGPU canvas context');
    const kernel = new Phase1PresentationKernel(
      canvas,
      adapter,
      device,
      context,
      navigator.gpu.getPreferredCanvasFormat(),
      initialView
    );
    await kernel.initializeDeviceResources();
    kernel.installDeviceHandlers(device, kernel.deviceEpoch);
    kernel.stateValue = 'ready';
    return kernel;
  }

  get diagnostics(): Phase1Diagnostics {
    const resources = this.resources;
    return {
      state: this.stateValue,
      deviceEpoch: this.deviceEpoch,
      resourceEpoch: this.resourceEpoch,
      recoveryCount: this.recoveryCount,
      validationErrors: this.validationErrors,
      frames: this.frames,
      submissions: this.submissions,
      historyFrames: this.historyFrames,
      fallbackFrames: this.fallbackFrames,
      anchorPromotions: this.anchorPromotions,
      acceptedTiles: this.acceptedTiles,
      tileCount: TILE_COUNT,
      lastFrameCpuMs: this.lastFrameCpuMs,
      frameCpuP95Ms: percentile95(this.frameCpuSamples),
      worstReprojectionErrorTexels: this.worstReprojectionErrorTexels,
      width: resources?.width ?? 0,
      height: resources?.height ?? 0
    };
  }

  onError(listener: (message: string) => void): void {
    this.errorListener = listener;
  }

  setView(view: PresentationView): void {
    if (!sameView(this.currentView, view)) {
      this.currentView = view;
      this.acceptedTiles = 0;
    }
  }

  render(cssWidth: number, cssHeight: number, devicePixelRatio: number): boolean {
    if (this.destroyed || (this.stateValue !== 'ready' && this.stateValue !== 'suspended')) return false;
    const started = performance.now();
    const width = Math.floor(cssWidth * devicePixelRatio);
    const height = Math.floor(cssHeight * devicePixelRatio);
    if (width <= 0 || height <= 0) {
      this.stateValue = 'suspended';
      return false;
    }
    const limit = this.device.limits.maxTextureDimension2D;
    const physicalWidth = Math.min(limit, Math.max(1, width));
    const physicalHeight = Math.min(limit, Math.max(1, height));
    this.ensureSize(physicalWidth, physicalHeight);
    const resources = this.resources;
    if (!resources) return false;

    const historyResources = this.resizeSource ?? resources;
    const transform = this.anchorView
      ? reprojectionTransform(this.anchorView, this.currentView)
      : { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };
    const admission = admitReprojection(
      transform,
      historyResources.width,
      historyResources.height,
      resources.width,
      resources.height
    );
    const historyValid = this.anchorView !== null && admission.accepted;
    this.worstReprojectionErrorTexels = admission.worstSourceTexelError;
    if (historyValid) this.historyFrames++;
    else this.fallbackFrames++;

    this.acceptedTiles = Math.min(TILE_COUNT, this.acceptedTiles + TILE_BATCH);
    this.device.queue.writeBuffer(this.reprojectUniform, 0, new Float32Array([
      admission.packed.scaleX,
      admission.packed.scaleY,
      admission.packed.offsetX,
      admission.packed.offsetY
    ]));
    this.device.queue.writeBuffer(this.reprojectUniform, 16, new Uint32Array([
      historyValid ? 1 : 0,
      resources.epoch,
      0,
      0
    ]));
    this.device.queue.writeBuffer(this.viewUniform, 0, new Float32Array([
      this.currentView.centerX,
      this.currentView.centerY,
      this.currentView.height,
      this.currentView.aspect
    ]));
    this.device.queue.writeBuffer(this.viewUniform, 16, new Uint32Array([
      resources.epoch,
      0,
      0,
      0
    ]));

    const reprojectGroup = this.device.createBindGroup({
      layout: this.reprojectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.reprojectUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: historyResources.anchor.createView() }
      ]
    });
    const overlayGroup = this.device.createBindGroup({
      layout: this.overlayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewUniform } },
        { binding: 1, resource: { buffer: this.tileBuffer } }
      ]
    });
    const presentGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: resources.candidate.createView() }
      ]
    });

    try {
      const canvasView = this.context.getCurrentTexture().createView();
      const encoder = this.device.createCommandEncoder({ label: 'phase1-frame' });
      const compose = encoder.beginRenderPass({
        colorAttachments: [{
          view: resources.candidate.createView(),
          clearValue: { r: 0.01, g: 0.015, b: 0.025, a: 1 },
          loadOp: 'clear',
          storeOp: 'store'
        }]
      });
      compose.setPipeline(this.reprojectPipeline);
      compose.setBindGroup(0, reprojectGroup);
      compose.draw(3);
      compose.setPipeline(this.overlayPipeline);
      compose.setBindGroup(0, overlayGroup);
      compose.draw(6, this.acceptedTiles);
      compose.end();

      const present = encoder.beginRenderPass({
        colorAttachments: [{
          view: canvasView,
          clearValue: { r: 0.01, g: 0.015, b: 0.025, a: 1 },
          loadOp: 'clear',
          storeOp: 'store'
        }]
      });
      present.setPipeline(this.presentPipeline);
      present.setBindGroup(0, presentGroup);
      present.draw(3);
      present.end();
      this.device.queue.submit([encoder.finish()]);
      this.submissions++;
      this.frames++;
    } catch (error) {
      this.errorListener?.(error instanceof Error ? error.message : String(error));
      return false;
    }

    if (this.resizeSource) {
      const retired = this.resizeSource;
      const retiredEpoch = this.deviceEpoch;
      this.resizeSource = null;
      void this.device.queue.onSubmittedWorkDone().then(() => {
        if (retiredEpoch === this.deviceEpoch) this.destroyResourceSet(retired);
      });
    }

    if (this.acceptedTiles === TILE_COUNT) {
      const previousAnchor = resources.anchor;
      resources.anchor = resources.candidate;
      resources.candidate = previousAnchor;
      this.anchorView = { ...this.currentView };
      this.anchorPromotions++;
      this.acceptedTiles = 0;
    }
    this.lastFrameCpuMs = performance.now() - started;
    this.frameCpuSamples.push(this.lastFrameCpuMs);
    if (this.frameCpuSamples.length > 240) this.frameCpuSamples.shift();
    return true;
  }

  forceDeviceLossForTest(): void {
    if (this.stateValue === 'ready') this.device.destroy();
  }

  destroy(): void {
    this.destroyed = true;
    this.destroyResourceSet(this.resources);
    this.destroyResourceSet(this.resizeSource);
    this.resources = null;
    this.resizeSource = null;
    this.reprojectUniform.destroy();
    this.viewUniform.destroy();
    this.tileBuffer.destroy();
    this.context.unconfigure();
    this.device.destroy();
  }

  private async initializeDeviceResources(): Promise<void> {
    const reprojectModule = this.device.createShaderModule({ code: reprojectShader });
    const overlayModule = this.device.createShaderModule({ code: overlayShader });
    const presentModule = this.device.createShaderModule({ code: presentShader });
    await Promise.all([
      this.assertShaderValid(reprojectModule, 'history reprojection'),
      this.assertShaderValid(overlayModule, 'tile overlay'),
      this.assertShaderValid(presentModule, 'canvas presentation')
    ]);
    [this.reprojectPipeline, this.overlayPipeline, this.presentPipeline] = await Promise.all([
      this.device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: reprojectModule, entryPoint: 'vertexMain' },
        fragment: { module: reprojectModule, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' }
      }),
      this.device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: overlayModule, entryPoint: 'vertexMain' },
        fragment: { module: overlayModule, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' }
      }),
      this.device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: presentModule, entryPoint: 'vertexMain' },
        fragment: { module: presentModule, entryPoint: 'fragmentMain', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' }
      })
    ]);
    this.sampler = this.device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
    this.reprojectUniform = this.device.createBuffer({
      size: REPROJECT_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.viewUniform = this.device.createBuffer({
      size: VIEW_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const rects = new Float32Array(TILE_COUNT * 4);
    for (let index = 0; index < TILE_COUNT; index++) {
      const x = index % TILE_COLUMNS;
      const y = Math.floor(index / TILE_COLUMNS);
      rects.set([
        x / TILE_COLUMNS,
        y / TILE_ROWS,
        (x + 1) / TILE_COLUMNS,
        (y + 1) / TILE_ROWS
      ], index * 4);
    }
    this.tileBuffer = this.device.createBuffer({
      size: rects.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(this.tileBuffer, 0, rects);
  }

  private ensureSize(width: number, height: number): void {
    if (this.stateValue === 'suspended') this.stateValue = 'ready';
    const current = this.resources;
    if (current?.width === width && current.height === height) return;
    const previous = current;
    this.resourceEpoch++;
    const next = this.createResourceSet(width, height, this.resourceEpoch);
    this.canvas.width = width;
    this.canvas.height = height;
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.resources = next;
    this.resizeSource = previous;
    this.acceptedTiles = 0;
  }

  private createResourceSet(width: number, height: number, epoch: number): ResourceSet {
    const create = (label: string) => this.device.createTexture({
      label,
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    return {
      anchor: create(`phase1-anchor-${epoch}`),
      candidate: create(`phase1-candidate-${epoch}`),
      width,
      height,
      epoch
    };
  }

  private destroyResourceSet(resources: ResourceSet | null): void {
    resources?.anchor.destroy();
    resources?.candidate.destroy();
  }

  private installDeviceHandlers(device: GPUDevice, epoch: number): void {
    device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
      if (epoch !== this.deviceEpoch) return;
      this.validationErrors++;
      this.errorListener?.(`WebGPU error: ${event.error.message}`);
    });
    void device.lost.then(info => {
      if (this.destroyed || epoch !== this.deviceEpoch) return;
      this.stateValue = 'lost';
      this.errorListener?.(`Device lost: ${info.message || info.reason}`);
      void this.recover();
    });
  }

  private async recover(): Promise<void> {
    if (this.destroyed || this.stateValue === 'recovering') return;
    this.stateValue = 'recovering';
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('No adapter available during recovery');
      const device = await adapter.requestDevice();
      this.destroyResourceSet(this.resources);
      this.destroyResourceSet(this.resizeSource);
      this.resources = null;
      this.resizeSource = null;
      this.adapter = adapter;
      this.device = device;
      this.deviceEpoch++;
      this.resourceEpoch++;
      this.anchorView = null;
      this.acceptedTiles = 0;
      await this.initializeDeviceResources();
      this.installDeviceHandlers(device, this.deviceEpoch);
      this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
      this.recoveryCount++;
      this.stateValue = 'ready';
    } catch (error) {
      this.stateValue = 'failed';
      this.errorListener?.(`Device recovery failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private async assertShaderValid(module: GPUShaderModule, label: string): Promise<void> {
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length === 0) return;
    throw new Error(`${label} WGSL line ${errors[0]?.lineNum}: ${errors[0]?.message}`);
  }
}
