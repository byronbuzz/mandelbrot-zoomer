import { fixedDifferenceToNumber, fixedSplitF32 } from '../bigFixed';
import { computeShader, presentShader } from './shaders';
import type { CpuReference, GpuReference, PreparedFrame, RenderSnapshot } from './types';

const PARAMETER_BYTES = 80;
const COMPUTE_TEXTURE_FORMAT: GPUTextureFormat = 'rgba8unorm';

export class WebGpuRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly canvasFormat: GPUTextureFormat;
  private readonly computePipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly fallbackOrbit: GPUBuffer;
  private readonly label: string;
  private displayWidth = 1;
  private displayHeight = 1;

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    canvasFormat: GPUTextureFormat,
    computePipeline: GPUComputePipeline,
    presentPipeline: GPURenderPipeline,
    label: string
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.canvasFormat = canvasFormat;
    this.computePipeline = computePipeline;
    this.presentPipeline = presentPipeline;
    this.label = label;
    this.sampler = device.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });
    this.fallbackOrbit = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.fallbackOrbit, 0, new Float32Array(4));
  }

  static async create(canvas: HTMLCanvasElement): Promise<WebGpuRenderer> {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a current Chromium browser with hardware acceleration enabled.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice();
    const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
    if (!context) throw new Error('Unable to create a WebGPU canvas context');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });

    const computeModule = device.createShaderModule({ code: computeShader });
    await WebGpuRenderer.assertShaderValid(computeModule, 'compute');
    const presentModule = device.createShaderModule({ code: presentShader });
    await WebGpuRenderer.assertShaderValid(presentModule, 'present');

    const computePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: computeModule, entryPoint: 'main' }
    });
    const presentPipeline = await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vertexMain' },
      fragment: { module: presentModule, entryPoint: 'fragmentMain', targets: [{ format: canvasFormat }] },
      primitive: { topology: 'triangle-list' }
    });
    const vendor = adapter.info.vendor || adapter.info.description || 'GPU';
    return new WebGpuRenderer(canvas, device, context, canvasFormat, computePipeline, presentPipeline, vendor);
  }

  get adapterLabel(): string { return this.label; }

  onDeviceError(listener: (message: string) => void): void {
    this.device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => listener(`WebGPU error: ${event.error.message}`));
    void this.device.lost.then((reason: GPUDeviceLostInfo) => listener(`GPU device lost: ${reason.message || reason.reason}`));
  }

  createReference(reference: CpuReference): GpuReference {
    const buffer = this.device.createBuffer({
      size: Math.max(16, reference.orbit.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(buffer, 0, reference.orbit);
    return {
      id: reference.id,
      cameraGeneration: reference.cameraGeneration,
      purpose: reference.purpose,
      centerX: reference.centerX,
      centerY: reference.centerY,
      requestedIterations: reference.requestedIterations,
      length: reference.length,
      escaped: reference.escaped,
      bits: reference.bits,
      generationMs: reference.generationMs,
      buffer
    };
  }

  destroyReference(reference: GpuReference): void { reference.buffer.destroy(); }

  async prepare(snapshot: RenderSnapshot): Promise<PreparedFrame> {
    const displayWidth = Math.max(1, Math.floor(snapshot.cssWidth * snapshot.devicePixelRatio));
    const displayHeight = Math.max(1, Math.floor(snapshot.cssHeight * snapshot.devicePixelRatio));
    this.resizeCanvas(displayWidth, displayHeight);
    const computeWidth = Math.max(1, Math.floor(displayWidth * snapshot.quality.resolution));
    const computeHeight = Math.max(1, Math.floor(displayHeight * snapshot.quality.resolution));
    const texture = this.device.createTexture({
      size: [computeWidth, computeHeight],
      format: COMPUTE_TEXTURE_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    const uniform = this.device.createBuffer({
      size: PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(uniform, 0, this.createParameterData(snapshot, computeWidth, computeHeight));
    const bindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: texture.createView() },
        { binding: 2, resource: { buffer: snapshot.reference?.buffer ?? this.fallbackOrbit } }
      ]
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(computeWidth / 8), Math.ceil(computeHeight / 8));
    pass.end();
    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      texture.destroy();
      uniform.destroy();
      throw error;
    }
    uniform.destroy();
    return {
      snapshot,
      texture,
      computeWidth,
      computeHeight,
      displayWidth,
      displayHeight,
      computeMs: Math.max(0.1, performance.now() - started)
    };
  }

  async present(frame: PreparedFrame): Promise<number> {
    const bindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: frame.texture.createView() }
      ]
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    frame.texture.destroy();
    return Math.max(0.1, performance.now() - started);
  }

  discard(frame: PreparedFrame): void { frame.texture.destroy(); }

  private resizeCanvas(width: number, height: number): void {
    if (this.displayWidth === width && this.displayHeight === height) return;
    this.displayWidth = width;
    this.displayHeight = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.context.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' });
  }

  private createParameterData(snapshot: RenderSnapshot, width: number, height: number): ArrayBuffer {
    const data = new ArrayBuffer(PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    const signed = new Int32Array(data);
    const [centerXHi, centerXLo] = fixedSplitF32(snapshot.camera.centerX);
    const [centerYHi, centerYLo] = fixedSplitF32(snapshot.camera.centerY);
    let referenceOffsetX = 0;
    let referenceOffsetY = 0;
    if (snapshot.reference) {
      referenceOffsetX = fixedDifferenceToNumber(snapshot.camera.centerX, snapshot.reference.centerX);
      referenceOffsetY = fixedDifferenceToNumber(snapshot.camera.centerY, snapshot.reference.centerY);
    }
    const offsetXHi = Math.fround(referenceOffsetX);
    const offsetYHi = Math.fround(referenceOffsetY);
    floats[0] = centerXHi;
    floats[1] = centerXLo;
    floats[2] = centerYHi;
    floats[3] = centerYLo;
    floats[4] = offsetXHi;
    floats[5] = Math.fround(referenceOffsetX - offsetXHi);
    floats[6] = offsetYHi;
    floats[7] = Math.fround(referenceOffsetY - offsetYHi);
    floats[8] = Math.fround(snapshot.camera.scale.mantissa);
    floats[9] = width / height;
    unsigned[10] = snapshot.quality.iterations;
    floats[11] = snapshot.palettePhase;
    unsigned[12] = width;
    unsigned[13] = height;
    unsigned[14] = snapshot.precision === 'f32' ? 0 : snapshot.precision === 'double-float' ? 1 : 2;
    unsigned[15] = snapshot.reference?.length ?? 1;
    signed[16] = snapshot.camera.scale.exponent;
    return data;
  }

  private static async assertShaderValid(module: GPUShaderModule, label: string): Promise<void> {
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message: GPUCompilationMessage) => message.type === 'error');
    if (errors.length === 0) return;
    const first = errors[0];
    throw new Error(`${label} WGSL line ${first.lineNum}: ${first.message}`);
  }
}
