import { fixedDifferenceToNumber, fixedSplitF32 } from '../bigFixed';
import { scaleToNumber } from '../binaryScale';
import { composeShader, computeShader, presentShader } from './shaders';
import type { CpuReference, GpuReference, PreparedFrame, RenderSnapshot, RenderTelemetry } from './types';

const PARAMETER_BYTES = 80;
const COMPOSE_PARAMETER_BYTES = 16;
const TILE_COLUMNS = 16;
const TILE_ROWS = 16;
const TELEMETRY_VALUES = 2 + TILE_COLUMNS * TILE_ROWS;
const TELEMETRY_BYTES = TELEMETRY_VALUES * Uint32Array.BYTES_PER_ELEMENT;
const COMPUTE_TEXTURE_FORMAT: GPUTextureFormat = 'rgba8unorm';

export class WebGpuRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly canvasFormat: GPUTextureFormat;
  private readonly computePipeline: GPUComputePipeline;
  private readonly composePipeline: GPUComputePipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly fallbackOrbit: GPUBuffer;
  private readonly label: string;
  private displayWidth = 1;
  private displayHeight = 1;
  private settledTexture: GPUTexture | null = null;
  private settledKey = '';

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    canvasFormat: GPUTextureFormat,
    computePipeline: GPUComputePipeline,
    composePipeline: GPUComputePipeline,
    presentPipeline: GPURenderPipeline,
    label: string
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.canvasFormat = canvasFormat;
    this.computePipeline = computePipeline;
    this.composePipeline = composePipeline;
    this.presentPipeline = presentPipeline;
    this.label = label;
    this.sampler = device.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });
    this.fallbackOrbit = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.fallbackOrbit, 0, new Float32Array(8));
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
    const composeModule = device.createShaderModule({ code: composeShader });
    await WebGpuRenderer.assertShaderValid(composeModule, 'compose');
    const presentModule = device.createShaderModule({ code: presentShader });
    await WebGpuRenderer.assertShaderValid(presentModule, 'present');

    const computePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: computeModule, entryPoint: 'main' }
    });
    const composePipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module: composeModule, entryPoint: 'main' }
    });
    const presentPipeline = await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module: presentModule, entryPoint: 'vertexMain' },
      fragment: { module: presentModule, entryPoint: 'fragmentMain', targets: [{ format: canvasFormat }] },
      primitive: { topology: 'triangle-list' }
    });
    const vendor = adapter.info.vendor || adapter.info.description || 'GPU';
    return new WebGpuRenderer(
      canvas,
      device,
      context,
      canvasFormat,
      computePipeline,
      composePipeline,
      presentPipeline,
      vendor
    );
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
    const accumulationKey = this.accumulationKey(snapshot, computeWidth, computeHeight);
    const captureTelemetry = snapshot.stage === 'full-quality' && snapshot.precision === 'perturbation';
    const repairRequested = snapshot.repairPass > 0;
    const canRepair = Boolean(
      repairRequested
      && captureTelemetry
      && this.settledTexture
      && this.settledKey === accumulationKey
    );
    if (repairRequested && !canRepair) {
      throw new Error('Secondary-reference repair base is no longer current');
    }

    const candidateTexture = this.createComputeTexture(computeWidth, computeHeight);
    const outputTexture = canRepair ? this.createComputeTexture(computeWidth, computeHeight) : candidateTexture;
    const uniform = this.device.createBuffer({
      size: PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const telemetryBuffer = this.device.createBuffer({
      size: TELEMETRY_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    const telemetryReadback = captureTelemetry
      ? this.device.createBuffer({ size: TELEMETRY_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
      : null;
    this.device.queue.writeBuffer(uniform, 0, this.createParameterData(snapshot, computeWidth, computeHeight));
    this.device.queue.writeBuffer(telemetryBuffer, 0, new Uint32Array(TELEMETRY_VALUES));

    const computeBindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: candidateTexture.createView() },
        { binding: 2, resource: { buffer: snapshot.reference?.buffer ?? this.fallbackOrbit } },
        { binding: 3, resource: { buffer: telemetryBuffer } }
      ]
    });

    const encoder = this.device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(computeWidth / 8), Math.ceil(computeHeight / 8));
    computePass.end();

    let composeUniform: GPUBuffer | null = null;
    if (canRepair && this.settledTexture) {
      encoder.clearBuffer(telemetryBuffer);
      composeUniform = this.device.createBuffer({
        size: COMPOSE_PARAMETER_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      this.device.queue.writeBuffer(composeUniform, 0, new Uint32Array([computeWidth, computeHeight, 0, 0]));
      const composeBindGroup = this.device.createBindGroup({
        layout: this.composePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: composeUniform } },
          { binding: 1, resource: this.settledTexture.createView() },
          { binding: 2, resource: candidateTexture.createView() },
          { binding: 3, resource: outputTexture.createView() },
          { binding: 4, resource: { buffer: telemetryBuffer } }
        ]
      });
      const composePass = encoder.beginComputePass();
      composePass.setPipeline(this.composePipeline);
      composePass.setBindGroup(0, composeBindGroup);
      composePass.dispatchWorkgroups(Math.ceil(computeWidth / 8), Math.ceil(computeHeight / 8));
      composePass.end();
    }

    if (telemetryReadback) encoder.copyBufferToBuffer(telemetryBuffer, 0, telemetryReadback, 0, TELEMETRY_BYTES);
    const started = performance.now();
    this.device.queue.submit([encoder.finish()]);
    try {
      await this.device.queue.onSubmittedWorkDone();
      const telemetry = telemetryReadback
        ? await this.readTelemetry(telemetryReadback, computeWidth * computeHeight)
        : null;
      uniform.destroy();
      composeUniform?.destroy();
      telemetryBuffer.destroy();
      telemetryReadback?.destroy();
      if (canRepair) candidateTexture.destroy();
      return {
        snapshot,
        texture: outputTexture,
        computeWidth,
        computeHeight,
        displayWidth,
        displayHeight,
        computeMs: Math.max(0.1, performance.now() - started),
        telemetry,
        retainAsSettled: captureTelemetry,
        accumulationKey
      };
    } catch (error) {
      candidateTexture.destroy();
      if (outputTexture !== candidateTexture) outputTexture.destroy();
      uniform.destroy();
      composeUniform?.destroy();
      telemetryBuffer.destroy();
      telemetryReadback?.destroy();
      throw error;
    }
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

    if (frame.retainAsSettled) {
      const previous = this.settledTexture;
      this.settledTexture = frame.texture;
      this.settledKey = frame.accumulationKey;
      if (previous && previous !== frame.texture) previous.destroy();
    } else {
      frame.texture.destroy();
      if (frame.snapshot.stage === 'full-quality' && frame.snapshot.precision !== 'perturbation') {
        this.clearSettledTexture();
      }
    }
    return Math.max(0.1, performance.now() - started);
  }

  discard(frame: PreparedFrame): void { frame.texture.destroy(); }

  private createComputeTexture(width: number, height: number): GPUTexture {
    return this.device.createTexture({
      size: [width, height],
      format: COMPUTE_TEXTURE_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
  }

  private accumulationKey(snapshot: RenderSnapshot, width: number, height: number): string {
    return [
      snapshot.generation,
      snapshot.camera.generation,
      `${width}x${height}`,
      snapshot.quality.iterations,
      snapshot.palettePhase.toFixed(6)
    ].join(':');
  }

  private clearSettledTexture(): void {
    this.settledTexture?.destroy();
    this.settledTexture = null;
    this.settledKey = '';
  }

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
      const viewportScale = Math.max(scaleToNumber(snapshot.camera.scale), Number.MIN_VALUE);
      referenceOffsetX = fixedDifferenceToNumber(snapshot.camera.centerX, snapshot.reference.centerX) / viewportScale;
      referenceOffsetY = fixedDifferenceToNumber(snapshot.camera.centerY, snapshot.reference.centerY) / viewportScale;
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

  private async readTelemetry(buffer: GPUBuffer, totalPixels: number): Promise<RenderTelemetry> {
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new Uint32Array(buffer.getMappedRange());
    const tileUnresolved = Array.from(values.slice(2, 2 + TILE_COLUMNS * TILE_ROWS));
    const telemetry: RenderTelemetry = {
      unresolvedPixels: values[0] ?? 0,
      exhaustedPixels: values[1] ?? 0,
      totalPixels,
      tileColumns: TILE_COLUMNS,
      tileRows: TILE_ROWS,
      tileUnresolved
    };
    buffer.unmap();
    return telemetry;
  }

  private static async assertShaderValid(module: GPUShaderModule, label: string): Promise<void> {
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message: GPUCompilationMessage) => message.type === 'error');
    if (errors.length === 0) return;
    const first = errors[0];
    throw new Error(`${label} WGSL line ${first.lineNum}: ${first.message}`);
  }
}
