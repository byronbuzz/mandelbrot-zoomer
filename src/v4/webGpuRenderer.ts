import { fixedDifferenceToNumber, fixedSplitF32 } from '../bigFixed';
import { scaleToNumber } from '../binaryScale';
import { fluidPresentShader } from '../v5/fluidPresentShader';
import { RenderBatchPolicy } from '../v5/renderBatchPolicy';
import { composeShader, computeShader } from './shaders';
import type { CpuReference, GpuReference, PreparedFrame, RenderSnapshot, RenderTelemetry } from './types';

const PARAMETER_BYTES = 80;
const COMPOSE_PARAMETER_BYTES = 16;
const PRESENT_PARAMETER_BYTES = 32;
const TILE_COLUMNS = 16;
const TILE_ROWS = 16;
const TELEMETRY_COUNTER_VALUES = 5;
const TELEMETRY_TILE_OFFSET = TELEMETRY_COUNTER_VALUES;
const TELEMETRY_MAX_EXPONENT_INDEX = TELEMETRY_TILE_OFFSET + TILE_COLUMNS * TILE_ROWS;
const TELEMETRY_VALUES = TELEMETRY_MAX_EXPONENT_INDEX + 1;
const TELEMETRY_BYTES = TELEMETRY_VALUES * Uint32Array.BYTES_PER_ELEMENT;
const COMPOSE_CLEAR_BYTES = TELEMETRY_MAX_EXPONENT_INDEX * Uint32Array.BYTES_PER_ELEMENT;
const COMPUTE_TEXTURE_FORMAT: GPUTextureFormat = 'rgba8unorm';
const MAX_POOLED_TEXTURES = 6;
const BATCH_ITERATION_THRESHOLD = 3000;
const BATCH_NOMINAL_WORK_THRESHOLD = 12_000_000_000;

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
  private readonly computeUniform: GPUBuffer;
  private readonly composeUniform: GPUBuffer;
  private readonly telemetryBuffer: GPUBuffer;
  private readonly telemetryReadback: GPUBuffer;
  private readonly presentationUniform: GPUBuffer;
  private readonly label: string;
  private readonly texturePool = new Map<string, GPUTexture[]>();
  private readonly textureKeys = new WeakMap<GPUTexture, string>();
  private displayWidth = 1;
  private displayHeight = 1;
  private settledTexture: GPUTexture | null = null;
  private settledKey = '';
  private presentedTexture: GPUTexture | null = null;
  private presentedSnapshot: RenderSnapshot | null = null;

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
    this.sampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge'
    });
    this.fallbackOrbit = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.fallbackOrbit, 0, new Float32Array(8));
    this.computeUniform = device.createBuffer({
      size: PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.composeUniform = device.createBuffer({
      size: COMPOSE_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.telemetryBuffer = device.createBuffer({
      size: TELEMETRY_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.telemetryReadback = device.createBuffer({
      size: TELEMETRY_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    this.presentationUniform = device.createBuffer({
      size: PRESENT_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
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
    const presentModule = device.createShaderModule({ code: fluidPresentShader });
    await WebGpuRenderer.assertShaderValid(presentModule, 'fluid presentation');

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
      size: Math.max(32, reference.orbit.byteLength),
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

  reproject(snapshot: RenderSnapshot): boolean {
    if (!this.presentedTexture || !this.presentedSnapshot) return false;
    const displayWidth = Math.max(1, Math.floor(snapshot.cssWidth * snapshot.devicePixelRatio));
    const displayHeight = Math.max(1, Math.floor(snapshot.cssHeight * snapshot.devicePixelRatio));
    this.resizeCanvas(displayWidth, displayHeight);
    const parameters = this.createPresentationData(this.presentedSnapshot, snapshot, 1, 1);
    if (!parameters) return false;
    this.submitPresentation(this.presentedTexture, this.presentedTexture, parameters);
    return true;
  }

  async prepare(
    snapshot: RenderSnapshot,
    shouldCancel: () => boolean = () => false
  ): Promise<PreparedFrame | null> {
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

    const candidateTexture = this.acquireComputeTexture(computeWidth, computeHeight);
    const outputTexture = canRepair
      ? this.acquireComputeTexture(computeWidth, computeHeight)
      : candidateTexture;
    const parameterData = this.createParameterData(snapshot, computeWidth, computeHeight);
    const parameterWords = new Uint32Array(parameterData);
    this.device.queue.writeBuffer(this.telemetryBuffer, 0, new Uint32Array(TELEMETRY_VALUES));

    const computeBindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.computeUniform } },
        { binding: 1, resource: candidateTexture.createView() },
        { binding: 2, resource: { buffer: snapshot.reference?.buffer ?? this.fallbackOrbit } },
        { binding: 3, resource: { buffer: this.telemetryBuffer } }
      ]
    });

    const nominalWork = computeWidth * computeHeight * snapshot.quality.iterations;
    const useBatches = snapshot.quality.iterations >= BATCH_ITERATION_THRESHOLD
      || nominalWork >= BATCH_NOMINAL_WORK_THRESHOLD;
    const batchPolicy = new RenderBatchPolicy();
    batchPolicy.reset(snapshot.quality.iterations, computeWidth);
    let rowStart = 0;
    let computeBatches = 0;
    const started = performance.now();

    try {
      while (rowStart < computeHeight) {
        if (shouldCancel()) {
          this.releaseFrameTextures(candidateTexture, outputTexture);
          return null;
        }
        const remainingRows = computeHeight - rowStart;
        const rowCount = useBatches ? batchPolicy.currentRows(remainingRows) : remainingRows;
        parameterWords[17] = rowStart;
        this.device.queue.writeBuffer(this.computeUniform, 0, parameterData);

        const encoder = this.device.createCommandEncoder();
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(computeWidth / 8), Math.ceil(rowCount / 8));
        computePass.end();

        const batchStarted = performance.now();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        batchPolicy.observe(performance.now() - batchStarted);
        computeBatches++;
        rowStart += rowCount;
      }

      if (shouldCancel()) {
        this.releaseFrameTextures(candidateTexture, outputTexture);
        return null;
      }

      if (canRepair && this.settledTexture) {
        this.device.queue.writeBuffer(
          this.composeUniform,
          0,
          new Uint32Array([computeWidth, computeHeight, 0, 0])
        );
        const composeBindGroup = this.device.createBindGroup({
          layout: this.composePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.composeUniform } },
            { binding: 1, resource: this.settledTexture.createView() },
            { binding: 2, resource: candidateTexture.createView() },
            { binding: 3, resource: outputTexture.createView() },
            { binding: 4, resource: { buffer: this.telemetryBuffer } }
          ]
        });
        const encoder = this.device.createCommandEncoder();
        encoder.clearBuffer(this.telemetryBuffer, 0, COMPOSE_CLEAR_BYTES);
        const composePass = encoder.beginComputePass();
        composePass.setPipeline(this.composePipeline);
        composePass.setBindGroup(0, composeBindGroup);
        composePass.dispatchWorkgroups(Math.ceil(computeWidth / 8), Math.ceil(computeHeight / 8));
        composePass.end();
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
      }

      let telemetry: RenderTelemetry | null = null;
      if (captureTelemetry) {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.telemetryBuffer, 0, this.telemetryReadback, 0, TELEMETRY_BYTES);
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        telemetry = await this.readTelemetry(this.telemetryReadback, computeWidth * computeHeight);
      }

      if (canRepair) this.releaseTexture(candidateTexture);
      return {
        snapshot,
        texture: outputTexture,
        computeWidth,
        computeHeight,
        displayWidth,
        displayHeight,
        computeMs: Math.max(0.1, performance.now() - started),
        computeBatches,
        telemetry,
        retainAsSettled: captureTelemetry,
        accumulationKey
      };
    } catch (error) {
      this.retireTextures(new Set([candidateTexture, outputTexture]));
      throw error;
    }
  }

  async present(frame: PreparedFrame): Promise<number> {
    const historyTexture = this.presentedTexture ?? frame.texture;
    const historySnapshot = this.presentedSnapshot ?? frame.snapshot;
    const newWeight = this.presentedTexture
      ? frame.snapshot.stage === 'interactive'
        ? .72
        : frame.snapshot.stage === 'refining'
          ? .88
          : 1
      : 1;
    const parameters = this.createPresentationData(historySnapshot, frame.snapshot, newWeight, 0)
      ?? this.createIdentityPresentationData(1, 0);
    const started = performance.now();
    this.submitPresentation(frame.texture, historyTexture, parameters);
    await this.device.queue.onSubmittedWorkDone();
    this.acceptPresentedFrame(frame);
    return Math.max(0.1, performance.now() - started);
  }

  discard(frame: PreparedFrame): void {
    this.retireTextures(new Set([frame.texture]));
  }

  private submitPresentation(newTexture: GPUTexture, historyTexture: GPUTexture, parameters: ArrayBuffer): void {
    this.device.queue.writeBuffer(this.presentationUniform, 0, parameters);
    const bindGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.presentationUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: newTexture.createView() },
        { binding: 3, resource: historyTexture.createView() }
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
    this.device.queue.submit([encoder.finish()]);
  }

  private acceptPresentedFrame(frame: PreparedFrame): void {
    const oldPresented = this.presentedTexture;
    const oldSettled = this.settledTexture;

    this.presentedTexture = frame.texture;
    this.presentedSnapshot = frame.snapshot;

    if (frame.retainAsSettled) {
      this.settledTexture = frame.texture;
      this.settledKey = frame.accumulationKey;
    } else if (frame.snapshot.stage === 'full-quality' && frame.snapshot.precision !== 'perturbation') {
      this.settledTexture = null;
      this.settledKey = '';
    }

    const retired = new Set<GPUTexture>();
    if (oldPresented) retired.add(oldPresented);
    if (oldSettled) retired.add(oldSettled);
    this.retireTextures(retired);
  }

  private retireTextures(candidates: ReadonlySet<GPUTexture>): void {
    const batch = [...candidates].filter(texture => (
      texture !== this.presentedTexture && texture !== this.settledTexture
    ));
    if (batch.length === 0) return;
    void this.device.queue.onSubmittedWorkDone().then(() => {
      for (const texture of batch) {
        if (texture !== this.presentedTexture && texture !== this.settledTexture) this.releaseTexture(texture);
      }
    }).catch(error => console.error('Unable to retire presentation textures', error));
  }

  private releaseFrameTextures(candidateTexture: GPUTexture, outputTexture: GPUTexture): void {
    const textures = new Set([candidateTexture, outputTexture]);
    for (const texture of textures) this.releaseTexture(texture);
  }

  private acquireComputeTexture(width: number, height: number): GPUTexture {
    const key = `${width}x${height}`;
    const available = this.texturePool.get(key);
    const texture = available?.pop() ?? this.device.createTexture({
      size: [width, height],
      format: COMPUTE_TEXTURE_FORMAT,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    this.textureKeys.set(texture, key);
    return texture;
  }

  private releaseTexture(texture: GPUTexture): void {
    if (texture === this.presentedTexture || texture === this.settledTexture) return;
    const key = this.textureKeys.get(texture);
    if (!key || this.pooledTextureCount() >= MAX_POOLED_TEXTURES) {
      texture.destroy();
      return;
    }
    const available = this.texturePool.get(key) ?? [];
    if (available.length >= 2) {
      texture.destroy();
      return;
    }
    available.push(texture);
    this.texturePool.set(key, available);
  }

  private pooledTextureCount(): number {
    let total = 0;
    for (const textures of this.texturePool.values()) total += textures.length;
    return total;
  }

  private createPresentationData(
    source: RenderSnapshot,
    target: RenderSnapshot,
    newWeight: number,
    mode: number
  ): ArrayBuffer | null {
    const sourceScale = Math.max(scaleToNumber(source.camera.scale), Number.MIN_VALUE);
    const scaleRatio = target.camera.scale.mantissa / source.camera.scale.mantissa
      * Math.pow(2, target.camera.scale.exponent - source.camera.scale.exponent);
    const sourceAspect = Math.max(1, source.cssWidth) / Math.max(1, source.cssHeight);
    const targetAspect = Math.max(1, target.cssWidth) / Math.max(1, target.cssHeight);
    const offsetX = fixedDifferenceToNumber(target.camera.centerX, source.camera.centerX)
      / sourceScale
      / sourceAspect;
    const offsetY = fixedDifferenceToNumber(target.camera.centerY, source.camera.centerY)
      / sourceScale;
    const scaleX = scaleRatio * targetAspect / sourceAspect;
    const scaleY = scaleRatio;
    const values = [scaleX, scaleY, offsetX, offsetY, newWeight];
    if (values.some(value => !Number.isFinite(value) || Math.abs(value) > 3.3e38)) return null;

    const data = new ArrayBuffer(PRESENT_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    floats[0] = scaleX;
    floats[1] = scaleY;
    floats[2] = offsetX;
    floats[3] = offsetY;
    floats[4] = newWeight;
    unsigned[5] = mode;
    return data;
  }

  private createIdentityPresentationData(newWeight: number, mode: number): ArrayBuffer {
    const data = new ArrayBuffer(PRESENT_PARAMETER_BYTES);
    const floats = new Float32Array(data);
    const unsigned = new Uint32Array(data);
    floats[0] = 1;
    floats[1] = 1;
    floats[4] = newWeight;
    unsigned[5] = mode;
    return data;
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
    unsigned[17] = 0;
    return data;
  }

  private async readTelemetry(buffer: GPUBuffer, totalPixels: number): Promise<RenderTelemetry> {
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new Uint32Array(buffer.getMappedRange());
    const tileUnresolved = Array.from(values.slice(
      TELEMETRY_TILE_OFFSET,
      TELEMETRY_TILE_OFFSET + TILE_COLUMNS * TILE_ROWS
    ));
    const exponentBits = values[TELEMETRY_MAX_EXPONENT_INDEX] ?? 0;
    const telemetry: RenderTelemetry = {
      unresolvedPixels: values[0] ?? 0,
      exhaustedPixels: values[1] ?? 0,
      magnitudeGuardPixels: values[2] ?? 0,
      nonFinitePixels: values[3] ?? 0,
      rebaseFailurePixels: values[4] ?? 0,
      maxPerturbationExponent: exponentBits === 0 ? null : exponentBits - 127,
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
