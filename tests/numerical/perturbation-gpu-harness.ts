import { tilePerturbationShader } from '../../src/numerical/tilePerturbationShader';

declare global {
  interface Window { __PERTURBATION_ORACLE__: Promise<unknown>; }
}

const TILE_SIZE = 3;
const TARGET = 1000;
const SAMPLE_EXPONENT = -126;
const ORACLE_BITS = 512;
const STATUS_ESCAPED = 1;

function shiftRounded(value: bigint, shift: number): bigint {
  if (shift >= 0) return value << BigInt(shift);
  const amount = BigInt(-shift);
  const half = 1n << (amount - 1n);
  return value >= 0n ? (value + half) >> amount : -((-value + half) >> amount);
}

function multiplyFixed(a: bigint, b: bigint): bigint {
  return shiftRounded(a * b, -ORACLE_BITS);
}

function exactPixel(pixelX: number, pixelY: number) {
  const step = 1n << BigInt(ORACLE_BITS + SAMPLE_EXPONENT);
  const cx = BigInt(pixelX - 1) * step;
  const cy = (1n << BigInt(ORACLE_BITS)) + BigInt(pixelY - 1) * step;
  const escapeSquared = 256n << BigInt(ORACLE_BITS * 2);
  let zx = 0n;
  let zy = 0n;
  for (let iteration = 0; iteration <= TARGET; iteration++) {
    if (zx * zx + zy * zy > escapeSquared) return { status: STATUS_ESCAPED, iteration };
    if (iteration === TARGET) return { status: 0, iteration };
    const nextX = multiplyFixed(zx, zx) - multiplyFixed(zy, zy) + cx;
    const nextY = 2n * multiplyFixed(zx, zy) + cy;
    zx = nextX;
    zy = nextY;
  }
  throw new Error('Unreachable exact iteration state');
}

function referenceOrbit(): Float32Array {
  const orbit = new Float32Array((TARGET + 1) * 16);
  for (let index = 0; index <= TARGET; index++) {
    const offset = index * 16;
    if (index === 1) orbit[offset + 8] = 1;
    else if (index >= 2 && index % 2 === 0) {
      orbit[offset] = -1;
      orbit[offset + 8] = 1;
    } else if (index >= 3) orbit[offset + 8] = -1;
  }
  return orbit;
}

function uniformData(): ArrayBuffer {
  const data = new ArrayBuffer(96);
  const floats = new Float32Array(data);
  const unsigned = new Uint32Array(data);
  const signed = new Int32Array(data);
  floats[2] = 1;
  signed[8] = SAMPLE_EXPONENT;
  unsigned[9] = TILE_SIZE;
  unsigned[10] = TARGET;
  unsigned[11] = TARGET;
  unsigned[12] = TARGET + 1;
  unsigned[13] = 1;
  unsigned[14] = 16;
  unsigned[15] = 0;
  floats[16] = 1e-6;
  return data;
}

async function run() {
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();
  const uncaptured: string[] = [];
  device.addEventListener('uncapturederror', event => uncaptured.push(event.error.message));
  const module = device.createShaderModule({ code: tilePerturbationShader });
  const compilationMessages = (await module.getCompilationInfo()).messages.map(message => ({
    type: message.type,
    message: message.message,
    lineNum: message.lineNum
  }));
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module, entryPoint: 'main' }
  });
  const pixelCount = TILE_SIZE * TILE_SIZE;
  const buffer = (size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ size, usage });
  const uniform = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const state = buffer(pixelCount * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const meta = buffer(pixelCount * 16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const counters = buffer(28, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const orbitValues = referenceOrbit();
  const orbit = buffer(orbitValues.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const result = device.createTexture({
    size: [TILE_SIZE, TILE_SIZE], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING
  });
  const quality = device.createTexture({
    size: [TILE_SIZE, TILE_SIZE], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING
  });
  const readback = buffer(pixelCount * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  device.queue.writeBuffer(uniform, 0, uniformData());
  device.queue.writeBuffer(
    orbit,
    0,
    orbitValues.buffer as ArrayBuffer,
    orbitValues.byteOffset,
    orbitValues.byteLength
  );
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: state } },
      { binding: 2, resource: { buffer: meta } },
      { binding: 3, resource: result.createView() },
      { binding: 4, resource: quality.createView() },
      { binding: 5, resource: { buffer: counters } },
      { binding: 6, resource: { buffer: orbit } }
    ]
  });
  device.pushErrorScope('validation');
  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(state);
  encoder.clearBuffer(meta);
  encoder.clearBuffer(counters);
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1, 1);
  pass.end();
  encoder.copyBufferToBuffer(meta, 0, readback, 0, pixelCount * 16);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  const validationError = (await device.popErrorScope())?.message ?? null;
  const comparisons = [];
  for (let y = 0; y < TILE_SIZE; y++) {
    for (let x = 0; x < TILE_SIZE; x++) {
      const index = y * TILE_SIZE + x;
      const expected = exactPixel(x, y);
      comparisons.push({
        x, y, expected,
        gpu: { iteration: mapped[index * 4], status: mapped[index * 4 + 1] }
      });
    }
  }
  return {
    adapter: adapter.info.vendor || adapter.info.description || 'GPU',
    compilationMessages,
    validationError,
    uncaptured,
    comparisons
  };
}

window.__PERTURBATION_ORACLE__ = run();
