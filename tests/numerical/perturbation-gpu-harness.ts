import { tilePerturbationShader } from '../../src/numerical/tilePerturbationShader';

declare global {
  interface Window { __PERTURBATION_ORACLE__: Promise<unknown>; }
}

const TILE_SIZE = 3;
const ORACLE_BITS = 512;
const STATUS_ACTIVE = 0;
const STATUS_ESCAPED = 1;

type Scenario = Readonly<{
  name: string;
  centerXRaw: bigint;
  centerYRaw: bigint;
  centerBits: number;
  sampleExponent: number;
  target: number;
}>;

const scenarios: readonly Scenario[] = [
  {
    name: 'periodic-c-i-1e35',
    centerXRaw: 0n,
    centerYRaw: 1n << BigInt(ORACLE_BITS),
    centerBits: ORACLE_BITS,
    sampleExponent: -126,
    target: 1000
  },
  {
    name: 'user-boundary-1e16',
    centerXRaw: -4630493738332497355489550221676732813543869506412596953088n,
    centerYRaw: -1062389749950687863094168984335460249475021418675417645056n,
    centerBits: 192,
    sampleExponent: -64,
    target: 1200
  }
];

function shiftRounded(value: bigint, shift: number): bigint {
  if (shift >= 0) return value << BigInt(shift);
  const amount = BigInt(-shift);
  const half = 1n << (amount - 1n);
  return value >= 0n ? (value + half) >> amount : -((-value + half) >> amount);
}

function multiplyFixed(a: bigint, b: bigint): bigint {
  return shiftRounded(a * b, -ORACLE_BITS);
}

function fixedToNumber(raw: bigint, bits: number): number {
  if (raw === 0n) return 0;
  const sign = raw < 0n ? -1 : 1;
  const magnitude = raw < 0n ? -raw : raw;
  const bitLength = magnitude.toString(2).length;
  const shift = Math.max(0, bitLength - 53);
  return sign * Number(magnitude >> BigInt(shift)) * Math.pow(2, shift - bits);
}

function numberToFixed(value: number, bits: number): bigint {
  if (value === 0) return 0n;
  const sign = value < 0 ? -1n : 1n;
  const magnitude = Math.abs(value);
  const exponent = Math.floor(Math.log2(magnitude));
  const mantissa = magnitude / Math.pow(2, exponent);
  const integerMantissa = BigInt(Math.round(mantissa * Math.pow(2, 53)));
  return sign * shiftRounded(integerMantissa, bits + exponent - 53);
}

function splitFixedF32(raw: bigint, bits: number): readonly number[] {
  let remaining = raw;
  const result: number[] = [];
  for (let index = 0; index < 8; index++) {
    const limb = Math.fround(fixedToNumber(remaining, bits));
    result.push(limb);
    remaining -= numberToFixed(limb, bits);
  }
  return result;
}

function exactCenter(scenario: Scenario): readonly [bigint, bigint] {
  return [
    shiftRounded(scenario.centerXRaw, ORACLE_BITS - scenario.centerBits),
    shiftRounded(scenario.centerYRaw, ORACLE_BITS - scenario.centerBits)
  ];
}

function exactPixel(scenario: Scenario, pixelX: number, pixelY: number) {
  const [centerX, centerY] = exactCenter(scenario);
  const step = 1n << BigInt(ORACLE_BITS + scenario.sampleExponent);
  const cx = centerX + BigInt(pixelX - 1) * step;
  const cy = centerY + BigInt(pixelY - 1) * step;
  const escapeSquared = 256n << BigInt(ORACLE_BITS * 2);
  let zx = 0n;
  let zy = 0n;
  for (let iteration = 0; iteration <= scenario.target; iteration++) {
    if (zx * zx + zy * zy > escapeSquared) return { status: STATUS_ESCAPED, iteration };
    if (iteration === scenario.target) return { status: STATUS_ACTIVE, iteration };
    const nextX = multiplyFixed(zx, zx) - multiplyFixed(zy, zy) + cx;
    const nextY = 2n * multiplyFixed(zx, zy) + cy;
    zx = nextX;
    zy = nextY;
  }
  throw new Error('Unreachable exact iteration state');
}

function referenceOrbit(scenario: Scenario): { values: Float32Array; length: number } {
  const [cx, cy] = exactCenter(scenario);
  const escapeSquared = 256n << BigInt(ORACLE_BITS * 2);
  let zx = 0n;
  let zy = 0n;
  const values: number[] = [];
  for (let index = 0; index <= scenario.target; index++) {
    values.push(...splitFixedF32(zx, ORACLE_BITS), ...splitFixedF32(zy, ORACLE_BITS));
    if (zx * zx + zy * zy > escapeSquared || index === scenario.target) break;
    const nextX = multiplyFixed(zx, zx) - multiplyFixed(zy, zy) + cx;
    const nextY = 2n * multiplyFixed(zx, zy) + cy;
    zx = nextX;
    zy = nextY;
  }
  return { values: new Float32Array(values), length: values.length / 16 };
}

function splitNumber(value: number): readonly [number, number] {
  const hi = Math.fround(value);
  return [hi, Math.fround(value - hi)];
}

function uniformData(scenario: Scenario, orbitLength: number): ArrayBuffer {
  const data = new ArrayBuffer(96);
  const floats = new Float32Array(data);
  const unsigned = new Uint32Array(data);
  const signed = new Int32Array(data);
  const [centerXHi, centerXLo] = splitNumber(fixedToNumber(scenario.centerXRaw, scenario.centerBits));
  const [centerYHi, centerYLo] = splitNumber(fixedToNumber(scenario.centerYRaw, scenario.centerBits));
  floats[0] = centerXHi;
  floats[1] = centerXLo;
  floats[2] = centerYHi;
  floats[3] = centerYLo;
  signed[8] = scenario.sampleExponent;
  unsigned[9] = TILE_SIZE;
  unsigned[10] = scenario.target;
  unsigned[11] = scenario.target;
  unsigned[12] = orbitLength;
  unsigned[13] = 1;
  unsigned[14] = 16;
  unsigned[15] = 0;
  floats[16] = 1e-6;
  return data;
}

async function runScenario(device: GPUDevice, pipeline: GPUComputePipeline, scenario: Scenario) {
  const pixelCount = TILE_SIZE * TILE_SIZE;
  const buffer = (size: number, usage: GPUBufferUsageFlags) => device.createBuffer({ size, usage });
  const uniform = buffer(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const state = buffer(pixelCount * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const meta = buffer(pixelCount * 16,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const counters = buffer(28, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const reference = referenceOrbit(scenario);
  const orbit = buffer(reference.values.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const result = device.createTexture({
    size: [TILE_SIZE, TILE_SIZE], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING
  });
  const quality = device.createTexture({
    size: [TILE_SIZE, TILE_SIZE], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING
  });
  const readback = buffer(pixelCount * 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  device.queue.writeBuffer(uniform, 0, uniformData(scenario, reference.length));
  device.queue.writeBuffer(
    orbit,
    0,
    reference.values.buffer as ArrayBuffer,
    reference.values.byteOffset,
    reference.values.byteLength
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
  const comparisons = [];
  for (let y = 0; y < TILE_SIZE; y++) {
    for (let x = 0; x < TILE_SIZE; x++) {
      const index = y * TILE_SIZE + x;
      comparisons.push({
        x, y,
        expected: exactPixel(scenario, x, y),
        gpu: { iteration: mapped[index * 4], status: mapped[index * 4 + 1] }
      });
    }
  }
  return { name: scenario.name, orbitLength: reference.length, comparisons };
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
    type: message.type, message: message.message, lineNum: message.lineNum
  }));
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module, entryPoint: 'main' }
  });
  device.pushErrorScope('validation');
  const scenarioReports = [];
  for (const scenario of scenarios) scenarioReports.push(await runScenario(device, pipeline, scenario));
  const validationError = (await device.popErrorScope())?.message ?? null;
  return {
    adapter: adapter.info.vendor || adapter.info.description || 'GPU',
    compilationMessages,
    validationError,
    uncaptured,
    scenarios: scenarioReports,
    comparisons: scenarioReports.flatMap(report => report.comparisons.map(comparison => ({
      scenario: report.name, orbitLength: report.orbitLength, ...comparison
    })))
  };
}

window.__PERTURBATION_ORACLE__ = run();
