import { tileDirectIterationShader } from '../../src/numerical/tileDirectShader';

type PointCase = {
  id: string;
  cx: number;
  cy: number;
  targetIterations: number;
  schedules: Array<{ id: string; chunks: number[] }>;
};

type BenchmarkCase = {
  id: string;
  centerX: number;
  centerY: number;
  sampleExponent: number;
  tileSize: number;
  targetIterations: number;
  chunkIterations: number;
  warmupRuns: number;
  measuredRuns: number;
};

type HarnessInput = {
  strictCases: PointCase[];
  sensitivityCases: PointCase[];
  benchmark: BenchmarkCase;
};

type GpuRunInput = {
  id: string;
  centerX: number;
  centerY: number;
  sampleExponent: number;
  tileSize: number;
  targetIterations: number;
  chunks: number[];
  readResult: boolean;
  separateTiming: boolean;
};

const status = document.querySelector<HTMLParagraphElement>('#status');
const GPU_BUFFER_USAGE = GPUBufferUsage;
const GPU_TEXTURE_USAGE = GPUTextureUsage;

function splitF32(value: number): [number, number] {
  const hi = Math.fround(value);
  return [hi, Math.fround(value - hi)];
}

function createUniform(device: GPUDevice, input: GpuRunInput, chunkIterations: number): GPUBuffer {
  const data = new ArrayBuffer(64);
  const view = new DataView(data);
  const [centerXHi, centerXLo] = splitF32(input.centerX);
  const [centerYHi, centerYLo] = splitF32(input.centerY);
  view.setFloat32(0, centerXHi, true);
  view.setFloat32(4, centerXLo, true);
  view.setFloat32(8, centerYHi, true);
  view.setFloat32(12, centerYLo, true);
  view.setInt32(16, input.sampleExponent, true);
  view.setUint32(20, input.tileSize, true);
  view.setUint32(24, input.targetIterations, true);
  view.setUint32(28, chunkIterations, true);
  view.setUint32(32, 0, true);
  view.setUint32(36, 1, true);
  const buffer = device.createBuffer({
    label: `${input.id}-direct-uniform-${chunkIterations}`,
    size: 64,
    usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function mappedCopy<T extends Uint32Array | Float32Array | Uint8Array>(
  buffer: GPUBuffer,
  create: (copy: ArrayBuffer) => T
): T {
  const copy = buffer.getMappedRange().slice(0);
  buffer.unmap();
  return create(copy);
}

async function runGpu(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  input: GpuRunInput
) {
  const pixelCount = input.tileSize * input.tileSize;
  const stateBytes = pixelCount * 16;
  const metaBytes = pixelCount * 16;
  const counterBytes = 28;
  const rowBytes = input.tileSize * 16;
  const resultBytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const resultReadbackBytes = resultBytesPerRow * input.tileSize;
  const resources: Array<{ destroy(): void }> = [];
  const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const value = device.createBuffer({ label, size, usage });
    resources.push(value);
    return value;
  };
  const stateBuffer = buffer(`${input.id}-state`, stateBytes,
    GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC | GPU_BUFFER_USAGE.COPY_DST);
  const metaBuffer = buffer(`${input.id}-meta`, metaBytes,
    GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC | GPU_BUFFER_USAGE.COPY_DST);
  const counterBuffer = buffer(`${input.id}-counters`, counterBytes,
    GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC | GPU_BUFFER_USAGE.COPY_DST);
  const resultTexture = device.createTexture({
    label: `${input.id}-result`,
    size: [input.tileSize, input.tileSize],
    format: 'rgba32float',
    usage: GPU_TEXTURE_USAGE.STORAGE_BINDING | GPU_TEXTURE_USAGE.COPY_SRC
  });
  const qualityTexture = device.createTexture({
    label: `${input.id}-quality`,
    size: [input.tileSize, input.tileSize],
    format: 'rgba8unorm',
    usage: GPU_TEXTURE_USAGE.STORAGE_BINDING | GPU_TEXTURE_USAGE.COPY_SRC
  });
  resources.push(resultTexture, qualityTexture);
  const stateReadback = buffer(`${input.id}-state-readback`, stateBytes,
    GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ);
  const metaReadback = buffer(`${input.id}-meta-readback`, metaBytes,
    GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ);
  const counterReadback = buffer(`${input.id}-counter-readback`, counterBytes,
    GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ);
  const resultReadback = input.readResult ? buffer(`${input.id}-result-readback`, resultReadbackBytes,
    GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ) : null;
  const uniforms = input.chunks.map(chunk => createUniform(device, input, chunk));
  resources.push(...uniforms);
  const bindGroups = uniforms.map(uniform => device.createBindGroup({
    label: `${input.id}-bind-group`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: stateBuffer } },
      { binding: 2, resource: { buffer: metaBuffer } },
      { binding: 3, resource: resultTexture.createView() },
      { binding: 4, resource: qualityTexture.createView() },
      { binding: 5, resource: { buffer: counterBuffer } }
    ]
  }));

  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('internal');
  device.pushErrorScope('validation');
  const computeEncoder = device.createCommandEncoder({ label: `${input.id}-compute-encoder` });
  computeEncoder.clearBuffer(stateBuffer);
  computeEncoder.clearBuffer(metaBuffer);
  computeEncoder.clearBuffer(counterBuffer);
  for (let index = 0; index < bindGroups.length; index += 1) {
    computeEncoder.clearBuffer(counterBuffer);
    const pass = computeEncoder.beginComputePass({ label: `${input.id}-chunk-${index}` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroups[index]);
    pass.dispatchWorkgroups(Math.ceil(input.tileSize / 8), Math.ceil(input.tileSize / 8));
    pass.end();
  }
  const computeStarted = performance.now();
  device.queue.submit([computeEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const queueCompletionWallMs = performance.now() - computeStarted;

  const readbackStarted = performance.now();
  const readbackEncoder = device.createCommandEncoder({ label: `${input.id}-readback-encoder` });
  readbackEncoder.copyBufferToBuffer(stateBuffer, 0, stateReadback, 0, stateBytes);
  readbackEncoder.copyBufferToBuffer(metaBuffer, 0, metaReadback, 0, metaBytes);
  readbackEncoder.copyBufferToBuffer(counterBuffer, 0, counterReadback, 0, counterBytes);
  if (resultReadback) {
    readbackEncoder.copyTextureToBuffer(
      { texture: resultTexture },
      { buffer: resultReadback, bytesPerRow: resultBytesPerRow, rowsPerImage: input.tileSize },
      [input.tileSize, input.tileSize]
    );
  }
  device.queue.submit([readbackEncoder.finish()]);
  await Promise.all([
    stateReadback.mapAsync(GPUMapMode.READ),
    metaReadback.mapAsync(GPUMapMode.READ),
    counterReadback.mapAsync(GPUMapMode.READ),
    resultReadback?.mapAsync(GPUMapMode.READ)
  ]);
  const readbackWallMs = performance.now() - readbackStarted;
  const state = mappedCopy(stateReadback, copy => new Float32Array(copy));
  const meta = mappedCopy(metaReadback, copy => new Uint32Array(copy));
  const counters = mappedCopy(counterReadback, copy => new Uint32Array(copy));
  const result = resultReadback
    ? mappedCopy(resultReadback, copy => {
      const padded = new Float32Array(copy);
      const packed = new Float32Array(pixelCount * 4);
      const sourceStride = resultBytesPerRow / 4;
      for (let y = 0; y < input.tileSize; y += 1) {
        packed.set(padded.subarray(y * sourceStride, y * sourceStride + input.tileSize * 4), y * input.tileSize * 4);
      }
      return packed;
    })
    : new Float32Array();
  let explicitIterations = 0;
  for (let index = 0; index < meta.length; index += 4) explicitIterations += meta[index];
  const stateBits = new Uint32Array(state.buffer, state.byteOffset, state.length);
  const resultBits = new Uint32Array(result.buffer, result.byteOffset, result.length);
  const validation = await device.popErrorScope();
  const internal = await device.popErrorScope();
  const outOfMemory = await device.popErrorScope();
  const scopedErrors = [validation, internal, outOfMemory]
    .filter((error): error is GPUError => Boolean(error))
    .map(error => error.message);
  for (const resource of resources.reverse()) resource.destroy();
  return {
    state: input.readResult ? Array.from(state) : [],
    stateBits: input.readResult ? Array.from(stateBits) : [],
    meta: input.readResult ? Array.from(meta) : [],
    counters: Array.from(counters),
    result: input.readResult ? Array.from(result) : [],
    resultBits: input.readResult ? Array.from(resultBits) : [],
    explicitIterations,
    queueCompletionWallMs,
    readbackWallMs,
    separateTiming: input.separateTiming,
    scopedErrors
  };
}

async function runSuite(input: HarnessInput) {
  if (!navigator.gpu) throw new Error('WebGPU is unavailable');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found');
  const device = await adapter.requestDevice();
  const uncapturedErrors: string[] = [];
  let deviceLostBeforeCompletion = false;
  let completed = false;
  device.addEventListener('uncapturederror', event => uncapturedErrors.push(event.error.message));
  void device.lost.then(info => {
    if (!completed) deviceLostBeforeCompletion = true;
    uncapturedErrors.push(`device-lost:${info.reason}:${info.message}`);
  });
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label: 'production-direct-shader', code: tileDirectIterationShader });
  const compilation = await module.getCompilationInfo();
  const compilationMessages = compilation.messages.map(message => ({
    type: message.type,
    message: message.message,
    lineNum: message.lineNum,
    linePos: message.linePos
  }));
  const pipeline = await device.createComputePipelineAsync({
    label: 'production-direct-pipeline',
    layout: 'auto',
    compute: { module, entryPoint: 'main' }
  });
  const pipelineValidation = await device.popErrorScope();

  const strictResults = [];
  for (const testCase of input.strictCases) {
    const schedules = [];
    for (const schedule of testCase.schedules) {
      schedules.push({
        id: schedule.id,
        chunks: schedule.chunks,
        run: await runGpu(device, pipeline, {
          id: `${testCase.id}-${schedule.id}`,
          centerX: testCase.cx,
          centerY: testCase.cy,
          sampleExponent: 0,
          tileSize: 1,
          targetIterations: testCase.targetIterations,
          chunks: schedule.chunks,
          readResult: true,
          separateTiming: true
        })
      });
    }
    strictResults.push({ id: testCase.id, schedules });
  }

  const sensitivityResults = [];
  for (const testCase of input.sensitivityCases) {
    const schedule = testCase.schedules[0];
    sensitivityResults.push({
      id: testCase.id,
      run: await runGpu(device, pipeline, {
        id: testCase.id,
        centerX: testCase.cx,
        centerY: testCase.cy,
        sampleExponent: 0,
        tileSize: 1,
        targetIterations: testCase.targetIterations,
        chunks: schedule.chunks,
        readResult: true,
        separateTiming: true
      })
    });
  }

  const benchmarkChunks = [];
  let benchmarkRemaining = input.benchmark.targetIterations;
  while (benchmarkRemaining > 0) {
    const chunk = Math.min(input.benchmark.chunkIterations, benchmarkRemaining);
    benchmarkChunks.push(chunk);
    benchmarkRemaining -= chunk;
  }
  const benchmarkRuns = [];
  const totalRuns = input.benchmark.warmupRuns + input.benchmark.measuredRuns;
  for (let index = 0; index < totalRuns; index += 1) {
    const run = await runGpu(device, pipeline, {
      id: `${input.benchmark.id}-${index}`,
      centerX: input.benchmark.centerX,
      centerY: input.benchmark.centerY,
      sampleExponent: input.benchmark.sampleExponent,
      tileSize: input.benchmark.tileSize,
      targetIterations: input.benchmark.targetIterations,
      chunks: benchmarkChunks,
      readResult: false,
      separateTiming: true
    });
    benchmarkRuns.push({ warmup: index < input.benchmark.warmupRuns, ...run });
  }
  await new Promise(resolve => setTimeout(resolve, 0));
  completed = true;
  if (status) status.textContent = 'Harness complete';
  return {
    adapter: {
      info: {
        vendor: adapter.info.vendor,
        architecture: adapter.info.architecture,
        device: adapter.info.device,
        description: adapter.info.description
      },
      features: Array.from(adapter.features).sort(),
      limits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
      }
    },
    compilationMessages,
    pipelineValidationError: pipelineValidation?.message ?? null,
    uncapturedErrors,
    deviceLostBeforeCompletion,
    strictResults,
    sensitivityResults,
    benchmark: {
      definition: input.benchmark,
      chunks: benchmarkChunks,
      runs: benchmarkRuns
    }
  };
}

Object.assign(window, { __NUMERICAL_HARNESS__: { runSuite } });
if (status) status.textContent = 'Harness ready';
