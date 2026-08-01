/// <reference lib="webworker" />

import type { ReferenceRequest, ReferenceResponse } from './referenceProtocol';

const worker = self as unknown as DedicatedWorkerGlobalScope;
const NUMBER_MANTISSA_BITS = 53;
const YIELD_INTERVAL = 64;
let newestRequestId = 0;

function fixedToNumber(raw: bigint, bits: number): number {
  if (raw === 0n) return 0;
  const sign = raw < 0n ? -1 : 1;
  const magnitude = raw < 0n ? -raw : raw;
  const bitLength = magnitude.toString(2).length;
  const shift = Math.max(0, bitLength - NUMBER_MANTISSA_BITS);
  return sign * Number(magnitude >> BigInt(shift)) * Math.pow(2, shift - bits);
}

function splitF32(value: number): [number, number] {
  const hi = Math.fround(value);
  return [hi, Math.fround(value - hi)];
}

function multiplyFixed(a: bigint, b: bigint, bits: number): bigint {
  const product = a * b;
  const shift = BigInt(bits);
  const half = 1n << (shift - 1n);
  return product >= 0n ? (product + half) >> shift : -((-product + half) >> shift);
}

function nextTask(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

async function buildReference(request: ReferenceRequest): Promise<void> {
  const started = performance.now();
  const bits = Math.max(request.centerX.bits, request.centerY.bits);
  const align = (raw: bigint, sourceBits: number): bigint => sourceBits === bits
    ? raw
    : sourceBits < bits
      ? raw << BigInt(bits - sourceBits)
      : raw >> BigInt(sourceBits - bits);

  const cx = align(BigInt(request.centerX.raw), request.centerX.bits);
  const cy = align(BigInt(request.centerY.raw), request.centerY.bits);
  const escapeComponent = 16n << BigInt(bits);
  let zx = 0n;
  let zy = 0n;
  let length = 0;
  let escaped = false;
  const orbit = new Float32Array((request.iterations + 1) * 4);

  for (let i = 0; i <= request.iterations; i++) {
    if ((i & (YIELD_INTERVAL - 1)) === 0) {
      await nextTask();
      if (request.id !== newestRequestId) return;
    }

    const zxNumber = fixedToNumber(zx, bits);
    const zyNumber = fixedToNumber(zy, bits);
    const [zxHi, zxLo] = splitF32(zxNumber);
    const [zyHi, zyLo] = splitF32(zyNumber);
    const offset = i * 4;
    orbit[offset] = zxHi;
    orbit[offset + 1] = zxLo;
    orbit[offset + 2] = zyHi;
    orbit[offset + 3] = zyLo;
    length = i + 1;
    if (i === request.iterations) break;

    const zx2 = multiplyFixed(zx, zx, bits);
    const zy2 = multiplyFixed(zy, zy, bits);
    const zxy = multiplyFixed(zx, zy, bits);
    zx = zx2 - zy2 + cx;
    zy = (zxy << 1n) + cy;

    if (zx > escapeComponent || zx < -escapeComponent || zy > escapeComponent || zy < -escapeComponent) {
      escaped = true;
      break;
    }
  }

  if (request.id !== newestRequestId) return;
  const response: ReferenceResponse = {
    id: request.id,
    bits,
    length,
    escaped,
    generationMs: performance.now() - started,
    orbit: orbit.slice(0, length * 4)
  };
  worker.postMessage(response, [response.orbit.buffer]);
}

worker.addEventListener('message', event => {
  const request = event.data as ReferenceRequest;
  newestRequestId = request.id;
  void buildReference(request);
});
