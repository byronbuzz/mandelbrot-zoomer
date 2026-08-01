/// <reference lib="webworker" />

import type { ReferenceRequest, ReferenceResponse } from './referenceProtocol';

const worker = self as unknown as DedicatedWorkerGlobalScope;
const NUMBER_MANTISSA_BITS = 53;
const SPLITTER = 134_217_729;

type DD = readonly [number, number];

function shiftRounded(value: bigint, shift: number): bigint {
  if (shift >= 0) return value << BigInt(shift);
  const amount = BigInt(-shift);
  const half = 1n << (amount - 1n);
  return value >= 0n ? (value + half) >> amount : -((-value + half) >> amount);
}

function fixedToNumber(raw: bigint, bits: number): number {
  if (raw === 0n) return 0;
  const sign = raw < 0n ? -1 : 1;
  const magnitude = raw < 0n ? -raw : raw;
  const bitLength = magnitude.toString(2).length;
  const shift = Math.max(0, bitLength - NUMBER_MANTISSA_BITS);
  return sign * Number(magnitude >> BigInt(shift)) * Math.pow(2, shift - bits);
}

function numberToFixed(value: number, bits: number): bigint {
  if (value === 0) return 0n;
  const sign = value < 0 ? -1n : 1n;
  const magnitude = Math.abs(value);
  const exponent = Math.floor(Math.log2(magnitude));
  const mantissa = magnitude / Math.pow(2, exponent);
  const integerMantissa = BigInt(Math.round(mantissa * Math.pow(2, NUMBER_MANTISSA_BITS)));
  return sign * shiftRounded(integerMantissa, bits + exponent - NUMBER_MANTISSA_BITS);
}

function fixedToDD(raw: bigint, bits: number): DD {
  const hi = fixedToNumber(raw, bits);
  const residual = raw - numberToFixed(hi, bits);
  return [hi, fixedToNumber(residual, bits)];
}

function quickTwoSum(a: number, b: number): DD {
  const s = a + b;
  return [s, b - (s - a)];
}

function twoSum(a: number, b: number): DD {
  const s = a + b;
  const bb = s - a;
  return [s, (a - (s - bb)) + (b - bb)];
}

function split(a: number): DD {
  const c = SPLITTER * a;
  const hi = c - (c - a);
  return [hi, a - hi];
}

function twoProduct(a: number, b: number): DD {
  const p = a * b;
  const [ah, al] = split(a);
  const [bh, bl] = split(b);
  const error = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  return [p, error];
}

function ddAdd(a: DD, b: DD): DD {
  const [s, e0] = twoSum(a[0], b[0]);
  const e = e0 + a[1] + b[1];
  return quickTwoSum(s, e);
}

function ddSub(a: DD, b: DD): DD {
  return ddAdd(a, [-b[0], -b[1]]);
}

function ddMul(a: DD, b: DD): DD {
  const [p, e0] = twoProduct(a[0], b[0]);
  const e = e0 + a[0] * b[1] + a[1] * b[0] + a[1] * b[1];
  return quickTwoSum(p, e);
}

function ddScale(a: DD, scale: number): DD {
  return ddMul(a, [scale, 0]);
}

function ddValue(a: DD): number {
  return a[0] + a[1];
}

function splitF32(value: DD): [number, number] {
  const hi = Math.fround(value[0]);
  const residual = (value[0] - hi) + value[1];
  return [hi, Math.fround(residual)];
}

let newestRequestId = 0;

worker.addEventListener('message', event => {
  const request = event.data as ReferenceRequest;
  newestRequestId = Math.max(newestRequestId, request.id);
  void buildReference(request);
});

async function buildReference(request: ReferenceRequest): Promise<void> {
  const started = performance.now();
  const bits = Math.max(request.centerX.bits, request.centerY.bits);
  const cx = fixedToDD(BigInt(request.centerX.raw), request.centerX.bits);
  const cy = fixedToDD(BigInt(request.centerY.raw), request.centerY.bits);
  let zx: DD = [0, 0];
  let zy: DD = [0, 0];
  let length = 0;
  let escaped = false;
  const orbit = new Float32Array((request.iterations + 1) * 4);

  for (let i = 0; i <= request.iterations; i++) {
    if ((i & 1023) === 0) {
      await new Promise<void>(resolve => setTimeout(resolve, 0));
      if (request.id !== newestRequestId) return;
    }

    const [zxHi, zxLo] = splitF32(zx);
    const [zyHi, zyLo] = splitF32(zy);
    const offset = i * 4;
    orbit[offset] = zxHi;
    orbit[offset + 1] = zxLo;
    orbit[offset + 2] = zyHi;
    orbit[offset + 3] = zyLo;
    length = i + 1;
    if (i === request.iterations) break;

    const zx2 = ddMul(zx, zx);
    const zy2 = ddMul(zy, zy);
    const zxy = ddMul(zx, zy);
    zx = ddAdd(ddSub(zx2, zy2), cx);
    zy = ddAdd(ddScale(zxy, 2), cy);

    const x = ddValue(zx);
    const y = ddValue(zy);
    const radius = x * x + y * y;
    if (!Number.isFinite(radius) || radius > 256) {
      escaped = true;
      break;
    }
  }

  if (request.id !== newestRequestId) return;
  const trimmed = orbit.slice(0, length * 4) as Float32Array<ArrayBuffer>;
  const response: ReferenceResponse = {
    id: request.id,
    bits: Math.min(bits, 106),
    length,
    escaped,
    generationMs: performance.now() - started,
    orbit: trimmed
  };
  worker.postMessage(response, [trimmed.buffer]);
}
