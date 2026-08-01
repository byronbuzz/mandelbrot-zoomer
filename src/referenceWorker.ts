/// <reference lib="webworker" />

import type { ReferenceRequest, ReferenceResponse, SerializedFixed } from './referenceProtocol';

const worker = self as unknown as DedicatedWorkerGlobalScope;
const NUMBER_MANTISSA_BITS = 53;
const SPLITTER = 134_217_729;
const TRIPLE_DOUBLE_THRESHOLD_BITS = 192;

type DD = readonly [number, number];
type TD = readonly [number, number, number];

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

function fixedToTD(raw: bigint, bits: number): TD {
  const hi = fixedToNumber(raw, bits);
  const residual1 = raw - numberToFixed(hi, bits);
  const mid = fixedToNumber(residual1, bits);
  const residual2 = residual1 - numberToFixed(mid, bits);
  return [hi, mid, fixedToNumber(residual2, bits)];
}

function ddToSerializedFixed(value: DD, bits: number): SerializedFixed {
  const raw = numberToFixed(value[0], bits) + numberToFixed(value[1], bits);
  return { raw: raw.toString(), bits };
}

function tdToSerializedFixed(value: TD, bits: number): SerializedFixed {
  const raw = numberToFixed(value[0], bits) + numberToFixed(value[1], bits) + numberToFixed(value[2], bits);
  return { raw: raw.toString(), bits };
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
  return quickTwoSum(s, e0 + a[1] + b[1]);
}

function ddSub(a: DD, b: DD): DD {
  return ddAdd(a, [-b[0], -b[1]]);
}

function ddMul(a: DD, b: DD): DD {
  const [p, e0] = twoProduct(a[0], b[0]);
  return quickTwoSum(p, e0 + a[0] * b[1] + a[1] * b[0] + a[1] * b[1]);
}

function ddScale(a: DD, scale: number): DD {
  return ddMul(a, [scale, 0]);
}

function growExpansion(expansion: number[], value: number): number[] {
  let q = value;
  const result: number[] = [];
  for (const component of expansion) {
    const [sum, error] = twoSum(q, component);
    if (error !== 0) result.push(error);
    q = sum;
  }
  if (q !== 0 || result.length === 0) result.push(q);
  return result;
}

function tdFromTerms(terms: readonly number[]): TD {
  const ordered = terms.filter(value => value !== 0 && Number.isFinite(value)).sort((a, b) => Math.abs(a) - Math.abs(b));
  if (ordered.length === 0) return [0, 0, 0];
  let expansion: number[] = [];
  for (const term of ordered) expansion = growExpansion(expansion, term);
  const largest = expansion.slice(-3).reverse();
  return [largest[0] ?? 0, largest[1] ?? 0, largest[2] ?? 0];
}

function tdAdd(a: TD, b: TD): TD {
  return tdFromTerms([a[0], a[1], a[2], b[0], b[1], b[2]]);
}

function tdSub(a: TD, b: TD): TD {
  return tdFromTerms([a[0], a[1], a[2], -b[0], -b[1], -b[2]]);
}

function tdMul(a: TD, b: TD): TD {
  const terms: number[] = [];
  for (const left of a) {
    for (const right of b) {
      const [product, error] = twoProduct(left, right);
      terms.push(error, product);
    }
  }
  return tdFromTerms(terms);
}

function tdScale(a: TD, scale: number): TD {
  return tdMul(a, [scale, 0, 0]);
}

function splitF32DD(value: DD): [number, number] {
  const hi = Math.fround(value[0]);
  return [hi, Math.fround((value[0] - hi) + value[1])];
}

function splitF32TD(value: TD): [number, number] {
  const hi = Math.fround(value[0]);
  return [hi, Math.fround((value[0] - hi) + value[1] + value[2])];
}

function buildDoubleDoubleReference(request: ReferenceRequest, bits: number): ReferenceResponse {
  const started = performance.now();
  const cx = fixedToDD(BigInt(request.centerX.raw), request.centerX.bits);
  const cy = fixedToDD(BigInt(request.centerY.raw), request.centerY.bits);
  let zx: DD = [0, 0];
  let zy: DD = [0, 0];
  let length = 0;
  let escaped = false;
  const orbit = new Float32Array((request.iterations + 1) * 4);

  for (let index = 0; index <= request.iterations; index++) {
    const [zxHi, zxLo] = splitF32DD(zx);
    const [zyHi, zyLo] = splitF32DD(zy);
    const offset = index * 4;
    orbit[offset] = zxHi;
    orbit[offset + 1] = zxLo;
    orbit[offset + 2] = zyHi;
    orbit[offset + 3] = zyLo;
    length = index + 1;
    if (index === request.iterations) break;

    const zx2 = ddMul(zx, zx);
    const zy2 = ddMul(zy, zy);
    const zxy = ddMul(zx, zy);
    zx = ddAdd(ddSub(zx2, zy2), cx);
    zy = ddAdd(ddScale(zxy, 2), cy);

    const x = zx[0] + zx[1];
    const y = zy[0] + zy[1];
    const radius = x * x + y * y;
    if (!Number.isFinite(radius) || radius > 256) {
      escaped = true;
      break;
    }
  }

  const trimmed = orbit.slice(0, length * 4) as Float32Array<ArrayBuffer>;
  return {
    id: request.id,
    bits: Math.min(bits, 106),
    length,
    escaped,
    generationMs: performance.now() - started,
    referenceCenterX: ddToSerializedFixed(cx, request.centerX.bits),
    referenceCenterY: ddToSerializedFixed(cy, request.centerY.bits),
    orbit: trimmed
  };
}

function buildTripleDoubleReference(request: ReferenceRequest, bits: number): ReferenceResponse {
  const started = performance.now();
  const cx = fixedToTD(BigInt(request.centerX.raw), request.centerX.bits);
  const cy = fixedToTD(BigInt(request.centerY.raw), request.centerY.bits);
  let zx: TD = [0, 0, 0];
  let zy: TD = [0, 0, 0];
  let length = 0;
  let escaped = false;
  const orbit = new Float32Array((request.iterations + 1) * 4);

  for (let index = 0; index <= request.iterations; index++) {
    const [zxHi, zxLo] = splitF32TD(zx);
    const [zyHi, zyLo] = splitF32TD(zy);
    const offset = index * 4;
    orbit[offset] = zxHi;
    orbit[offset + 1] = zxLo;
    orbit[offset + 2] = zyHi;
    orbit[offset + 3] = zyLo;
    length = index + 1;
    if (index === request.iterations) break;

    const zx2 = tdMul(zx, zx);
    const zy2 = tdMul(zy, zy);
    const zxy = tdMul(zx, zy);
    zx = tdAdd(tdSub(zx2, zy2), cx);
    zy = tdAdd(tdScale(zxy, 2), cy);

    const x = zx[0] + zx[1] + zx[2];
    const y = zy[0] + zy[1] + zy[2];
    const radius = x * x + y * y;
    if (!Number.isFinite(radius) || radius > 256) {
      escaped = true;
      break;
    }
  }

  const trimmed = orbit.slice(0, length * 4) as Float32Array<ArrayBuffer>;
  return {
    id: request.id,
    bits: Math.min(bits, 159),
    length,
    escaped,
    generationMs: performance.now() - started,
    referenceCenterX: tdToSerializedFixed(cx, request.centerX.bits),
    referenceCenterY: tdToSerializedFixed(cy, request.centerY.bits),
    orbit: trimmed
  };
}

function buildReference(request: ReferenceRequest): void {
  const bits = Math.max(request.centerX.bits, request.centerY.bits);
  const response = bits > TRIPLE_DOUBLE_THRESHOLD_BITS
    ? buildTripleDoubleReference(request, bits)
    : buildDoubleDoubleReference(request, bits);
  worker.postMessage(response, [response.orbit.buffer]);
}

worker.addEventListener('message', event => {
  buildReference(event.data as ReferenceRequest);
});
