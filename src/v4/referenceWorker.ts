/// <reference lib="webworker" />

import type { ReferenceCandidate, ReferenceRequest, ReferenceResponse, SerializedFixed } from './types';

const worker = self as unknown as DedicatedWorkerGlobalScope;
const NUMBER_MANTISSA_BITS = 53;
const SPLITTER = 134_217_729;
const TRIPLE_DOUBLE_THRESHOLD_BITS = 192;
const PROBE_TRIPLE_DOUBLE_THRESHOLD_BITS = 224;
const QUAD_DOUBLE_THRESHOLD_BITS = 256;
const DENSE_COARSE_PROBE_ITERATIONS = 1024;
const DENSE_MIDDLE_PROBE_ITERATIONS = 3072;
const REFERENCE_CONTRACT_VERSION = 2;
const LEGACY_REFERENCE_TRANSPORT_BITS = 96;
const TRIPLE_DOUBLE_TRANSPORT_BITS = 144;
const WIDE_REFERENCE_TRANSPORT_BITS = 192;

type DD = readonly [number, number];
type TD = readonly [number, number, number];
type QD = readonly [number, number, number, number];
type FloatExpansion4 = [number, number, number, number];
type FloatExpansion8 = [number, number, number, number, number, number, number, number];

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

function fixedToQD(raw: bigint, bits: number): QD {
  const hi = fixedToNumber(raw, bits);
  const residual1 = raw - numberToFixed(hi, bits);
  const midHi = fixedToNumber(residual1, bits);
  const residual2 = residual1 - numberToFixed(midHi, bits);
  const midLo = fixedToNumber(residual2, bits);
  const residual3 = residual2 - numberToFixed(midLo, bits);
  return [hi, midHi, midLo, fixedToNumber(residual3, bits)];
}

function ddToSerializedFixed(value: DD, bits: number): SerializedFixed {
  const raw = numberToFixed(value[0], bits) + numberToFixed(value[1], bits);
  return { raw: raw.toString(), bits };
}

function tdToSerializedFixed(value: TD, bits: number): SerializedFixed {
  const raw = numberToFixed(value[0], bits) + numberToFixed(value[1], bits) + numberToFixed(value[2], bits);
  return { raw: raw.toString(), bits };
}

function qdToSerializedFixed(value: QD, bits: number): SerializedFixed {
  const raw = value.reduce((sum, component) => sum + numberToFixed(component, bits), 0n);
  return { raw: raw.toString(), bits };
}

function quickTwoSum(a: number, b: number): DD {
  const sum = a + b;
  return [sum, b - (sum - a)];
}

function twoSum(a: number, b: number): DD {
  const sum = a + b;
  const virtualB = sum - a;
  return [sum, (a - (sum - virtualB)) + (b - virtualB)];
}

function split(value: number): DD {
  const scaled = SPLITTER * value;
  const hi = scaled - (scaled - value);
  return [hi, value - hi];
}

function twoProduct(a: number, b: number): DD {
  const product = a * b;
  const [aHi, aLo] = split(a);
  const [bHi, bLo] = split(b);
  const error = ((aHi * bHi - product) + aHi * bLo + aLo * bHi) + aLo * bLo;
  return [product, error];
}

function ddAdd(a: DD, b: DD): DD {
  const [sum, error] = twoSum(a[0], b[0]);
  return quickTwoSum(sum, error + a[1] + b[1]);
}

function ddSub(a: DD, b: DD): DD { return ddAdd(a, [-b[0], -b[1]]); }

function ddMul(a: DD, b: DD): DD {
  const [product, error] = twoProduct(a[0], b[0]);
  return quickTwoSum(product, error + a[0] * b[1] + a[1] * b[0] + a[1] * b[1]);
}

function ddScale(value: DD, scale: number): DD { return ddMul(value, [scale, 0]); }

function growExpansion(expansion: number[], value: number): number[] {
  let accumulator = value;
  const result: number[] = [];
  for (const component of expansion) {
    const [sum, error] = twoSum(accumulator, component);
    if (error !== 0) result.push(error);
    accumulator = sum;
  }
  if (accumulator !== 0 || result.length === 0) result.push(accumulator);
  return result;
}

function tdFromTerms(terms: readonly number[]): TD {
  const ordered = terms
    .filter(value => value !== 0 && Number.isFinite(value))
    .sort((a, b) => Math.abs(a) - Math.abs(b));
  if (ordered.length === 0) return [0, 0, 0];
  let expansion: number[] = [];
  for (const term of ordered) expansion = growExpansion(expansion, term);
  const largest = expansion.slice(-3).reverse();
  return [largest[0] ?? 0, largest[1] ?? 0, largest[2] ?? 0];
}

function tdAdd(a: TD, b: TD): TD { return tdFromTerms([...a, ...b]); }
function tdSub(a: TD, b: TD): TD { return tdFromTerms([...a, -b[0], -b[1], -b[2]]); }

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

function tdScale(value: TD, scale: number): TD { return tdMul(value, [scale, 0, 0]); }

function qdFromTerms(terms: readonly number[]): QD {
  const ordered = terms
    .filter(value => value !== 0 && Number.isFinite(value))
    .sort((a, b) => Math.abs(a) - Math.abs(b));
  if (ordered.length === 0) return [0, 0, 0, 0];
  let expansion: number[] = [];
  for (const term of ordered) expansion = growExpansion(expansion, term);
  const largest = expansion.slice(-4).reverse();
  return [largest[0] ?? 0, largest[1] ?? 0, largest[2] ?? 0, largest[3] ?? 0];
}

function qdAdd(a: QD, b: QD): QD { return qdFromTerms([...a, ...b]); }
function qdSub(a: QD, b: QD): QD { return qdFromTerms([...a, -b[0], -b[1], -b[2], -b[3]]); }

function qdMul(a: QD, b: QD): QD {
  const terms: number[] = [];
  for (const left of a) {
    for (const right of b) {
      const [product, error] = twoProduct(left, right);
      terms.push(error, product);
    }
  }
  return qdFromTerms(terms);
}

function qdScale(value: QD, scale: number): QD { return qdMul(value, [scale, 0, 0, 0]); }

function splitF32ExpansionDD(value: DD): FloatExpansion4 {
  let remaining = value;
  const result: FloatExpansion4 = [0, 0, 0, 0];
  for (let index = 0; index < result.length; index++) {
    const limb = Math.fround(remaining[0] + remaining[1]);
    result[index] = limb;
    remaining = ddSub(remaining, [limb, 0]);
  }
  return result;
}

function splitF32ExpansionTD(value: TD): FloatExpansion4 {
  let remaining = value;
  const result: FloatExpansion4 = [0, 0, 0, 0];
  for (let index = 0; index < result.length; index++) {
    const limb = Math.fround(remaining[0] + remaining[1] + remaining[2]);
    result[index] = limb;
    remaining = tdSub(remaining, [limb, 0, 0]);
  }
  return result;
}

function splitF32ExpansionTD8(value: TD): FloatExpansion8 {
  let remaining = value;
  const result: FloatExpansion8 = [0, 0, 0, 0, 0, 0, 0, 0];
  for (let index = 0; index < result.length; index++) {
    const limb = Math.fround(remaining[0] + remaining[1] + remaining[2]);
    result[index] = limb;
    remaining = tdSub(remaining, [limb, 0, 0]);
  }
  return result;
}

function splitF32ExpansionQD8(value: QD): FloatExpansion8 {
  let remaining = value;
  const result: FloatExpansion8 = [0, 0, 0, 0, 0, 0, 0, 0];
  for (let index = 0; index < result.length; index++) {
    const limb = Math.fround(remaining[0] + remaining[1] + remaining[2] + remaining[3]);
    result[index] = limb;
    remaining = qdSub(remaining, [limb, 0, 0, 0]);
  }
  return result;
}

function transportLayout(maxTransportBits: 96 | 192, arithmeticBits: 96 | 144 | 192): {
  transportBits: 96 | 144 | 192;
  floatsPerPoint: 8 | 16;
  limbCount: 4 | 6 | 8;
} {
  const transportBits = Math.min(maxTransportBits, arithmeticBits) as 96 | 144 | 192;
  return {
    transportBits,
    floatsPerPoint: transportBits > 96 ? 16 : 8,
    limbCount: (transportBits / 24) as 4 | 6 | 8
  };
}

function storeOrbitPoint(
  orbit: Float32Array,
  pointIndex: number,
  floatsPerPoint: 8 | 16,
  limbCount: 4 | 6 | 8,
  x: readonly number[],
  y: readonly number[]
): void {
  const offset = pointIndex * floatsPerPoint;
  orbit.set(x.slice(0, limbCount), offset);
  orbit.set(y.slice(0, limbCount), offset + floatsPerPoint / 2);
}

function probeDoubleDouble(candidate: ReferenceCandidate, iterations: number): number {
  const cx = fixedToDD(BigInt(candidate.centerX.raw), candidate.centerX.bits);
  const cy = fixedToDD(BigInt(candidate.centerY.raw), candidate.centerY.bits);
  let zx: DD = [0, 0];
  let zy: DD = [0, 0];
  for (let index = 0; index < iterations; index++) {
    const zx2 = ddMul(zx, zx);
    const zy2 = ddMul(zy, zy);
    const zxy = ddMul(zx, zy);
    zx = ddAdd(ddSub(zx2, zy2), cx);
    zy = ddAdd(ddScale(zxy, 2), cy);
    const x = zx[0] + zx[1];
    const y = zy[0] + zy[1];
    if (!Number.isFinite(x) || !Number.isFinite(y) || x * x + y * y > 256) return index + 1;
  }
  return iterations + 1;
}

function probeTripleDouble(candidate: ReferenceCandidate, iterations: number): number {
  const cx = fixedToTD(BigInt(candidate.centerX.raw), candidate.centerX.bits);
  const cy = fixedToTD(BigInt(candidate.centerY.raw), candidate.centerY.bits);
  let zx: TD = [0, 0, 0];
  let zy: TD = [0, 0, 0];
  for (let index = 0; index < iterations; index++) {
    const zx2 = tdMul(zx, zx);
    const zy2 = tdMul(zy, zy);
    const zxy = tdMul(zx, zy);
    zx = tdAdd(tdSub(zx2, zy2), cx);
    zy = tdAdd(tdScale(zxy, 2), cy);
    const x = zx[0] + zx[1] + zx[2];
    const y = zy[0] + zy[1] + zy[2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || x * x + y * y > 256) return index + 1;
  }
  return iterations + 1;
}

function probeQuadDouble(candidate: ReferenceCandidate, iterations: number): number {
  const cx = fixedToQD(BigInt(candidate.centerX.raw), candidate.centerX.bits);
  const cy = fixedToQD(BigInt(candidate.centerY.raw), candidate.centerY.bits);
  let zx: QD = [0, 0, 0, 0];
  let zy: QD = [0, 0, 0, 0];
  for (let index = 0; index < iterations; index++) {
    const zx2 = qdMul(zx, zx);
    const zy2 = qdMul(zy, zy);
    const zxy = qdMul(zx, zy);
    zx = qdAdd(qdSub(zx2, zy2), cx);
    zy = qdAdd(qdScale(zxy, 2), cy);
    const x = zx[0] + zx[1] + zx[2] + zx[3];
    const y = zy[0] + zy[1] + zy[2] + zy[3];
    if (!Number.isFinite(x) || !Number.isFinite(y) || x * x + y * y > 256) return index + 1;
  }
  return iterations + 1;
}

function multiplyFixed(a: bigint, b: bigint, bits: number): bigint {
  return shiftRounded(a * b, -bits);
}

function probeExactFixed(candidate: ReferenceCandidate, iterations: number, bits: number): number {
  const cx = shiftRounded(BigInt(candidate.centerX.raw), bits - candidate.centerX.bits);
  const cy = shiftRounded(BigInt(candidate.centerY.raw), bits - candidate.centerY.bits);
  const escapeSquared = 256n << BigInt(bits * 2);
  let zx = 0n;
  let zy = 0n;
  for (let index = 0; index < iterations; index++) {
    const nextX = multiplyFixed(zx, zx, bits) - multiplyFixed(zy, zy, bits) + cx;
    const nextY = 2n * multiplyFixed(zx, zy, bits) + cy;
    zx = nextX;
    zy = nextY;
    if (zx * zx + zy * zy > escapeSquared) return index + 1;
  }
  return iterations + 1;
}

function probeCandidate(candidate: ReferenceCandidate, iterations: number, bits: number): number {
  return bits >= QUAD_DOUBLE_THRESHOLD_BITS
    ? probeExactFixed(candidate, iterations, bits)
    : bits >= PROBE_TRIPLE_DOUBLE_THRESHOLD_BITS
    ? probeTripleDouble(candidate, iterations)
    : probeDoubleDouble(candidate, iterations);
}

function selectCandidate(request: ReferenceRequest, bits: number): ReferenceCandidate {
  const candidates = request.candidates.length > 0
    ? [...request.candidates]
    : [{ centerX: request.centerX, centerY: request.centerY }];
  const requestedProbe = Math.max(0, Math.min(request.iterations, request.probeIterations));
  if (requestedProbe === 0 || candidates.length === 1) return candidates[0];

  const stages = [
    Math.min(requestedProbe, DENSE_COARSE_PROBE_ITERATIONS),
    Math.min(requestedProbe, DENSE_MIDDLE_PROBE_ITERATIONS),
    requestedProbe
  ].filter((stage, index, all) => stage > 0 && all.indexOf(stage) === index);

  let contenders = candidates;
  for (const stage of stages) {
    let bestCandidate = contenders[0];
    let bestScore = -1;
    const survivors: ReferenceCandidate[] = [];
    for (const candidate of contenders) {
      const score = probeCandidate(candidate, stage, bits);
      if (score > bestScore) {
        bestCandidate = candidate;
        bestScore = score;
      }
      if (score > stage) survivors.push(candidate);
    }
    if (survivors.length === 0) return bestCandidate;
    if (stage === requestedProbe) return survivors[0];
    contenders = survivors;
  }
  return contenders[0];
}

function buildDoubleDoubleReference(request: ReferenceRequest, candidate: ReferenceCandidate, bits: number): ReferenceResponse {
  const started = performance.now();
  const layout = transportLayout(request.maxTransportBits, LEGACY_REFERENCE_TRANSPORT_BITS);
  const cx = fixedToDD(BigInt(candidate.centerX.raw), candidate.centerX.bits);
  const cy = fixedToDD(BigInt(candidate.centerY.raw), candidate.centerY.bits);
  let zx: DD = [0, 0];
  let zy: DD = [0, 0];
  let length = 0;
  let escaped = false;
  const orbit = new Float32Array((request.iterations + 1) * layout.floatsPerPoint);
  for (let index = 0; index <= request.iterations; index++) {
    const x = splitF32ExpansionDD(zx);
    const y = splitF32ExpansionDD(zy);
    storeOrbitPoint(orbit, index, layout.floatsPerPoint, layout.limbCount, x, y);
    length = index + 1;
    const approximateX = zx[0] + zx[1];
    const approximateY = zy[0] + zy[1];
    if (!Number.isFinite(approximateX) || !Number.isFinite(approximateY)
      || approximateX * approximateX + approximateY * approximateY > 256) {
      escaped = true;
      break;
    }
    if (index === request.iterations) break;
    const zx2 = ddMul(zx, zx);
    const zy2 = ddMul(zy, zy);
    const zxy = ddMul(zx, zy);
    zx = ddAdd(ddSub(zx2, zy2), cx);
    zy = ddAdd(ddScale(zxy, 2), cy);
  }
  const trimmed = orbit.slice(0, length * layout.floatsPerPoint) as Float32Array<ArrayBuffer>;
  return {
    id: request.id,
    cameraGeneration: request.cameraGeneration,
    purpose: request.purpose,
    bits: Math.min(bits, layout.transportBits),
    workingBits: bits,
    transportBits: Math.min(bits, layout.transportBits),
    floatsPerPoint: layout.floatsPerPoint,
    contractVersion: REFERENCE_CONTRACT_VERSION,
    length,
    escaped,
    generationMs: performance.now() - started,
    referenceCenterX: ddToSerializedFixed(cx, candidate.centerX.bits),
    referenceCenterY: ddToSerializedFixed(cy, candidate.centerY.bits),
    orbit: trimmed
  };
}

function buildTripleDoubleReference(request: ReferenceRequest, candidate: ReferenceCandidate, bits: number): ReferenceResponse {
  const started = performance.now();
  const layout = transportLayout(request.maxTransportBits, TRIPLE_DOUBLE_TRANSPORT_BITS);
  const cx = fixedToTD(BigInt(candidate.centerX.raw), candidate.centerX.bits);
  const cy = fixedToTD(BigInt(candidate.centerY.raw), candidate.centerY.bits);
  let zx: TD = [0, 0, 0];
  let zy: TD = [0, 0, 0];
  let length = 0;
  let escaped = false;
  const orbit = new Float32Array((request.iterations + 1) * layout.floatsPerPoint);
  for (let index = 0; index <= request.iterations; index++) {
    const x = layout.limbCount > 4 ? splitF32ExpansionTD8(zx) : splitF32ExpansionTD(zx);
    const y = layout.limbCount > 4 ? splitF32ExpansionTD8(zy) : splitF32ExpansionTD(zy);
    storeOrbitPoint(orbit, index, layout.floatsPerPoint, layout.limbCount, x, y);
    length = index + 1;
    const approximateX = zx[0] + zx[1] + zx[2];
    const approximateY = zy[0] + zy[1] + zy[2];
    if (!Number.isFinite(approximateX) || !Number.isFinite(approximateY)
      || approximateX * approximateX + approximateY * approximateY > 256) {
      escaped = true;
      break;
    }
    if (index === request.iterations) break;
    const zx2 = tdMul(zx, zx);
    const zy2 = tdMul(zy, zy);
    const zxy = tdMul(zx, zy);
    zx = tdAdd(tdSub(zx2, zy2), cx);
    zy = tdAdd(tdScale(zxy, 2), cy);
  }
  const trimmed = orbit.slice(0, length * layout.floatsPerPoint) as Float32Array<ArrayBuffer>;
  return {
    id: request.id,
    cameraGeneration: request.cameraGeneration,
    purpose: request.purpose,
    bits: Math.min(bits, layout.transportBits),
    workingBits: bits,
    transportBits: Math.min(bits, layout.transportBits),
    floatsPerPoint: layout.floatsPerPoint,
    contractVersion: REFERENCE_CONTRACT_VERSION,
    length,
    escaped,
    generationMs: performance.now() - started,
    referenceCenterX: tdToSerializedFixed(cx, candidate.centerX.bits),
    referenceCenterY: tdToSerializedFixed(cy, candidate.centerY.bits),
    orbit: trimmed
  };
}

function splitFixedF32Expansion(raw: bigint, bits: number): FloatExpansion8 {
  let remaining = raw;
  const result: FloatExpansion8 = [0, 0, 0, 0, 0, 0, 0, 0];
  for (let index = 0; index < result.length; index++) {
    const limb = Math.fround(fixedToNumber(remaining, bits));
    result[index] = limb;
    remaining -= numberToFixed(limb, bits);
  }
  return result;
}

function buildExactFixedReference(request: ReferenceRequest, candidate: ReferenceCandidate, bits: number): ReferenceResponse {
  const started = performance.now();
  const layout = transportLayout(request.maxTransportBits, WIDE_REFERENCE_TRANSPORT_BITS);
  const cx = shiftRounded(BigInt(candidate.centerX.raw), bits - candidate.centerX.bits);
  const cy = shiftRounded(BigInt(candidate.centerY.raw), bits - candidate.centerY.bits);
  const escapeSquared = 256n << BigInt(bits * 2);
  let zx = 0n;
  let zy = 0n;
  let length = 0;
  let escaped = false;
  const orbit = new Float32Array((request.iterations + 1) * layout.floatsPerPoint);
  for (let index = 0; index <= request.iterations; index++) {
    const x = splitFixedF32Expansion(zx, bits);
    const y = splitFixedF32Expansion(zy, bits);
    storeOrbitPoint(orbit, index, layout.floatsPerPoint, layout.limbCount, x, y);
    length = index + 1;
    if (zx * zx + zy * zy > escapeSquared) {
      escaped = true;
      break;
    }
    if (index === request.iterations) break;
    const nextX = multiplyFixed(zx, zx, bits) - multiplyFixed(zy, zy, bits) + cx;
    const nextY = 2n * multiplyFixed(zx, zy, bits) + cy;
    zx = nextX;
    zy = nextY;
  }
  const trimmed = orbit.slice(0, length * layout.floatsPerPoint) as Float32Array<ArrayBuffer>;
  return {
    id: request.id,
    cameraGeneration: request.cameraGeneration,
    purpose: request.purpose,
    bits: Math.min(bits, layout.transportBits),
    workingBits: bits,
    transportBits: Math.min(bits, layout.transportBits),
    floatsPerPoint: layout.floatsPerPoint,
    contractVersion: REFERENCE_CONTRACT_VERSION,
    length,
    escaped,
    generationMs: performance.now() - started,
    referenceCenterX: candidate.centerX,
    referenceCenterY: candidate.centerY,
    orbit: trimmed
  };
}

function buildReference(request: ReferenceRequest): void {
  const bits = Math.max(request.centerX.bits, request.centerY.bits);
  const candidate = selectCandidate(request, bits);
  const response = bits >= QUAD_DOUBLE_THRESHOLD_BITS
    ? buildExactFixedReference(request, candidate, bits)
    : bits >= TRIPLE_DOUBLE_THRESHOLD_BITS
    ? buildTripleDoubleReference(request, candidate, bits)
    : buildDoubleDoubleReference(request, candidate, bits);
  worker.postMessage(response, [response.orbit.buffer]);
}

worker.addEventListener('message', event => buildReference(event.data as ReferenceRequest));
