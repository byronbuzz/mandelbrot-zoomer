import type { ReferenceRequest, ReferenceResponse } from '../../src/v4/types';

declare global {
  interface Window {
    __REFERENCE_TRANSPORT_ORACLE__: Promise<unknown>;
  }
}

const SOURCE_BITS = 256;
const ORACLE_BITS = 384;
const CENTER_X_RAW = -1074788477571874238124292423478415414129187618816n << 96n;
const CENTER_Y_RAW = -225037893802927241757723552826372553133585858560n << 96n;

function shiftRounded(value: bigint, shift: number): bigint {
  if (shift >= 0) return value << BigInt(shift);
  const amount = BigInt(-shift);
  const half = 1n << (amount - 1n);
  return value >= 0n ? (value + half) >> amount : -((-value + half) >> amount);
}

function multiplyFixed(a: bigint, b: bigint): bigint {
  return shiftRounded(a * b, -ORACLE_BITS);
}

function exactOrbit(iterations: number): readonly (readonly [bigint, bigint])[] {
  const cx = CENTER_X_RAW << BigInt(ORACLE_BITS - SOURCE_BITS);
  const cy = CENTER_Y_RAW << BigInt(ORACLE_BITS - SOURCE_BITS);
  let zx = 0n;
  let zy = 0n;
  const result: Array<readonly [bigint, bigint]> = [];
  for (let index = 0; index <= iterations; index++) {
    result.push([zx, zy]);
    const nextX = multiplyFixed(zx, zx) - multiplyFixed(zy, zy) + cx;
    const nextY = 2n * multiplyFixed(zx, zy) + cy;
    zx = nextX;
    zy = nextY;
  }
  return result;
}

function f32ToFixed(value: number): bigint {
  if (value === 0) return 0n;
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value, false);
  const encoded = view.getUint32(0, false);
  const sign = (encoded >>> 31) === 0 ? 1n : -1n;
  const exponent = (encoded >>> 23) & 0xff;
  const fraction = encoded & 0x7fffff;
  if (exponent === 0xff) throw new Error('Non-finite reference limb');
  if (exponent === 0) {
    return sign * shiftRounded(BigInt(fraction), ORACLE_BITS - 149);
  }
  const mantissa = BigInt((1 << 23) | fraction);
  return sign * shiftRounded(mantissa, ORACLE_BITS + exponent - 127 - 23);
}

function transportedCoordinate(
  response: ReferenceResponse,
  pointIndex: number,
  coordinate: 0 | 1
): bigint {
  const limbCount = response.floatsPerPoint / 2;
  const start = pointIndex * response.floatsPerPoint + coordinate * limbCount;
  let result = 0n;
  for (let limb = 0; limb < limbCount; limb++) {
    result += f32ToFixed(response.orbit[start + limb] ?? 0);
  }
  return result;
}

function bitLength(value: bigint): number {
  const magnitude = value < 0n ? -value : value;
  return magnitude === 0n ? 0 : magnitude.toString(2).length;
}

function accuracyBits(actual: bigint, expected: bigint): number {
  const errorBits = bitLength(actual - expected);
  return errorBits === 0 ? ORACLE_BITS : ORACLE_BITS - errorBits;
}

function requestReference(maxTransportBits: 96 | 192): Promise<ReferenceResponse> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../../src/v4/referenceWorker.ts', import.meta.url), { type: 'module' });
    const timeout = window.setTimeout(() => {
      worker.terminate();
      reject(new Error(`Reference worker timed out for ${maxTransportBits}-bit transport`));
    }, 30_000);
    worker.addEventListener('error', event => {
      window.clearTimeout(timeout);
      worker.terminate();
      reject(new Error(event.message));
    });
    worker.addEventListener('message', event => {
      window.clearTimeout(timeout);
      worker.terminate();
      resolve(event.data as ReferenceResponse);
    });
    const centerX = { raw: CENTER_X_RAW.toString(), bits: SOURCE_BITS };
    const centerY = { raw: CENTER_Y_RAW.toString(), bits: SOURCE_BITS };
    const request: ReferenceRequest = {
      id: maxTransportBits,
      cameraGeneration: 1,
      purpose: 'settled',
      centerX,
      centerY,
      iterations: 8,
      probeIterations: 0,
      maxTransportBits,
      candidates: [{ centerX, centerY }]
    };
    worker.postMessage(request);
  });
}

async function runOracle() {
  const [legacy, wide] = await Promise.all([
    requestReference(96),
    requestReference(192)
  ]);
  const oracle = exactOrbit(8);
  const comparisons = [1, 2, 3, 4].map(pointIndex => {
    const [expectedX, expectedY] = oracle[pointIndex];
    return {
      pointIndex,
      legacyXBits: accuracyBits(transportedCoordinate(legacy, pointIndex, 0), expectedX),
      legacyYBits: accuracyBits(transportedCoordinate(legacy, pointIndex, 1), expectedY),
      wideXBits: accuracyBits(transportedCoordinate(wide, pointIndex, 0), expectedX),
      wideYBits: accuracyBits(transportedCoordinate(wide, pointIndex, 1), expectedY)
    };
  });
  return {
    contractVersion: wide.contractVersion,
    legacy: { transportBits: legacy.transportBits, floatsPerPoint: legacy.floatsPerPoint },
    wide: { transportBits: wide.transportBits, floatsPerPoint: wide.floatsPerPoint },
    comparisons
  };
}

window.__REFERENCE_TRANSPORT_ORACLE__ = runOracle();
