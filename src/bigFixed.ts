export type BigFixed = Readonly<{ raw: bigint; bits: number }>;

const NUMBER_MANTISSA_BITS = 53;

function shiftRounded(value: bigint, shift: number): bigint {
  if (shift >= 0) return value << BigInt(shift);
  const amount = BigInt(-shift);
  const half = 1n << (amount - 1n);
  return value >= 0n ? (value + half) >> amount : -((-value + half) >> amount);
}

export function fixed(raw: bigint, bits: number): BigFixed {
  return { raw, bits };
}

export function fixedFromNumber(value: number, bits: number): BigFixed {
  if (!Number.isFinite(value)) throw new Error('Cannot convert a non-finite number to fixed point');
  if (value === 0) return fixed(0n, bits);
  const sign = value < 0 ? -1n : 1n;
  const magnitude = Math.abs(value);
  const exponent = Math.floor(Math.log2(magnitude));
  const mantissa = magnitude / Math.pow(2, exponent);
  const integerMantissa = BigInt(Math.round(mantissa * Math.pow(2, NUMBER_MANTISSA_BITS)));
  return fixed(sign * shiftRounded(integerMantissa, bits + exponent - NUMBER_MANTISSA_BITS), bits);
}

export function fixedFromMantissaExponent(mantissa: number, exponent: number, bits: number): BigFixed {
  if (mantissa === 0) return fixed(0n, bits);
  const base = fixedFromNumber(mantissa, NUMBER_MANTISSA_BITS);
  return fixed(shiftRounded(base.raw, bits + exponent - NUMBER_MANTISSA_BITS), bits);
}

export function fixedRescale(value: BigFixed, bits: number): BigFixed {
  return bits === value.bits ? value : fixed(shiftRounded(value.raw, bits - value.bits), bits);
}

export function fixedAdd(a: BigFixed, b: BigFixed): BigFixed {
  const bits = Math.max(a.bits, b.bits);
  return fixed(fixedRescale(a, bits).raw + fixedRescale(b, bits).raw, bits);
}

export function fixedSub(a: BigFixed, b: BigFixed): BigFixed {
  const bits = Math.max(a.bits, b.bits);
  return fixed(fixedRescale(a, bits).raw - fixedRescale(b, bits).raw, bits);
}

export function fixedAddScaled(a: BigFixed, mantissa: number, exponent: number): BigFixed {
  return fixedAdd(a, fixedFromMantissaExponent(mantissa, exponent, a.bits));
}

export function fixedToNumber(value: BigFixed): number {
  if (value.raw === 0n) return 0;
  const sign = value.raw < 0n ? -1 : 1;
  const magnitude = value.raw < 0n ? -value.raw : value.raw;
  const bitLength = magnitude.toString(2).length;
  const shift = Math.max(0, bitLength - NUMBER_MANTISSA_BITS);
  const top = Number(magnitude >> BigInt(shift));
  return sign * top * Math.pow(2, shift - value.bits);
}

export function fixedDifferenceToNumber(a: BigFixed, b: BigFixed): number {
  return fixedToNumber(fixedSub(a, b));
}

export function fixedSplitF32(value: BigFixed): [number, number] {
  const approximate = fixedToNumber(value);
  const hi = Math.fround(approximate);
  return [hi, Math.fround(approximate - hi)];
}

export function requiredCoordinateBits(log2Magnification: number, guardBits = 112): number {
  const required = Math.ceil(Math.max(0, log2Magnification) + guardBits);
  return Math.min(4096, Math.max(160, Math.ceil(required / 32) * 32));
}

export function serializeFixed(value: BigFixed): { raw: string; bits: number } {
  return { raw: value.raw.toString(), bits: value.bits };
}

export function deserializeFixed(value: { raw: string; bits: number }): BigFixed {
  return fixed(BigInt(value.raw), value.bits);
}
