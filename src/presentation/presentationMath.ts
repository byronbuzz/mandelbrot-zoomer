import { fixedSub, type BigFixed } from '../bigFixed';
import type { BinaryScale } from '../binaryScale';

const NUMBER_MANTISSA_BITS = 53;

function finitePow2(mantissa: number, exponent: number): number {
  if (mantissa === 0) return 0;
  if (!Number.isFinite(mantissa) || !Number.isInteger(exponent)) return Number.NaN;
  if (exponent > 1023 || exponent < -1074) return exponent > 0
    ? Math.sign(mantissa) * Number.POSITIVE_INFINITY
    : 0;
  return mantissa * Math.pow(2, exponent);
}

export function scaleRatio(numerator: BinaryScale, denominator: BinaryScale): number {
  return finitePow2(
    numerator.mantissa / denominator.mantissa,
    numerator.exponent - denominator.exponent
  );
}

export function scaleOverDyadic(scale: BinaryScale, dyadicExponent: number): number {
  return finitePow2(scale.mantissa, scale.exponent - dyadicExponent);
}

export function fixedDifferenceOverDyadic(
  left: BigFixed,
  right: BigFixed,
  dyadicExponent: number
): number {
  const difference = fixedSub(left, right);
  if (difference.raw === 0n) return 0;
  const sign = difference.raw < 0n ? -1 : 1;
  const magnitude = difference.raw < 0n ? -difference.raw : difference.raw;
  const bitLength = magnitude.toString(2).length;
  const shift = Math.max(0, bitLength - NUMBER_MANTISSA_BITS);
  const top = Number(magnitude >> BigInt(shift));
  return sign * finitePow2(top, shift - difference.bits - dyadicExponent);
}

export function fixedDifferenceOverScale(
  left: BigFixed,
  right: BigFixed,
  scale: BinaryScale
): number {
  const normalized = fixedDifferenceOverDyadic(left, right, scale.exponent);
  return normalized / scale.mantissa;
}

export type PackedTransform = Readonly<{
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}>;

export function packTransform(transform: PackedTransform): PackedTransform {
  return {
    scaleX: Math.fround(transform.scaleX),
    scaleY: Math.fround(transform.scaleY),
    offsetX: Math.fround(transform.offsetX),
    offsetY: Math.fround(transform.offsetY)
  };
}

export function transformIsFinite(transform: PackedTransform): boolean {
  return Object.values(transform).every(value => Number.isFinite(value) && Math.abs(value) <= 3.3e38);
}
