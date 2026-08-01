export type BinaryScale = Readonly<{ mantissa: number; exponent: number }>;

export function normalizeScale(mantissa: number, exponent: number): BinaryScale {
  if (!(mantissa > 0) || !Number.isFinite(mantissa)) throw new Error('Viewport scale must be finite and positive');
  let m = mantissa;
  let e = exponent;
  while (m >= 1) { m *= .5; e++; }
  while (m < .5) { m *= 2; e--; }
  return { mantissa: m, exponent: e };
}

export function scaleFromNumber(value: number): BinaryScale {
  if (!(value > 0) || !Number.isFinite(value)) throw new Error('Viewport scale must be finite and positive');
  const exponent = Math.floor(Math.log2(value)) + 1;
  return normalizeScale(value / Math.pow(2, exponent), exponent);
}

export function scaleMultiply(scale: BinaryScale, factor: number): BinaryScale {
  return normalizeScale(scale.mantissa * factor, scale.exponent);
}

export function scaleLog2(scale: BinaryScale): number {
  return Math.log2(scale.mantissa) + scale.exponent;
}

export function scaleLog10(scale: BinaryScale): number {
  return Math.log10(scale.mantissa) + scale.exponent * Math.LOG10E * Math.LN2;
}

export function scaleToNumber(scale: BinaryScale): number {
  return scale.mantissa * Math.pow(2, scale.exponent);
}

export function scaleDeltaParts(scale: BinaryScale, multiplier: number): { mantissa: number; exponent: number } {
  if (multiplier === 0) return { mantissa: 0, exponent: 0 };
  const sign = multiplier < 0 ? -1 : 1;
  const normalized = normalizeScale(Math.abs(multiplier) * scale.mantissa, scale.exponent);
  return { mantissa: sign * normalized.mantissa, exponent: normalized.exponent };
}
