export type DoubleDouble = { hi: number; lo: number };

export function dd(value: number): DoubleDouble {
  return { hi: value, lo: 0 };
}

export function ddValue(value: DoubleDouble): number {
  return value.hi + value.lo;
}

export function ddAdd(value: DoubleDouble, delta: number): DoubleDouble {
  const sum = value.hi + delta;
  const virtualDelta = sum - value.hi;
  const error = (value.hi - (sum - virtualDelta)) + (delta - virtualDelta) + value.lo;
  const hi = sum + error;
  return { hi, lo: error - (hi - sum) };
}

export function ddSub(value: DoubleDouble, delta: number): DoubleDouble {
  return ddAdd(value, -delta);
}

export function ddDifference(a: DoubleDouble, b: number): number {
  return (a.hi - b) + a.lo;
}

export function ddSplit(value: DoubleDouble): [number, number] {
  const hi = Math.fround(value.hi);
  const residual = (value.hi - hi) + value.lo;
  return [hi, Math.fround(residual)];
}
