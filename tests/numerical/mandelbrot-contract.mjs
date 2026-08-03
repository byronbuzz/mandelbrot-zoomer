export const DIRECT_CONTRACT_VERSION = 1;
export const BAILOUT_RADIUS_SQUARED = 256;
export const SMOOTH_ABSOLUTE_FLOOR = 0.001;
export const SMOOTH_ULP_MULTIPLIER = 8;

export const DirectStatus = Object.freeze({
  ACTIVE: 0,
  ESCAPED: 1,
  ANALYTIC_INTERIOR: 2,
  NON_FINITE: 3,
  CAP: 4
});

export function analyticInterior(cx, cy) {
  const shiftedX = cx - 0.25;
  const ySquared = cy * cy;
  const q = shiftedX * shiftedX + ySquared;
  const cardioid = q * (q + shiftedX) <= 0.25 * ySquared;
  const bulbX = cx + 1;
  const bulb = bulbX * bulbX + ySquared <= 0.0625;
  return cardioid || bulb;
}

export function smoothEscape(iteration, radiusSquared) {
  return iteration + 1 - Math.log2(Math.log2(Math.sqrt(Math.max(radiusSquared, 4.000001))));
}

export function directF64(cx, cy, targetIterations, chunks = null) {
  if (!Number.isInteger(targetIterations) || targetIterations < 0) {
    throw new Error(`Invalid target iteration count ${targetIterations}`);
  }
  const schedule = chunks ?? (targetIterations === 0 ? [] : [targetIterations]);
  const zeroDispatch = Array.isArray(schedule) && targetIterations === 0
    && schedule.length === 1 && schedule[0] === 0;
  if (!Array.isArray(schedule) || (targetIterations > 0 && schedule.length === 0)
    || (!zeroDispatch && schedule.some(value => !Number.isInteger(value) || value <= 0))) {
    throw new Error('Chunk schedule must contain positive integers.');
  }
  if (analyticInterior(cx, cy)) {
    return {
      status: DirectStatus.ANALYTIC_INTERIOR,
      iteration: 0,
      x: 0,
      y: 0,
      radiusSquared: 0,
      previousRadiusSquared: 0,
      smooth: null,
      explicitIterations: 0
    };
  }

  let x = 0;
  let y = 0;
  let iteration = 0;
  let radiusSquared = 0;
  let previousRadiusSquared = 0;
  let chunkIndex = 0;
  while (iteration < targetIterations) {
    const chunkIterations = schedule[chunkIndex % schedule.length];
    const iterationEnd = Math.min(targetIterations, iteration + chunkIterations);
    for (;;) {
      radiusSquared = x * x + y * y;
      if (iteration >= iterationEnd || radiusSquared > BAILOUT_RADIUS_SQUARED) break;
      previousRadiusSquared = radiusSquared;
      const nextX = x * x - y * y + cx;
      const nextY = 2 * x * y + cy;
      x = nextX;
      y = nextY;
      iteration += 1;
    }
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(radiusSquared)) {
      return {
        status: DirectStatus.NON_FINITE,
        iteration,
        x,
        y,
        radiusSquared,
        previousRadiusSquared,
        smooth: null,
        explicitIterations: iteration
      };
    }
    if (radiusSquared > BAILOUT_RADIUS_SQUARED) {
      return {
        status: DirectStatus.ESCAPED,
        iteration,
        x,
        y,
        radiusSquared,
        previousRadiusSquared,
        smooth: smoothEscape(iteration, radiusSquared),
        explicitIterations: iteration
      };
    }
    chunkIndex += 1;
  }

  radiusSquared = x * x + y * y;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(radiusSquared)) {
    return {
      status: DirectStatus.NON_FINITE,
      iteration,
      x,
      y,
      radiusSquared,
      previousRadiusSquared,
      smooth: null,
      explicitIterations: iteration
    };
  }
  if (radiusSquared > BAILOUT_RADIUS_SQUARED) {
    return {
      status: DirectStatus.ESCAPED,
      iteration,
      x,
      y,
      radiusSquared,
      previousRadiusSquared,
      smooth: smoothEscape(iteration, radiusSquared),
      explicitIterations: iteration
    };
  }
  return {
    status: DirectStatus.CAP,
    iteration,
    x,
    y,
    radiusSquared,
    previousRadiusSquared,
    smooth: null,
    explicitIterations: iteration
  };
}

export function coordinateAt(center, pixelIndex, tileSize, sampleExponent) {
  return center + (pixelIndex + 0.5 - tileSize * 0.5) * 2 ** sampleExponent;
}

export function ulpF32(value) {
  if (!Number.isFinite(value)) return Number.POSITIVE_INFINITY;
  const storage = new ArrayBuffer(4);
  const float = new Float32Array(storage);
  const bits = new Uint32Array(storage);
  float[0] = Math.fround(value);
  if (float[0] === 0) return 2 ** -149;
  const original = float[0];
  bits[0] += original > 0 ? 1 : -1;
  return Math.abs(float[0] - original);
}

export function smoothTolerance(expected) {
  return Math.max(SMOOTH_ABSOLUTE_FLOOR, SMOOTH_ULP_MULTIPLIER * ulpF32(expected));
}

export function normalizeGpuResult(meta, result, quality, targetIterations) {
  const iteration = meta[0];
  const metaStatus = meta[1];
  const resultStatus = Math.round(result[1]);
  const resultEvidence = Math.round(result[2]);
  const accepted = (quality?.[0] ?? 0) > 0;
  if (metaStatus === DirectStatus.NON_FINITE) {
    return { status: DirectStatus.NON_FINITE, iteration, smooth: null };
  }
  if (accepted && metaStatus === DirectStatus.ESCAPED
    && resultStatus === DirectStatus.ESCAPED && resultEvidence === iteration) {
    return { status: DirectStatus.ESCAPED, iteration, smooth: result[0] };
  }
  if (accepted && metaStatus === DirectStatus.ANALYTIC_INTERIOR
    && resultStatus === DirectStatus.ANALYTIC_INTERIOR && iteration === 0) {
    return { status: DirectStatus.ANALYTIC_INTERIOR, iteration, smooth: null };
  }
  if (accepted && metaStatus === DirectStatus.ACTIVE && iteration === targetIterations
    && resultStatus === DirectStatus.CAP && resultEvidence === iteration) {
    return { status: DirectStatus.CAP, iteration, smooth: null };
  }
  return { status: DirectStatus.ACTIVE, iteration, smooth: null };
}

export function chunkSchedules(targetIterations, escapeIteration = null) {
  const repeated = size => {
    const schedule = [];
    let remaining = targetIterations;
    while (remaining > 0) {
      const next = Math.min(size, remaining);
      schedule.push(next);
      remaining -= next;
    }
    return schedule;
  };
  if (targetIterations === 0) return [{ id: 'zero-target', chunks: [0] }];
  const schedules = [
    { id: 'monolithic', chunks: [Math.max(1, targetIterations)] },
    { id: 'ones', chunks: repeated(1) },
    { id: 'prime-7', chunks: repeated(7) },
    { id: 'workgroup-64', chunks: repeated(64) },
    { id: 'fibonacci', chunks: [] }
  ];
  const fibonacci = [1, 2, 3, 5, 8, 13, 21, 34];
  let remaining = targetIterations;
  let index = 0;
  while (remaining > 0) {
    const next = Math.min(fibonacci[index % fibonacci.length], remaining);
    schedules[4].chunks.push(next);
    remaining -= next;
    index += 1;
  }
  if (escapeIteration !== null && escapeIteration > 1 && escapeIteration < targetIterations) {
    schedules.push({
      id: 'escape-boundary',
      chunks: [escapeIteration - 1, 1, targetIterations - escapeIteration].filter(value => value > 0)
    });
  }
  return schedules;
}
