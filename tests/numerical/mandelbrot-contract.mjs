export const DIRECT_CONTRACT_VERSION = 1;
export const BAILOUT_RADIUS_SQUARED = 256;

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

export function directF64(cx, cy, targetIterations, chunks = [targetIterations]) {
  if (!Number.isInteger(targetIterations) || targetIterations < 0) {
    throw new Error(`Invalid target iteration count ${targetIterations}`);
  }
  if (!Array.isArray(chunks) || chunks.length === 0 || chunks.some(value => !Number.isInteger(value) || value <= 0)) {
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
    const chunkIterations = chunks[chunkIndex % chunks.length];
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

export function normalizeGpuResult(meta, result) {
  const iteration = meta[0];
  const metaStatus = meta[1];
  const resultStatus = Math.round(result[1]);
  if (metaStatus === DirectStatus.NON_FINITE) {
    return { status: DirectStatus.NON_FINITE, iteration, smooth: null };
  }
  if (metaStatus === DirectStatus.ESCAPED || resultStatus === DirectStatus.ESCAPED) {
    return { status: DirectStatus.ESCAPED, iteration, smooth: result[0] };
  }
  if (metaStatus === DirectStatus.ANALYTIC_INTERIOR || resultStatus === DirectStatus.ANALYTIC_INTERIOR) {
    return { status: DirectStatus.ANALYTIC_INTERIOR, iteration, smooth: null };
  }
  if (iteration >= Math.round(result[2]) && resultStatus === DirectStatus.CAP) {
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
