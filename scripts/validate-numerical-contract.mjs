import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { platform, release } from 'node:os';
import { resolve } from 'node:path';
import {
  BAILOUT_RADIUS_SQUARED,
  DIRECT_CONTRACT_VERSION,
  DirectStatus,
  SMOOTH_ABSOLUTE_FLOOR,
  SMOOTH_ULP_MULTIPLIER,
  chunkSchedules,
  coordinateAt,
  directF64
} from '../tests/numerical/mandelbrot-contract.mjs';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const fixturePath = resolve(root, 'tests/fixtures/shallow-direct-v1.json');
const shaderPath = resolve(root, 'src/numerical/tileDirectShader.ts');
const browserHarnessPath = resolve(root, 'tests/numerical/direct-gpu-harness.ts');
const fixtureSource = readFileSync(fixturePath, 'utf8');
const shaderSource = readFileSync(shaderPath, 'utf8');
const browserHarnessSource = readFileSync(browserHarnessPath, 'utf8');
const fixture = JSON.parse(fixtureSource);
const failures = [];

function hash(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sameNumber(left, right) {
  return Object.is(left, right) || (Number.isNaN(left) && Number.isNaN(right));
}

function assert(condition, message) {
  if (!condition) failures.push(message);
}

assert(fixture.schemaVersion === 1, `Unsupported fixture schema ${fixture.schemaVersion}`);
assert(fixture.contractVersion === DIRECT_CONTRACT_VERSION, 'Fixture/oracle contract version mismatch');
assert(fixture.bailoutRadiusSquared === BAILOUT_RADIUS_SQUARED, 'Fixture/oracle bailout mismatch');
assert(fixture.escapeComparison === '>', 'The direct contract requires a strict bailout comparison');
assert(fixture.smoothTolerance.absoluteFloor === SMOOTH_ABSOLUTE_FLOOR,
  'Fixture/oracle smooth absolute floor mismatch');
assert(fixture.smoothTolerance.ulpMultiplier === SMOOTH_ULP_MULTIPLIER,
  'Fixture/oracle smooth ULP multiplier mismatch');
for (const needle of [
  'radiusSquared > 256.0',
  'iteration >= iterationEnd || radiusSquared > 256.0',
  'q * (q + shiftedX) <= 0.25 * ySquared',
  'bulbX * bulbX + ySquared <= 0.0625',
  'fma(x, x, -(y * y))',
  'fma(2.0 * x, y, dsValue(coordinate[1]))',
  'f32(pixelX) + 0.5 - halfSize'
]) {
  assert(shaderSource.includes(needle), `Production direct shader no longer matches frozen contract: ${needle}`);
}
assert(browserHarnessSource.includes("from '../../src/numerical/tileDirectShader'"),
  'Browser harness must import the production direct shader.');
for (const forbidden of ['src/app', 'src/presentation', 'src/scheduler', 'tilePerturbationShader']) {
  assert(!browserHarnessSource.includes(forbidden), `Browser harness must remain isolated from ${forbidden}`);
}

const results = [];
for (const testCase of fixture.strictCases) {
  const monolithic = directF64(testCase.cx, testCase.cy, testCase.targetIterations);
  assert(monolithic.status === testCase.expectedStatus,
    `${testCase.id}: status ${monolithic.status} != ${testCase.expectedStatus}`);
  assert(monolithic.iteration === testCase.expectedIteration,
    `${testCase.id}: iteration ${monolithic.iteration} != ${testCase.expectedIteration}`);
  if (monolithic.status === DirectStatus.ESCAPED) {
    assert(sameNumber(monolithic.smooth, testCase.expectedSmooth), `${testCase.id}: smooth golden changed`);
    assert(sameNumber(monolithic.previousRadiusSquared, testCase.expectedPreviousRadiusSquared),
      `${testCase.id}: previous-radius golden changed`);
    assert(sameNumber(monolithic.radiusSquared, testCase.expectedEscapeRadiusSquared),
      `${testCase.id}: escape-radius golden changed`);
  }

  const scheduleResults = [];
  for (const schedule of chunkSchedules(testCase.targetIterations, monolithic.status === DirectStatus.ESCAPED
    ? monolithic.iteration : null)) {
    const chunked = directF64(testCase.cx, testCase.cy, testCase.targetIterations, schedule.chunks);
    for (const key of ['status', 'iteration', 'x', 'y', 'radiusSquared', 'previousRadiusSquared', 'smooth']) {
      assert(sameNumber(chunked[key], monolithic[key]), `${testCase.id}/${schedule.id}: ${key} changed across chunks`);
    }
    scheduleResults.push({ id: schedule.id, chunks: schedule.chunks.length });
  }
  results.push({
    id: testCase.id,
    status: monolithic.status,
    iteration: monolithic.iteration,
    radiusSquared: monolithic.radiusSquared,
    previousRadiusSquared: monolithic.previousRadiusSquared,
    smooth: monolithic.smooth,
    schedules: scheduleResults
  });
}

const byId = new Map(results.map(result => [result.id, result]));
for (const testCase of fixture.strictCases.filter(item => item.pair)) {
  const result = byId.get(testCase.id);
  const pair = byId.get(testCase.pair);
  assert(Boolean(pair), `${testCase.id}: missing conjugate pair ${testCase.pair}`);
  if (result && pair) {
    assert(result.status === pair.status, `${testCase.id}: conjugate status mismatch`);
    assert(result.iteration === pair.iteration, `${testCase.id}: conjugate iteration mismatch`);
    assert(sameNumber(result.smooth, pair.smooth), `${testCase.id}: conjugate smooth mismatch`);
  }
}

const targetEdgeCap = byId.get('target-edge-cap');
const targetEdgeEscape = byId.get('target-edge-escape');
assert(targetEdgeCap?.status === DirectStatus.CAP, 'Target-edge cap fixture did not cap');
assert(targetEdgeEscape?.status === DirectStatus.ESCAPED, 'Escape discovered at the target must win over cap');
const equalityCap = byId.get('strict-bailout-equality-cap');
assert(equalityCap?.radiusSquared === BAILOUT_RADIUS_SQUARED,
  'Exact bailout equality must remain capped under the strict comparison');

const gridResults = [];
for (const grid of fixture.gridCases) {
  const classes = { escaped: 0, analytic: 0, capped: 0, nonFinite: 0 };
  let explicitIterations = 0;
  for (let y = 0; y < grid.tileSize; y += 1) {
    const cy = coordinateAt(grid.centerY, y, grid.tileSize, grid.sampleExponent);
    for (let x = 0; x < grid.tileSize; x += 1) {
      const cx = coordinateAt(grid.centerX, x, grid.tileSize, grid.sampleExponent);
      const monolithic = directF64(cx, cy, grid.targetIterations);
      const chunked = directF64(cx, cy, grid.targetIterations, grid.chunks);
      for (const key of ['status', 'iteration', 'x', 'y', 'radiusSquared', 'smooth']) {
        assert(sameNumber(chunked[key], monolithic[key]), `${grid.id}[${x},${y}]: ${key} changed across chunks`);
      }
      explicitIterations += monolithic.explicitIterations;
      if (monolithic.status === DirectStatus.ESCAPED) classes.escaped += 1;
      else if (monolithic.status === DirectStatus.ANALYTIC_INTERIOR) classes.analytic += 1;
      else if (monolithic.status === DirectStatus.CAP) classes.capped += 1;
      else classes.nonFinite += 1;
    }
  }
  gridResults.push({ id: grid.id, pixels: grid.tileSize ** 2, classes, explicitIterations });
}

const benchmark = fixture.benchmark;
const cpuRuns = [];
let benchmarkSummary = null;
for (let run = 0; run < 3; run += 1) {
  const started = performance.now();
  const classes = { escaped: 0, analytic: 0, capped: 0, nonFinite: 0 };
  let explicitIterations = 0;
  for (let y = 0; y < benchmark.tileSize; y += 1) {
    const cy = coordinateAt(benchmark.centerY, y, benchmark.tileSize, benchmark.sampleExponent);
    for (let x = 0; x < benchmark.tileSize; x += 1) {
      const cx = coordinateAt(benchmark.centerX, x, benchmark.tileSize, benchmark.sampleExponent);
      const result = directF64(cx, cy, benchmark.targetIterations, [benchmark.chunkIterations]);
      explicitIterations += result.explicitIterations;
      if (result.status === DirectStatus.ESCAPED) classes.escaped += 1;
      else if (result.status === DirectStatus.ANALYTIC_INTERIOR) classes.analytic += 1;
      else if (result.status === DirectStatus.CAP) classes.capped += 1;
      else classes.nonFinite += 1;
    }
  }
  const wallMs = performance.now() - started;
  cpuRuns.push(wallMs);
  benchmarkSummary = { classes, explicitIterations };
}

const sortedCpuRuns = [...cpuRuns].sort((left, right) => left - right);
const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  contractVersion: DIRECT_CONTRACT_VERSION,
  fixture: {
    path: 'tests/fixtures/shallow-direct-v1.json',
    sha256: hash(fixtureSource),
    strictCases: fixture.strictCases.length,
    sensitivityCases: fixture.sensitivityCases.length,
    gridCases: fixture.gridCases.length
  },
  productionShader: {
    path: 'src/numerical/tileDirectShader.ts',
    sha256: hash(shaderSource)
  },
  environment: {
    node: process.version,
    os: `${platform()} ${release()}`
  },
  strictResults: results,
  gridResults,
  cpuBenchmark: {
    scene: benchmark.id,
    metric: 'CPU f64 oracle wall time; excludes I/O',
    samplesMs: cpuRuns,
    medianMs: sortedCpuRuns[Math.floor(sortedCpuRuns.length / 2)],
    ...benchmarkSummary
  },
  failures
};

const outputDirectory = resolve(root, process.env.NUMERICAL_OUTPUT ?? 'test-results/numerical');
mkdirSync(outputDirectory, { recursive: true });
writeFileSync(resolve(outputDirectory, 'cpu-oracle.json'), JSON.stringify(report, null, 2));

if (failures.length) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log(
  `Validated ${fixture.strictCases.length} strict direct-oracle cases across chunk schedules; `
  + `CPU benchmark median ${report.cpuBenchmark.medianMs.toFixed(3)} ms for `
  + `${benchmarkSummary.explicitIterations.toLocaleString()} explicit iterations.`
);
