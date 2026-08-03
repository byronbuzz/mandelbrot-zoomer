import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { platform, release } from 'node:os';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';
import {
  DirectStatus,
  chunkSchedules,
  coordinateAt,
  directF64,
  normalizeGpuResult,
  smoothTolerance
} from '../tests/numerical/mandelbrot-contract.mjs';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const fixturePath = resolve(root, 'tests/fixtures/shallow-direct-v1.json');
const shaderPath = resolve(root, 'src/numerical/tileDirectShader.ts');
const runnerPath = resolve(root, 'scripts/run-numerical-browser-gate.mjs');
const harnessPath = resolve(root, 'tests/numerical/direct-gpu-harness.ts');
const oraclePath = resolve(root, 'tests/numerical/mandelbrot-contract.mjs');
const fixtureSource = readFileSync(fixturePath, 'utf8');
const shaderSource = readFileSync(shaderPath, 'utf8');
const runnerSource = readFileSync(runnerPath, 'utf8');
const harnessSource = readFileSync(harnessPath, 'utf8');
const oracleSource = readFileSync(oraclePath, 'utf8');
const fixture = JSON.parse(fixtureSource);
const runId = (process.env.NUMERICAL_RUN_ID ?? new Date().toISOString().replace(/[:.]/g, '-'))
  .replace(/[^a-zA-Z0-9_-]/g, '-');
const outputRoot = resolve(root, process.env.NUMERICAL_OUTPUT ?? 'test-results/numerical');
const outputDirectory = resolve(outputRoot, runId);
mkdirSync(outputDirectory, { recursive: true });
const port = Number(process.env.NUMERICAL_PORT ?? 4176);
const baseUrl = `http://127.0.0.1:${port}/mandelbrot-zoomer/`;
const browserCandidates = [
  process.env.WEBGPU_BROWSER_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
].filter(Boolean);
const executablePath = browserCandidates.find(candidate => existsSync(candidate));
if (!executablePath) throw new Error('Set WEBGPU_BROWSER_PATH to a Chrome or Edge executable with WebGPU support.');
const viteEntry = resolve(root, 'node_modules/vite/bin/vite.js');
const server = spawn(process.execPath, [
  viteEntry, '--host', '127.0.0.1', '--port', String(port), '--strictPort', '--configLoader', 'runner'
], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
let serverOutput = '';
server.stdout.on('data', chunk => { serverOutput += String(chunk); });
server.stderr.on('data', chunk => { serverOutput += String(chunk); });

function hash(value) {
  return createHash('sha256').update(value).digest('hex');
}

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1)))];
}

function uniqueSchedules(targetIterations, escapeIteration) {
  const unique = new Map();
  for (const schedule of chunkSchedules(targetIterations, escapeIteration)) {
    const key = JSON.stringify(schedule.chunks);
    if (!unique.has(key)) unique.set(key, schedule);
  }
  return Array.from(unique.values());
}

function stableRunHash(run) {
  return hash(JSON.stringify({
    stateBits: run.stateBits,
    meta: run.meta,
    counters: run.counters,
    resultBits: run.resultBits,
    quality: run.quality
  }));
}

function scopedMessages(scopedErrors) {
  return Object.entries(scopedErrors ?? {})
    .filter(([, message]) => Boolean(message))
    .map(([scope, message]) => `${scope}: ${message}`);
}

async function waitForServer() {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}tests/numerical-fixture.html`);
      if (response.ok) return;
    } catch {}
    await new Promise(resolvePromise => setTimeout(resolvePromise, 100));
  }
  throw new Error(`Numerical fixture server did not start.\n${serverOutput}`);
}

const failures = [];
const consoleErrors = [];
let browser;
let context;
let rawGpuReport = null;
let browserVersion = null;
let startedAt = new Date().toISOString();
let terminalStatus = 'running';
let terminalError = null;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath,
    headless: process.env.NUMERICAL_HEADED !== '1',
    args: ['--enable-unsafe-webgpu', '--disable-background-timer-throttling']
  });
  browserVersion = browser.version();
  context = await browser.newContext({ viewport: { width: 1000, height: 700 } });
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  const page = await context.newPage();
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) {
      consoleErrors.push(message.text());
    }
  });
  page.on('pageerror', error => consoleErrors.push(error.message));
  page.on('response', response => {
    if (response.status() >= 400 && !response.url().endsWith('/favicon.ico')) {
      consoleErrors.push(`${response.status()} ${response.url()}`);
    }
  });
  await page.goto(`${baseUrl}tests/numerical-fixture.html`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => Boolean(window.__NUMERICAL_HARNESS__), null, { timeout: 20_000 });

  const strictCases = fixture.strictCases.map(testCase => {
    const oracle = directF64(testCase.cx, testCase.cy, testCase.targetIterations);
    return {
      id: testCase.id,
      cx: testCase.cx,
      cy: testCase.cy,
      targetIterations: testCase.targetIterations,
      schedules: uniqueSchedules(testCase.targetIterations,
        oracle.status === DirectStatus.ESCAPED ? oracle.iteration : null)
    };
  });
  const sensitivityCases = fixture.sensitivityCases.map(testCase => ({
    id: testCase.id,
    cx: testCase.cx,
    cy: testCase.cy,
    targetIterations: testCase.targetIterations,
    schedules: uniqueSchedules(testCase.targetIterations, null)
  }));
  rawGpuReport = await page.evaluate(
    input => window.__NUMERICAL_HARNESS__.runSuite(input),
    { strictCases, sensitivityCases, gridCases: fixture.gridCases, benchmark: fixture.benchmark }
  );
  await page.screenshot({ path: resolve(outputDirectory, 'numerical-fixture.png'), type: 'png' });

  if (rawGpuReport.compilationMessages.some(message => message.type === 'error')) {
    failures.push('Production direct WGSL reported compilation errors.');
  }
  const pipelineErrors = scopedMessages(rawGpuReport.pipelineScopedErrors);
  if (pipelineErrors.length) failures.push(`Pipeline scopes: ${pipelineErrors.join('; ')}`);
  if (rawGpuReport.uncapturedErrors.length) failures.push(`Uncaptured WebGPU errors: ${rawGpuReport.uncapturedErrors.join('; ')}`);
  if (rawGpuReport.deviceLostBeforeCompletion) failures.push('WebGPU device was lost before the suite completed.');
  if (consoleErrors.length) failures.push(`Browser errors: ${consoleErrors.join('; ')}`);

  const strictComparisons = [];
  for (const testCase of fixture.strictCases) {
    const oracle = directF64(testCase.cx, testCase.cy, testCase.targetIterations);
    if (oracle.status !== testCase.expectedStatus || oracle.iteration !== testCase.expectedIteration) {
      failures.push(`${testCase.id}: checked-in golden disagrees with CPU oracle`);
    }
    const gpuCase = rawGpuReport.strictResults.find(result => result.id === testCase.id);
    if (!gpuCase) {
      failures.push(`${testCase.id}: missing GPU result`);
      continue;
    }
    const baseline = gpuCase.schedules.find(schedule => schedule.id === 'monolithic') ?? gpuCase.schedules[0];
    const baselineHash = stableRunHash(baseline.run);
    const scheduleComparisons = [];
    for (const schedule of gpuCase.schedules) {
      const run = schedule.run;
      const normalized = normalizeGpuResult(
        run.meta.slice(0, 4), run.result.slice(0, 4), run.quality.slice(0, 4), testCase.targetIterations
      );
      const runHash = stableRunHash(run);
      const runScopeErrors = scopedMessages(run.scopedErrors);
      if (runScopeErrors.length) failures.push(`${testCase.id}/${schedule.id}: ${runScopeErrors.join('; ')}`);
      if (normalized.status !== oracle.status) {
        failures.push(`${testCase.id}/${schedule.id}: GPU status ${normalized.status} != oracle ${oracle.status}`);
      }
      if (normalized.iteration !== oracle.iteration) {
        failures.push(`${testCase.id}/${schedule.id}: GPU iteration ${normalized.iteration} != oracle ${oracle.iteration}`);
      }
      if (runHash !== baselineHash) failures.push(`${testCase.id}/${schedule.id}: chunk schedule changed output bits`);
      const expectedCounters = [0, 0, 0, 0, 0, 0, 0];
      if (testCase.expectedStatus === DirectStatus.ESCAPED) expectedCounters[1] = 1;
      else if (testCase.expectedStatus === DirectStatus.ANALYTIC_INTERIOR) expectedCounters[2] = 1;
      else if (testCase.expectedStatus === DirectStatus.CAP) expectedCounters[3] = 1;
      else if (testCase.expectedStatus === DirectStatus.NON_FINITE) expectedCounters[4] = 1;
      if (JSON.stringify(run.counters) !== JSON.stringify(expectedCounters)) {
        failures.push(`${testCase.id}/${schedule.id}: counters ${run.counters} != ${expectedCounters}`);
      }
      let smoothAbsoluteError = null;
      if (oracle.status === DirectStatus.ESCAPED) {
        smoothAbsoluteError = Math.abs(normalized.smooth - oracle.smooth);
        const tolerance = smoothTolerance(oracle.smooth);
        if (!Number.isFinite(smoothAbsoluteError) || smoothAbsoluteError > tolerance) {
          failures.push(`${testCase.id}/${schedule.id}: smooth error ${smoothAbsoluteError} > ${tolerance}`);
        }
      }
      scheduleComparisons.push({
        id: schedule.id,
        chunks: schedule.chunks,
        outputSha256: runHash,
        normalized,
        smoothAbsoluteError,
        queueCompletionWallMs: run.queueCompletionWallMs,
        readbackWallMs: run.readbackWallMs
      });
    }
    strictComparisons.push({
      id: testCase.id,
      oracle: {
        status: oracle.status,
        iteration: oracle.iteration,
        smooth: oracle.smooth,
        previousRadiusSquared: oracle.previousRadiusSquared,
        radiusSquared: oracle.radiusSquared
      },
      schedules: scheduleComparisons
    });
  }

  const gridComparisons = [];
  for (const grid of fixture.gridCases) {
    const gpuGrid = rawGpuReport.gridResults.find(result => result.id === grid.id);
    if (!gpuGrid) {
      failures.push(`${grid.id}: missing GPU grid result`);
      continue;
    }
    const gridScopeErrors = scopedMessages(gpuGrid.run.scopedErrors);
    if (gridScopeErrors.length) failures.push(`${grid.id}: ${gridScopeErrors.join('; ')}`);
    const classCounts = { escaped: 0, analytic: 0, capped: 0, nonFinite: 0 };
    let explicitIterations = 0;
    let maxSmoothAbsoluteError = 0;
    for (let y = 0; y < grid.tileSize; y += 1) {
      const cy = coordinateAt(grid.centerY, y, grid.tileSize, grid.sampleExponent);
      for (let x = 0; x < grid.tileSize; x += 1) {
        const cx = coordinateAt(grid.centerX, x, grid.tileSize, grid.sampleExponent);
        const oracle = directF64(cx, cy, grid.targetIterations, grid.chunks);
        const pixel = y * grid.tileSize + x;
        const normalized = normalizeGpuResult(
          gpuGrid.run.meta.slice(pixel * 4, pixel * 4 + 4),
          gpuGrid.run.result.slice(pixel * 4, pixel * 4 + 4),
          gpuGrid.run.quality.slice(pixel * 4, pixel * 4 + 4),
          grid.targetIterations
        );
        explicitIterations += oracle.explicitIterations;
        if (oracle.status === DirectStatus.ESCAPED) classCounts.escaped += 1;
        else if (oracle.status === DirectStatus.ANALYTIC_INTERIOR) classCounts.analytic += 1;
        else if (oracle.status === DirectStatus.CAP) classCounts.capped += 1;
        else classCounts.nonFinite += 1;
        if (normalized.status !== oracle.status || normalized.iteration !== oracle.iteration) {
          failures.push(`${grid.id}[${x},${y}]: GPU ${normalized.status}/${normalized.iteration} != oracle ${oracle.status}/${oracle.iteration}`);
        }
        if (oracle.status === DirectStatus.ESCAPED) {
          const smoothError = Math.abs(normalized.smooth - oracle.smooth);
          maxSmoothAbsoluteError = Math.max(maxSmoothAbsoluteError, smoothError);
          if (!Number.isFinite(smoothError) || smoothError > smoothTolerance(oracle.smooth)) {
            failures.push(`${grid.id}[${x},${y}]: smooth error ${smoothError}`);
          }
        }
      }
    }
    const expectedCounters = [0, classCounts.escaped, classCounts.analytic, classCounts.capped,
      classCounts.nonFinite, 0, 0];
    if (JSON.stringify(gpuGrid.run.counters) !== JSON.stringify(expectedCounters)) {
      failures.push(`${grid.id}: counters ${gpuGrid.run.counters} != ${expectedCounters}`);
    }
    gridComparisons.push({
      id: grid.id,
      pixels: grid.tileSize ** 2,
      classCounts,
      explicitIterations,
      maxSmoothAbsoluteError,
      outputSha256: stableRunHash(gpuGrid.run)
    });
  }

  const sensitivityComparisons = rawGpuReport.sensitivityResults.map(result => {
    const baselineHash = stableRunHash(result.schedules[0].run);
    const schedules = result.schedules.map(schedule => {
      const scopeErrors = scopedMessages(schedule.run.scopedErrors);
      if (scopeErrors.length) failures.push(`${result.id}/${schedule.id}: ${scopeErrors.join('; ')}`);
      const outputSha256 = stableRunHash(schedule.run);
      if (outputSha256 !== baselineHash) failures.push(`${result.id}/${schedule.id}: chunk schedule changed output bits`);
      const normalized = normalizeGpuResult(
        schedule.run.meta.slice(0, 4), schedule.run.result.slice(0, 4), schedule.run.quality.slice(0, 4),
        fixture.sensitivityCases.find(item => item.id === result.id).targetIterations
      );
      if (normalized.status === DirectStatus.NON_FINITE) failures.push(`${result.id}: non-finite sensitivity result`);
      return { id: schedule.id, outputSha256, normalized };
    });
    return { id: result.id, schedules };
  });
  const sensitivityById = new Map(sensitivityComparisons.map(result => [result.id, result]));
  const signedPositive = sensitivityById.get('signed-zero-positive')?.schedules[0]?.outputSha256;
  const signedNegative = sensitivityById.get('signed-zero-negative')?.schedules[0]?.outputSha256;
  if (!signedPositive || signedPositive !== signedNegative) failures.push('Signed-zero conjugacy changed direct output bits.');
  if (sensitivityById.get('cusp-f32-inside')?.schedules[0]?.normalized.status !== DirectStatus.ANALYTIC_INTERIOR) {
    failures.push('Inside-cusp sensitivity point lost analytic classification.');
  }
  if (sensitivityById.get('cusp-f32-outside')?.schedules[0]?.normalized.status === DirectStatus.ANALYTIC_INTERIOR) {
    failures.push('Outside-cusp sensitivity point was incorrectly admitted analytically.');
  }

  const continuationScopeErrors = scopedMessages(rawGpuReport.continuation.scopedErrors);
  if (continuationScopeErrors.length) failures.push(`continuation: ${continuationScopeErrors.join('; ')}`);
  const [suppressed, published, continued] = rawGpuReport.continuation.stages;
  const suppressedNormalized = normalizeGpuResult(suppressed.meta, suppressed.result, suppressed.quality, 2);
  const publishedNormalized = normalizeGpuResult(published.meta, published.result, published.quality, 2);
  const continuedNormalized = normalizeGpuResult(continued.meta, continued.result, continued.quality, 3);
  if (suppressedNormalized.status !== DirectStatus.ACTIVE || suppressedNormalized.iteration !== 2) {
    failures.push('Suppressed cap did not retain active recurrence metadata at iteration 2.');
  }
  if (suppressed.quality[0] !== 0) failures.push('Suppressed cap published accepted quality.');
  if (publishedNormalized.status !== DirectStatus.CAP || publishedNormalized.iteration !== 2) {
    failures.push('Accepted cap was not published at iteration 2.');
  }
  if (continuedNormalized.status !== DirectStatus.ESCAPED || continuedNormalized.iteration !== 3) {
    failures.push('Capped recurrence did not continue to escape at iteration 3.');
  }
  const continuationComparison = {
    capSuppressed: suppressedNormalized,
    capPublished: publishedNormalized,
    continuedEscape: continuedNormalized
  };

  const measuredRuns = rawGpuReport.benchmark.runs.filter(run => !run.warmup);
  for (const run of rawGpuReport.benchmark.runs) {
    const benchmarkScopeErrors = scopedMessages(run.scopedErrors);
    if (benchmarkScopeErrors.length) failures.push(`benchmark: ${benchmarkScopeErrors.join('; ')}`);
    const counterTotal = run.counters.slice(0, 5).reduce((sum, value) => sum + value, 0);
    if (counterTotal !== fixture.benchmark.tileSize ** 2) {
      failures.push(`benchmark: final counters cover ${counterTotal} pixels`);
    }
    if ((run.counters[4] ?? 0) !== 0) failures.push(`benchmark: ${run.counters[4]} non-finite pixels`);
  }
  const queueTimes = measuredRuns.map(run => run.queueCompletionWallMs);
  const readbackTimes = measuredRuns.map(run => run.readbackWallMs);
  const explicitIterations = measuredRuns[0]?.explicitIterations ?? 0;
  for (const run of measuredRuns) {
    if (run.explicitIterations !== explicitIterations
      || JSON.stringify(run.counters) !== JSON.stringify(measuredRuns[0].counters)) {
      failures.push('Benchmark measured runs produced inconsistent work/class counters.');
      break;
    }
  }
  const benchmarkReport = {
    definition: rawGpuReport.benchmark.definition,
    chunks: rawGpuReport.benchmark.chunks,
    metric: 'fresh-run queue completion latency; includes prior uniform uploads and possible lazy initialization; telemetry only',
    queueCompletionWallMs: {
      samples: queueTimes,
      p50: percentile(queueTimes, 0.5),
      p95: percentile(queueTimes, 0.95),
      min: Math.min(...queueTimes),
      max: Math.max(...queueTimes)
    },
    readbackWallMs: {
      samples: readbackTimes,
      p50: percentile(readbackTimes, 0.5),
      p95: percentile(readbackTimes, 0.95)
    },
    explicitIterations,
    finalCounters: measuredRuns[0]?.counters ?? []
  };

  const report = {
    schemaVersion: 1,
    startedAt,
    completedAt: new Date().toISOString(),
    pass: failures.length === 0,
    commit: process.env.GITHUB_SHA ?? 'local-no-git',
    fixture: { path: 'tests/fixtures/shallow-direct-v1.json', sha256: hash(fixtureSource) },
    productionShader: { path: 'src/numerical/tileDirectShader.ts', sha256: hash(shaderSource) },
    harnessSources: {
      runner: { path: 'scripts/run-numerical-browser-gate.mjs', sha256: hash(runnerSource) },
      browser: { path: 'tests/numerical/direct-gpu-harness.ts', sha256: hash(harnessSource) },
      oracle: { path: 'tests/numerical/mandelbrot-contract.mjs', sha256: hash(oracleSource) }
    },
    environment: {
      os: `${platform()} ${release()}`,
      node: process.version,
      browser: browserVersion,
      executablePath,
      adapter: rawGpuReport.adapter
    },
    validation: {
      compilationMessages: rawGpuReport.compilationMessages,
      pipelineScopedErrors: rawGpuReport.pipelineScopedErrors,
      uncapturedErrors: rawGpuReport.uncapturedErrors,
      deviceLostBeforeCompletion: rawGpuReport.deviceLostBeforeCompletion,
      consoleErrors
    },
    strictComparisons,
    gridComparisons,
    sensitivityComparisons,
    continuationComparison,
    benchmark: benchmarkReport,
    failures
  };
  writeFileSync(resolve(outputDirectory, 'browser-report.json'), JSON.stringify(report, null, 2));
  if (failures.length) throw new Error(failures.join('\n'));
  terminalStatus = 'passed';
  console.log(
    `Numerical browser gate passed ${strictComparisons.length} strict cases on `
    + `${rawGpuReport.adapter.info.vendor || 'unlabelled adapter'}; fresh-run queue latency p50 `
    + `${benchmarkReport.queueCompletionWallMs.p50.toFixed(3)} ms; `
    + `${explicitIterations.toLocaleString()} explicit iterations observed.`
  );
} catch (error) {
  terminalStatus = 'failed';
  terminalError = error instanceof Error ? error.stack ?? error.message : String(error);
  const failurePath = resolve(outputDirectory, 'browser-failure.json');
  writeFileSync(failurePath, JSON.stringify({
    schemaVersion: 1,
    startedAt,
    failedAt: new Date().toISOString(),
    error: terminalError,
    failures,
    consoleErrors,
    serverOutput,
    browserVersion
  }, null, 2));
  throw error;
} finally {
  const cleanup = [];
  if (context) cleanup.push(context.tracing.stop({ path: resolve(outputDirectory, 'numerical-playwright-trace.zip') }));
  if (browser) cleanup.push(browser.close());
  await Promise.allSettled(cleanup);
  server.kill();
  writeFileSync(resolve(outputDirectory, 'manifest.json'), JSON.stringify({
    schemaVersion: 1,
    runId,
    status: terminalStatus,
    startedAt,
    completedAt: new Date().toISOString(),
    commit: process.env.GITHUB_SHA ?? 'local-no-git',
    report: terminalStatus === 'passed' ? 'browser-report.json' : null,
    failure: terminalStatus === 'failed' ? 'browser-failure.json' : null,
    error: terminalError
  }, null, 2));
}
