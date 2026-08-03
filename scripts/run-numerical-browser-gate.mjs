import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { platform, release } from 'node:os';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';
import {
  DirectStatus,
  chunkSchedules,
  directF64,
  normalizeGpuResult
} from '../tests/numerical/mandelbrot-contract.mjs';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const fixturePath = resolve(root, 'tests/fixtures/shallow-direct-v1.json');
const shaderPath = resolve(root, 'src/numerical/tileDirectShader.ts');
const fixtureSource = readFileSync(fixturePath, 'utf8');
const shaderSource = readFileSync(shaderPath, 'utf8');
const fixture = JSON.parse(fixtureSource);
const outputDirectory = resolve(root, process.env.NUMERICAL_OUTPUT ?? 'test-results/numerical');
mkdirSync(outputDirectory, { recursive: true });
const port = Number(process.env.NUMERICAL_PORT ?? 4176);
const baseUrl = `http://127.0.0.1:${port}/mandelbrot-zoomer/`;
const viteEntry = resolve(root, 'node_modules/vite/bin/vite.js');
const server = spawn(process.execPath, [
  viteEntry, '--host', '127.0.0.1', '--port', String(port), '--strictPort', '--configLoader', 'runner'
], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
let serverOutput = '';
server.stdout.on('data', chunk => { serverOutput += String(chunk); });
server.stderr.on('data', chunk => { serverOutput += String(chunk); });

const browserCandidates = [
  process.env.WEBGPU_BROWSER_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
].filter(Boolean);
const executablePath = browserCandidates.find(candidate => existsSync(candidate));
if (!executablePath) throw new Error('Set WEBGPU_BROWSER_PATH to a Chrome or Edge executable with WebGPU support.');

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
    resultBits: run.resultBits
  }));
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
    schedules: [{ id: 'monolithic', chunks: [testCase.targetIterations] }]
  }));
  rawGpuReport = await page.evaluate(
    input => window.__NUMERICAL_HARNESS__.runSuite(input),
    { strictCases, sensitivityCases, benchmark: fixture.benchmark }
  );
  await page.screenshot({ path: resolve(outputDirectory, 'numerical-fixture.png'), type: 'png' });

  if (rawGpuReport.compilationMessages.some(message => message.type === 'error')) {
    failures.push('Production direct WGSL reported compilation errors.');
  }
  if (rawGpuReport.pipelineValidationError) failures.push(`Pipeline validation: ${rawGpuReport.pipelineValidationError}`);
  if (rawGpuReport.uncapturedErrors.length) failures.push(`Uncaptured WebGPU errors: ${rawGpuReport.uncapturedErrors.join('; ')}`);
  if (rawGpuReport.deviceLostBeforeCompletion) failures.push('WebGPU device was lost before the suite completed.');
  if (consoleErrors.length) failures.push(`Browser errors: ${consoleErrors.join('; ')}`);

  const strictComparisons = [];
  for (const testCase of fixture.strictCases) {
    const oracle = directF64(testCase.cx, testCase.cy, testCase.targetIterations);
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
      const normalized = normalizeGpuResult(run.meta.slice(0, 4), run.result.slice(0, 4));
      const runHash = stableRunHash(run);
      if (run.scopedErrors.length) failures.push(`${testCase.id}/${schedule.id}: ${run.scopedErrors.join('; ')}`);
      if (normalized.status !== testCase.expectedStatus) {
        failures.push(`${testCase.id}/${schedule.id}: GPU status ${normalized.status} != ${testCase.expectedStatus}`);
      }
      if (normalized.iteration !== testCase.expectedIteration) {
        failures.push(`${testCase.id}/${schedule.id}: GPU iteration ${normalized.iteration} != ${testCase.expectedIteration}`);
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
        if (!Number.isFinite(smoothAbsoluteError) || smoothAbsoluteError > 0.02) {
          failures.push(`${testCase.id}/${schedule.id}: smooth error ${smoothAbsoluteError} > 0.02`);
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

  const measuredRuns = rawGpuReport.benchmark.runs.filter(run => !run.warmup);
  for (const run of rawGpuReport.benchmark.runs) {
    if (run.scopedErrors.length) failures.push(`benchmark: ${run.scopedErrors.join('; ')}`);
    const counterTotal = run.counters.slice(0, 5).reduce((sum, value) => sum + value, 0);
    if (counterTotal !== fixture.benchmark.tileSize ** 2) {
      failures.push(`benchmark: final counters cover ${counterTotal} pixels`);
    }
    if ((run.counters[4] ?? 0) !== 0) failures.push(`benchmark: ${run.counters[4]} non-finite pixels`);
  }
  const queueTimes = measuredRuns.map(run => run.queueCompletionWallMs);
  const readbackTimes = measuredRuns.map(run => run.readbackWallMs);
  const explicitIterations = measuredRuns[0]?.explicitIterations ?? 0;
  const benchmarkReport = {
    definition: rawGpuReport.benchmark.definition,
    chunks: rawGpuReport.benchmark.chunks,
    metric: 'queue completion wall time; compilation, allocation and readback excluded; telemetry only',
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
    explicitIterationsPerSecondAtP50: explicitIterations / (percentile(queueTimes, 0.5) / 1000),
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
    environment: {
      os: `${platform()} ${release()}`,
      node: process.version,
      browser: browserVersion,
      executablePath,
      adapter: rawGpuReport.adapter
    },
    validation: {
      compilationMessages: rawGpuReport.compilationMessages,
      pipelineValidationError: rawGpuReport.pipelineValidationError,
      uncapturedErrors: rawGpuReport.uncapturedErrors,
      deviceLostBeforeCompletion: rawGpuReport.deviceLostBeforeCompletion,
      consoleErrors
    },
    strictComparisons,
    sensitivityResults: rawGpuReport.sensitivityResults,
    benchmark: benchmarkReport,
    failures
  };
  writeFileSync(resolve(outputDirectory, 'browser-report.json'), JSON.stringify(report, null, 2));
  if (failures.length) throw new Error(failures.join('\n'));
  console.log(
    `Numerical browser gate passed ${strictComparisons.length} strict cases on `
    + `${rawGpuReport.adapter.info.vendor || 'unlabelled adapter'}; benchmark p50 `
    + `${benchmarkReport.queueCompletionWallMs.p50.toFixed(3)} ms for `
    + `${explicitIterations.toLocaleString()} explicit iterations.`
  );
} catch (error) {
  const failurePath = resolve(outputDirectory, 'browser-failure.json');
  writeFileSync(failurePath, JSON.stringify({
    schemaVersion: 1,
    startedAt,
    failedAt: new Date().toISOString(),
    error: error instanceof Error ? error.stack ?? error.message : String(error),
    failures,
    consoleErrors,
    serverOutput,
    browserVersion
  }, null, 2));
  throw error;
} finally {
  if (context) await context.tracing.stop({ path: resolve(outputDirectory, 'numerical-playwright-trace.zip') });
  if (browser) await browser.close();
  server.kill();
}
