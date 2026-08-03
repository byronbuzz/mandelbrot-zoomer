import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const fixture = JSON.parse(readFileSync(resolve(root, 'tests/fixtures/sharpness-boundary-start.json'), 'utf8'));
const outputDirectory = resolve(root, process.env.PERFORMANCE_OUTPUT ?? 'test-results/performance-stage-throughput');
mkdirSync(outputDirectory, { recursive: true });
const port = Number(process.env.PERFORMANCE_PORT ?? 4177);
const localUrl = `http://127.0.0.1:${port}/mandelbrot-zoomer/`;
const baselineUrl = process.env.PERFORMANCE_BASELINE
  ?? 'https://byronbuzz.github.io/mandelbrot-zoomer/?deploy=501804c';
const expectedBaselineBuild = process.env.PERFORMANCE_BASELINE_BUILD ?? '1.4 · 501804c';
const viteEntry = resolve(root, 'node_modules/vite/bin/vite.js');
const requireImprovement = process.env.PERFORMANCE_REQUIRE_IMPROVEMENT !== '0';
const minimumNeutralRatio = Number(process.env.PERFORMANCE_MINIMUM_NEUTRAL_RATIO ?? 0.95);
const server = spawn(process.execPath, [
  viteEntry, 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort', '--configLoader', 'runner'
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
if (!executablePath) throw new Error('No hardware WebGPU Chrome/Edge executable was found.');

async function waitForServer() {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(localUrl)).ok) return;
    } catch {}
    await new Promise(resolvePromise => setTimeout(resolvePromise, 100));
  }
  throw new Error(`Performance preview did not start.\n${serverOutput}`);
}

function serializedCamera() {
  return {
    centerXRaw: fixture.camera.centerX.raw,
    centerXBits: fixture.camera.centerX.bits,
    centerYRaw: fixture.camera.centerY.raw,
    centerYBits: fixture.camera.centerY.bits,
    scaleMantissa: fixture.camera.scale.mantissa,
    scaleExponent: fixture.camera.scale.exponent
  };
}

async function measure(context, url, label, expectedBuild = null) {
  const page = await context.newPage();
  const errors = [];
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) {
      errors.push(message.text());
    }
  });
  page.on('pageerror', error => errors.push(error.message));
  const separator = url.includes('?') ? '&' : '?';
  await page.goto(`${url}${separator}testIterations=${fixture.iterationTarget}&continuityTest=1&performanceGate=1`, {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(() => Boolean(window.__ZOOMER_TEST__), null, { timeout: 20_000 });
  const loadedBuild = await page.evaluate(() => (
    window.__ZOOMER_TEST__.diagnostics().build ?? window.__ZOOMER_DIAGNOSTICS__?.().build
  ));
  if (expectedBuild !== null && loadedBuild !== expectedBuild) {
    throw new Error(`${label} loaded ${loadedBuild}; expected immutable baseline ${expectedBuild}.`);
  }
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('paused'));
  const target = await page.evaluate(
    camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'moving'),
    serializedCamera()
  );
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('running'));
  const samples = [];
  const started = Date.now();
  for (const targetMs of [250, 500, 1000, 1500]) {
    const remaining = targetMs - (Date.now() - started);
    if (remaining > 0) await page.waitForTimeout(remaining);
    const diagnostics = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics());
    samples.push({ elapsedMs: Date.now() - started, ...diagnostics });
  }
  await page.screenshot({ path: resolve(outputDirectory, `${label}-1500ms.png`), type: 'png' });
  const settledTarget = await page.evaluate(
    camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'settled'),
    serializedCamera()
  );
  const settledFrame = await page.evaluate(
    ({ viewRevision, requestId }) => window.__ZOOMER_TEST__.waitForResolved(viewRevision, requestId, 0),
    settledTarget
  );
  const settledDiagnostics = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics());
  await page.screenshot({ path: resolve(outputDirectory, `${label}-settled.png`), type: 'png' });
  await page.close();
  return { label, loadedBuild, target, samples, settledTarget, settledFrame, settledDiagnostics, errors };
}

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath,
    headless: true,
    args: ['--enable-unsafe-webgpu', '--disable-background-timer-throttling']
  });
  const context = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  const baseline = await measure(context, baselineUrl, 'baseline-501804c', expectedBaselineBuild);
  const optimized = await measure(context, localUrl, 'optimized');
  const baseline1000 = baseline.samples.find(sample => sample.elapsedMs >= 1000).field.completedChunks;
  const optimized1000 = optimized.samples.find(sample => sample.elapsedMs >= 1000).field.completedChunks;
  const report = {
    schemaVersion: 1,
    capturedAt: new Date().toISOString(),
    adapter: optimized.samples.at(-1).adapterLabel,
    fixture: fixture.id,
    baselineUrl,
    expectedBaselineBuild,
    baseline,
    optimized,
    comparison: {
      baselineCompletedChunksAt1000Ms: baseline1000,
      optimizedCompletedChunksAt1000Ms: optimized1000,
      completedChunkRatioAt1000Ms: optimized1000 / Math.max(1, baseline1000)
    }
  };
  writeFileSync(resolve(outputDirectory, 'report.json'), JSON.stringify(report, null, 2));
  const failures = [...baseline.errors, ...optimized.errors];
  if (/swiftshader|software|llvmpipe/i.test(report.adapter)) failures.push(`Hardware GPU required, got ${report.adapter}`);
  if (optimized.samples.some(sample => sample.presentation.validationErrors !== 0)) {
    failures.push('Optimized trace reported WebGPU validation errors.');
  }
  for (const [label, result] of [['baseline', baseline], ['optimized', optimized]]) {
    const frame = result.settledFrame;
    if (frame.invalidPixels !== 0
      || frame.provisionalCapPixels !== 0
      || frame.qualityRegressionPixels !== 0
      || frame.escapedToProvisionalBlackPixels !== 0
      || frame.conflictPixels !== 0) {
      failures.push(`${label} settled cap-publication transition failed continuity checks.`);
    }
  }
  if (requireImprovement && optimized1000 <= baseline1000) {
    failures.push(`Optimized movement completed ${optimized1000} chunks versus baseline ${baseline1000}.`);
  } else if (!requireImprovement && optimized1000 / Math.max(1, baseline1000) < minimumNeutralRatio) {
    failures.push(
      `Precision-stage movement regressed to ${optimized1000} chunks versus baseline ${baseline1000}`
      + ` (minimum ${minimumNeutralRatio.toFixed(2)}x).`
    );
  }
  if (failures.length) throw new Error(failures.join('\n'));
  console.log(
    `${requireImprovement ? 'Performance' : 'No-regression'} browser gate passed on ${report.adapter}: `
    + `${baseline1000} -> ${optimized1000} completed chunks at 1 s `
    + `(${report.comparison.completedChunkRatioAt1000Ms.toFixed(2)}x).`
  );
} finally {
  if (browser) await browser.close();
  server.kill();
}
