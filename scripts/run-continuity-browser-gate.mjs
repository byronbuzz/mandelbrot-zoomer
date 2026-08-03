import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const fixturePath = resolve(root, process.env.CONTINUITY_FIXTURE ?? 'tests/fixtures/continuity-boundary.json');
const fixtureSource = JSON.parse(readFileSync(fixturePath, 'utf8'));
const fixture = fixtureSource.camera?.centerX
  ? {
    ...fixtureSource,
    deepOrder: fixtureSource.deepOrder ?? fixtureSource.camera.log10Magnification,
    coarseOrder: fixtureSource.coarseOrder ?? 2,
    camera: {
      centerXRaw: fixtureSource.camera.centerX.raw,
      centerXBits: fixtureSource.camera.centerX.bits,
      centerYRaw: fixtureSource.camera.centerY.raw,
      centerYBits: fixtureSource.camera.centerY.bits,
      scaleMantissa: fixtureSource.camera.scale.mantissa,
      scaleExponent: fixtureSource.camera.scale.exponent
    }
  }
  : fixtureSource;
if (fixture.provisional && process.env.ALLOW_PROVISIONAL_FIXTURE !== '1') {
  throw new Error('The continuity fixture is provisional. Capture the user-selected complex boundary fixture before the release gate.');
}

const outputDirectory = resolve(root, process.env.CONTINUITY_OUTPUT ?? 'test-results/continuity');
mkdirSync(outputDirectory, { recursive: true });
const port = Number(process.env.CONTINUITY_PORT ?? 4174);
const baseUrl = `http://127.0.0.1:${port}/mandelbrot-zoomer/`;
const nodeExecutable = process.execPath;
const viteEntry = resolve(root, 'node_modules/vite/bin/vite.js');
const server = spawn(nodeExecutable, [
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
if (!executablePath) throw new Error('Set WEBGPU_BROWSER_PATH to a Chrome or Edge executable with WebGPU support.');

function normalizeScale(value) {
  if (!(value > 0) || !Number.isFinite(value)) throw new Error(`Invalid scale ${value}`);
  const exponent = Math.floor(Math.log2(value)) + 1;
  return { mantissa: value / 2 ** exponent, exponent };
}

function cameraAtOrder(order) {
  const deepScale = fixture.camera.scaleMantissa * 2 ** fixture.camera.scaleExponent;
  const scale = normalizeScale(deepScale * 10 ** (fixture.deepOrder - order));
  return { ...fixture.camera, scaleMantissa: scale.mantissa, scaleExponent: scale.exponent };
}

async function waitForServer() {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) return;
    } catch {}
    await new Promise(resolvePromise => setTimeout(resolvePromise, 100));
  }
  throw new Error(`Preview server did not start.\n${serverOutput}`);
}

function assertFrame(frame, label, requestId, minimumBatchRevision = 0) {
  const failures = [];
  if (frame.requestId !== requestId) failures.push(`request ${frame.requestId} != ${requestId}`);
  if (frame.completedBatchRevision < minimumBatchRevision) failures.push(
    `batch revision ${frame.completedBatchRevision} < ${minimumBatchRevision}`
  );
  if (frame.invalidPixels !== 0) failures.push(`${frame.invalidPixels} invalid pixels`);
  if (frame.qualityRegressionPixels !== 0) failures.push(`${frame.qualityRegressionPixels} quality regressions`);
  if (frame.escapedToProvisionalBlackPixels !== 0) failures.push(
    `${frame.escapedToProvisionalBlackPixels} escaped-to-provisional-black regressions`
  );
  if (frame.conflictPixels !== 0) failures.push(`${frame.conflictPixels} conflict pixels`);
  if (frame.droppedReadbacks !== 0) failures.push(`${frame.droppedReadbacks} dropped readbacks`);
  if (failures.length) throw new Error(`${label}: ${failures.join('; ')}`);
}

function assertNoRegression(frame, label, requestId) {
  const failures = [];
  if (frame.requestId !== requestId) failures.push(`request ${frame.requestId} != ${requestId}`);
  if (frame.qualityRegressionPixels !== 0) failures.push(`${frame.qualityRegressionPixels} quality regressions`);
  if (frame.escapedToProvisionalBlackPixels !== 0) failures.push(
    `${frame.escapedToProvisionalBlackPixels} escaped-to-provisional-black regressions`
  );
  if (frame.conflictPixels !== 0) failures.push(`${frame.conflictPixels} conflict pixels`);
  if (frame.droppedReadbacks !== 0) failures.push(`${frame.droppedReadbacks} dropped readbacks`);
  if (failures.length) throw new Error(`${label}: ${failures.join('; ')}`);
}

async function nextFrameForRequest(page, afterFrame, requestId) {
  return page.evaluate(async ({ afterFrame, requestId }) => {
    let after = afterFrame;
    for (;;) {
      const frame = await window.__ZOOMER_TEST__.nextPresentedFrame(after);
      if (frame.requestId === requestId) return frame;
      after = frame.frameId;
    }
  }, { afterFrame, requestId });
}

let browser;
const records = [];
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath,
    headless: process.env.CONTINUITY_HEADED !== '1',
    args: ['--enable-unsafe-webgpu', '--disable-background-timer-throttling']
  });
  const context = await browser.newContext({ viewport: { width: 1600, height: 900 } });
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) {
      consoleErrors.push(message.text());
    }
  });
  page.on('response', response => {
    if (response.status() >= 400 && !response.url().endsWith('/favicon.ico')) {
      consoleErrors.push(`${response.status()} ${response.url()}`);
    }
  });
  page.on('pageerror', error => consoleErrors.push(error.message));
  await page.goto(`${baseUrl}?testIterations=${fixture.iterationTarget}&continuityTest=1&browserGate=1`, {
    waitUntil: 'domcontentloaded'
  });
  await page.waitForFunction(() => Boolean(window.__ZOOMER_TEST__), null, { timeout: 20_000 });
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('paused'));
  const deep = await page.evaluate(
    camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'settled'),
    cameraAtOrder(fixture.deepOrder)
  );
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('running'));
  const resolved = await page.evaluate(
    ({ viewRevision, requestId }) => window.__ZOOMER_TEST__.waitForResolved(viewRevision, requestId, 0),
    deep
  );
  assertFrame(resolved, 'deep resolved seed', deep.requestId);
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('paused'));

  // A cold deep-to-coarse reveal has no historical pixels outside the deep
  // footprint. It may be uncovered, but it must never regress a coloured pixel.
  const coldBefore = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics().presentation.reducedFrame);
  const coldTarget = await page.evaluate(
    camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'moving'),
    cameraAtOrder(fixture.coarseOrder)
  );
  const coldFrame = await nextFrameForRequest(page, coldBefore, coldTarget.requestId);
  assertNoRegression(coldFrame, 'cold deep-to-coarse transformed', coldTarget.requestId);
  await page.screenshot({
    path: resolve(outputDirectory, 'cold-deep-to-coarse-transformed.png'), type: 'png'
  });
  records.push({ direction: 'cold-out', order: fixture.coarseOrder, target: coldTarget, transformed: coldFrame });

  // Resolve the coarse and deep endpoints once. Their GPU-validated checkpoint
  // surfaces become the retained multiscale coverage for the actual round trip.
  const coarseSeed = await page.evaluate(
    camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'settled'),
    cameraAtOrder(fixture.coarseOrder)
  );
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('running'));
  const coarseResolved = await page.evaluate(
    ({ viewRevision, requestId }) => window.__ZOOMER_TEST__.waitForResolved(viewRevision, requestId, 0),
    coarseSeed
  );
  assertFrame(coarseResolved, 'coarse resolved seed', coarseSeed.requestId);
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('paused'));
  const deepSeed = await page.evaluate(
    camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'settled'),
    cameraAtOrder(fixture.deepOrder)
  );
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('running'));
  const deepResolved = await page.evaluate(
    ({ viewRevision, requestId }) => window.__ZOOMER_TEST__.waitForResolved(viewRevision, requestId, 0),
    deepSeed
  );
  assertFrame(deepResolved, 'warmed deep resolved seed', deepSeed.requestId);
  await page.evaluate(() => window.__ZOOMER_TEST__.setSchedulerGate('paused'));

  const outward = [];
  for (let order = fixture.deepOrder - 1; order >= fixture.coarseOrder; order--) outward.push(order);
  const inward = [];
  for (let order = fixture.coarseOrder + 1; order <= fixture.deepOrder; order++) inward.push(order);
  for (const [direction, orders] of [['out', outward], ['in', inward]]) {
    for (const order of orders) {
      const before = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics().presentation.reducedFrame);
      const target = await page.evaluate(
        camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'moving'),
        cameraAtOrder(order)
      );
      const transformed = await nextFrameForRequest(page, before, target.requestId);
      assertFrame(transformed, `${direction} order ${order} transformed`, target.requestId);
      await page.screenshot({
        path: resolve(outputDirectory, `${direction}-order-${order}-transformed.png`),
        type: 'png'
      });
      const batch = await page.evaluate(() => window.__ZOOMER_TEST__.releaseBatches(1));
      let afterBatch = transformed;
      for (let attempt = 0; attempt < 120 && afterBatch.completedBatchRevision < batch.batchRevision; attempt++) {
        afterBatch = await nextFrameForRequest(page, afterBatch.frameId, target.requestId);
      }
      assertFrame(afterBatch, `${direction} order ${order} first batch`, target.requestId, batch.batchRevision);
      records.push({ direction, order, target, transformed, batch, afterBatch });
    }
  }

  const beforeLoss = await page.evaluate(() => window.__ZOOMER_DIAGNOSTICS__());
  await page.evaluate(() => window.__ZOOMER_FORCE_DEVICE_LOSS__());
  await page.waitForFunction(
    previousEpoch => {
      const diagnostics = window.__ZOOMER_DIAGNOSTICS__();
      return diagnostics.rendererEpoch > previousEpoch && !diagnostics.recovering;
    },
    beforeLoss.rendererEpoch,
    { timeout: 20_000 }
  );
  const recoveredTarget = await page.evaluate(
    camera => window.__ZOOMER_TEST__.setExactCamera(camera, 'settled'),
    cameraAtOrder(fixture.deepOrder)
  );
  const recoveredFrame = await page.evaluate(
    ({ viewRevision, requestId }) => window.__ZOOMER_TEST__.waitForResolved(viewRevision, requestId, 0),
    recoveredTarget
  );
  assertFrame(recoveredFrame, 'device-loss recovered frame', recoveredTarget.requestId);
  const afterLoss = await page.evaluate(() => window.__ZOOMER_DIAGNOSTICS__());
  records.push({
    direction: 'device-loss-recovery',
    target: recoveredTarget,
    transformed: recoveredFrame,
    rendererEpochBefore: beforeLoss.rendererEpoch,
    rendererEpochAfter: afterLoss.rendererEpoch
  });
  await page.screenshot({
    path: resolve(outputDirectory, 'device-loss-recovered.png'), type: 'png'
  });

  const finalDiagnostics = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics());
  if (
    /swiftshader|software|llvmpipe/i.test(finalDiagnostics.adapterLabel)
    && process.env.ALLOW_SOFTWARE_GPU !== '1'
  ) {
    throw new Error(`Continuity gate requires a hardware GPU; received ${finalDiagnostics.adapterLabel}`);
  }
  if (finalDiagnostics.presentation.validationErrors !== 0) {
    throw new Error(`${finalDiagnostics.presentation.validationErrors} WebGPU validation errors`);
  }
  if (consoleErrors.length) throw new Error(`Browser errors:\n${consoleErrors.join('\n')}`);
  writeFileSync(resolve(outputDirectory, 'diagnostics.json'), JSON.stringify({
    fixture, executablePath, records, finalDiagnostics, consoleErrors
  }, null, 2));
  await context.tracing.stop({ path: resolve(outputDirectory, 'playwright-trace.zip') });
  console.log(`Continuity browser gate passed with ${records.length} transformed/batch pairs.`);
} finally {
  if (browser) await browser.close();
  server.kill();
}
