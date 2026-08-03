import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const fixturePath = resolve(root, process.env.PRECISION_FIXTURE ?? 'tests/fixtures/precision-user-failure-2026-08-03.json');
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
const outputDirectory = resolve(root, process.env.PRECISION_OUTPUT ?? 'test-results/precision-overlap-canary');
mkdirSync(outputDirectory, { recursive: true });
const port = Number(process.env.PRECISION_PORT ?? 4179);
const expectReferenceDrain = process.env.PRECISION_EXPECT_DRAIN !== '0';
const stressMs = Math.max(0, Number(process.env.PRECISION_STRESS_MS ?? 0));
const referenceTransport = process.env.PRECISION_REFERENCE_TRANSPORT;
const transportQuery = referenceTransport ? `&referenceTransport=${referenceTransport}` : '';
const url = `http://127.0.0.1:${port}/mandelbrot-zoomer/?testIterations=${fixture.iterationTarget}&continuityTest=1${transportQuery}`;
const viteEntry = resolve(root, 'node_modules/vite/bin/vite.js');
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
      if ((await fetch(url)).ok) return;
    } catch {}
    await new Promise(resolvePromise => setTimeout(resolvePromise, 100));
  }
  throw new Error(`Precision preview did not start.\n${serverOutput}`);
}

const camera = {
  centerXRaw: fixture.camera.centerX.raw,
  centerXBits: fixture.camera.centerX.bits,
  centerYRaw: fixture.camera.centerY.raw,
  centerYBits: fixture.camera.centerY.bits,
  scaleMantissa: fixture.camera.scale.mantissa,
  scaleExponent: fixture.camera.scale.exponent
};

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath,
    headless: true,
    args: ['--enable-unsafe-webgpu', '--disable-background-timer-throttling']
  });
  const page = await browser.newPage({
    viewport: {
      width: fixture.viewport?.cssWidth ?? 1200,
      height: fixture.viewport?.cssHeight ?? 800
    },
    deviceScaleFactor: fixture.viewport?.devicePixelRatio ?? 1
  });
  const errors = [];
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) {
      errors.push(message.text());
    }
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => Boolean(window.__ZOOMER_TEST__), null, { timeout: 20_000 });

  // Supersede several queued demands, then require active reference work to
  // survive and activate perturbation while the final request is still moving.
  for (let exponent = -11; exponent >= -14; exponent--) {
    await page.evaluate(
      ({ exactCamera, scaleExponent }) => window.__ZOOMER_TEST__.setExactCamera(
        { ...exactCamera, scaleExponent }, 'moving'
      ),
      { exactCamera: camera, scaleExponent: exponent }
    );
  }
  const movingTarget = await page.evaluate(
    exactCamera => window.__ZOOMER_TEST__.setExactCamera(exactCamera, 'moving'),
    camera
  );

  const movingDeadline = Date.now() + 60_000;
  let movingDiagnostics;
  while (Date.now() < movingDeadline) {
    movingDiagnostics = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics());
    if (movingDiagnostics.requestId === movingTarget.requestId
      && movingDiagnostics.field.requestId === movingTarget.requestId
      && movingDiagnostics.field.interaction === 'moving'
      && movingDiagnostics.field.finestTiles > 0
      && movingDiagnostics.field.finestPerturbationTiles > 0
      && movingDiagnostics.field.finestPerturbationTiles
        >= Math.ceil(movingDiagnostics.field.finestTiles * 0.9)
      && movingDiagnostics.field.submittedChunks > 0) break;
    await page.waitForTimeout(100);
  }
  await page.screenshot({ path: resolve(outputDirectory, 'precision-overlap-moving.png'), type: 'png' });

  const settledTarget = await page.evaluate(
    exactCamera => window.__ZOOMER_TEST__.setExactCamera(exactCamera, 'settled'),
    camera
  );
  const settledDeadline = Date.now() + 60_000;
  let diagnostics;
  const settledTrace = [];
  while (Date.now() < settledDeadline) {
    diagnostics = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics());
    settledTrace.push({ phase: 'settling', elapsedMs: 60_000 - (settledDeadline - Date.now()), ...diagnostics });
    if (diagnostics.requestId === settledTarget.requestId
      && diagnostics.field.requestId === settledTarget.requestId
      && diagnostics.field.perturbationTiles > 0
      && (!expectReferenceDrain || diagnostics.field.pendingReferences === 0)
      && diagnostics.field.submittedChunks > 0) break;
    await page.waitForTimeout(100);
  }
  if (stressMs > 0) {
    const stressStarted = Date.now();
    while (Date.now() - stressStarted < stressMs) {
      diagnostics = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics());
      settledTrace.push({ phase: 'stress', elapsedMs: Date.now() - stressStarted, ...diagnostics });
      await page.waitForTimeout(Math.min(250, Math.max(0, stressMs - (Date.now() - stressStarted))));
    }
    diagnostics = await page.evaluate(() => window.__ZOOMER_TEST__.diagnostics());
    settledTrace.push({ phase: 'stress', elapsedMs: Date.now() - stressStarted, ...diagnostics });
  }
  await page.screenshot({ path: resolve(outputDirectory, 'precision-overlap-settled.png'), type: 'png' });
  const report = {
    capturedAt: new Date().toISOString(), executablePath, fixture,
    movingTarget, movingDiagnostics, settledTarget, diagnostics, settledTrace, errors
  };
  writeFileSync(resolve(outputDirectory, 'report.json'), JSON.stringify(report, null, 2));

  const failures = [...errors];
  const movingField = movingDiagnostics?.field;
  const field = diagnostics?.field;
  const presentation = diagnostics?.presentation;
  if (movingDiagnostics?.requestId !== movingTarget.requestId
    || movingField?.requestId !== movingTarget.requestId) {
    failures.push(`Moving diagnostics belonged to request ${movingDiagnostics?.requestId}/${movingField?.requestId}, expected ${movingTarget.requestId}.`);
  }
  if (movingField?.interaction !== 'moving' || movingField?.perturbationTiles <= 0) {
    failures.push('No perturbation tile activated during continuous movement.');
  }
  if ((movingField?.finestPerturbationTiles ?? 0) < Math.ceil((movingField?.finestTiles ?? 1) * 0.9)) {
    failures.push(
      `Moving finest-level perturbation coverage was ${movingField?.finestPerturbationTiles ?? 0}/${movingField?.finestTiles ?? 0}; expected at least 90%.`
    );
  }
  if (!field || field.perturbationTiles <= 0) failures.push('No focus tile activated perturbation in the overlap band.');
  if (diagnostics?.requestId !== settledTarget.requestId
    || field?.requestId !== settledTarget.requestId) {
    failures.push(`Settled diagnostics belonged to request ${diagnostics?.requestId}/${field?.requestId}, expected ${settledTarget.requestId}.`);
  }
  if (expectReferenceDrain && field?.pendingReferences !== 0) {
    failures.push('Reference demand did not drain after settling.');
  }
  if (field?.referenceFailures !== 0) failures.push(`Reference failures: ${field.referenceFailures}.`);
  const expectedTransportBits = Number(
    process.env.PRECISION_EXPECT_TRANSPORT
      ?? fixture.expectedTransportBits
      ?? (referenceTransport === '96' ? 96 : 144)
  );
  if (field?.referenceTransportBits !== expectedTransportBits) {
    failures.push(`Expected declared ${expectedTransportBits}-bit transport, got ${field?.referenceTransportBits}.`);
  }
  if ((field?.referenceWorkingBits ?? 0) < 224) failures.push(`Expected at least 224 working bits, got ${field?.referenceWorkingBits}.`);
  if (field?.precisionLimitedTiles !== 0) failures.push('Overlap canary was incorrectly precision-limited.');
  if (presentation?.validationErrors !== 0) failures.push('WebGPU validation errors were reported.');
  const firstAdmittedIndex = settledTrace.findIndex(sample => (sample.field?.finestTiles ?? 0) > 0);
  if (firstAdmittedIndex >= 0 && settledTrace.slice(firstAdmittedIndex).some(
    sample => sample.field?.finestTiles === 0
  )) {
    failures.push('Finest spatial level disappeared during reference activation.');
  }
  if (stressMs >= 5_000) {
    const quietWindow = settledTrace.filter(
      sample => sample.phase === 'stress' && sample.elapsedMs >= stressMs - 5_000
    );
    if (quietWindow.length === 0 || quietWindow.some(sample => sample.busy
      || sample.field?.pendingReferences !== 0
      || sample.field?.queuedChunks !== 0
      || sample.field?.inFlightBatches !== 0)) {
      failures.push('Renderer did not remain continuously quiet for the final five seconds.');
    }
    if (quietWindow.some(sample => sample.requestId !== settledTarget.requestId
      || sample.field?.requestId !== settledTarget.requestId)) {
      failures.push('The settled request identity changed during the final quiet window.');
    }
  }
  if (/swiftshader|software|llvmpipe/i.test(diagnostics?.adapterLabel ?? '')) {
    failures.push(`Hardware GPU required, got ${diagnostics?.adapterLabel}.`);
  }
  if (failures.length) throw new Error(failures.join('\n'));
  console.log(
    `Precision overlap gate passed on ${diagnostics.adapterLabel}: `
    + `${field.perturbationTiles} perturbation tiles, ${field.referenceTransportBits}/${field.referenceWorkingBits} transport/working bits.`
  );
} finally {
  if (browser) await browser.close();
  server.kill();
}
