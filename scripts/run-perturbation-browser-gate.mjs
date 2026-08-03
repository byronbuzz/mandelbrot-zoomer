import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const outputDirectory = resolve(root, process.env.PERTURBATION_OUTPUT ?? 'test-results/perturbation-oracle');
mkdirSync(outputDirectory, { recursive: true });
const port = Number(process.env.PERTURBATION_PORT ?? 4194);
const url = `http://127.0.0.1:${port}/mandelbrot-zoomer/tests/perturbation-oracle-fixture.html`;
const server = spawn(process.execPath, [
  resolve(root, 'node_modules/vite/bin/vite.js'),
  '--host', '127.0.0.1', '--port', String(port), '--strictPort', '--configLoader', 'runner'
], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
let browser;
try {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try { if ((await fetch(url)).ok) break; } catch {}
    await new Promise(resolvePromise => setTimeout(resolvePromise, 100));
  }
  const executablePath = [
    process.env.WEBGPU_BROWSER_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
  ].filter(Boolean).find(candidate => existsSync(candidate));
  if (!executablePath) throw new Error('No Chrome or Edge executable found.');
  browser = await chromium.launch({ executablePath, headless: true, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage({ viewport: { width: 900, height: 600 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) errors.push(message.text());
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => Boolean(window.__PERTURBATION_ORACLE__), null, { timeout: 20_000 });
  const report = await page.evaluate(() => window.__PERTURBATION_ORACLE__);
  await page.screenshot({ path: resolve(outputDirectory, 'perturbation-oracle.png') });
  writeFileSync(resolve(outputDirectory, 'report.json'), JSON.stringify({ report, errors }, null, 2));
  const failures = [...errors];
  if (report.compilationMessages.some(message => message.type === 'error')) failures.push('WGSL compilation error.');
  if (report.validationError) failures.push(report.validationError);
  failures.push(...report.uncaptured);
  for (const comparison of report.comparisons) {
    const exactSampleAvailable = comparison.expected.iteration < comparison.orbitLength;
    const exactMatch = comparison.gpu.status === comparison.expected.status
      && comparison.gpu.iteration === comparison.expected.iteration;
    const correctlyUnresolved = !exactSampleAvailable
      && comparison.gpu.status === 6
      && comparison.gpu.iteration === comparison.orbitLength;
    if (!exactMatch && !correctlyUnresolved) {
      failures.push(
        `${comparison.scenario} ${comparison.x},${comparison.y}: GPU ${comparison.gpu.status}/${comparison.gpu.iteration}`
        + ` != exact ${comparison.expected.status}/${comparison.expected.iteration}`
      );
    }
  }
  if (failures.length) throw new Error(failures.join('\n'));
  console.log(
    `Scaled perturbation oracle passed on ${report.adapter}: `
    + `${report.comparisons.length} exact pixels across periodic and user boundary scenarios.`
  );
} finally {
  if (browser) await browser.close();
  server.kill();
}
