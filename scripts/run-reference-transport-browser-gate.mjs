import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(new URL('..', import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const outputDirectory = resolve(root, process.env.REFERENCE_OUTPUT ?? 'test-results/reference-transport');
mkdirSync(outputDirectory, { recursive: true });
const port = Number(process.env.REFERENCE_PORT ?? 4188);
const url = `http://127.0.0.1:${port}/mandelbrot-zoomer/tests/reference-transport-fixture.html`;
const server = spawn(process.execPath, [
  resolve(root, 'node_modules/vite/bin/vite.js'),
  '--host', '127.0.0.1', '--port', String(port), '--strictPort', '--configLoader', 'runner'
], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
let serverOutput = '';
server.stdout.on('data', chunk => { serverOutput += String(chunk); });
server.stderr.on('data', chunk => { serverOutput += String(chunk); });

const candidates = [
  process.env.WEBGPU_BROWSER_PATH,
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
].filter(Boolean);
const executablePath = candidates.find(candidate => existsSync(candidate));
if (!executablePath) throw new Error('No Chrome or Edge executable found.');

let browser;
try {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try { if ((await fetch(url)).ok) break; } catch {}
    await new Promise(resolvePromise => setTimeout(resolvePromise, 100));
  }
  browser = await chromium.launch({ executablePath, headless: true });
  const page = await browser.newPage({ viewport: { width: 900, height: 600 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().startsWith('Failed to load resource:')) {
      errors.push(message.text());
    }
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => Boolean(window.__REFERENCE_TRANSPORT_ORACLE__), null, { timeout: 20_000 });
  const report = await page.evaluate(() => window.__REFERENCE_TRANSPORT_ORACLE__);
  await page.screenshot({ path: resolve(outputDirectory, 'reference-transport-fixture.png') });
  writeFileSync(resolve(outputDirectory, 'report.json'), JSON.stringify({ report, errors }, null, 2));
  const failures = [...errors];
  if (report.contractVersion !== 2) failures.push(`Expected contract version 2, got ${report.contractVersion}.`);
  if (report.legacy.transportBits !== 96 || report.legacy.floatsPerPoint !== 8) {
    failures.push(`Legacy layout was ${report.legacy.transportBits}/${report.legacy.floatsPerPoint}.`);
  }
  if (report.wide.transportBits !== 192 || report.wide.floatsPerPoint !== 16) {
    failures.push(`Wide layout was ${report.wide.transportBits}/${report.wide.floatsPerPoint}.`);
  }
  const first = report.comparisons[0];
  const legacyMinimum = Math.min(first.legacyXBits, first.legacyYBits);
  const wideMinimum = Math.min(first.wideXBits, first.wideYBits);
  if (legacyMinimum > 112) failures.push(`Legacy transport unexpectedly claimed ${legacyMinimum} bits.`);
  if (wideMinimum < 150) failures.push(`Wide transport retained only ${wideMinimum} bits.`);
  if (wideMinimum - legacyMinimum < 40) {
    failures.push(`Wide transport improved point 1 by only ${wideMinimum - legacyMinimum} bits.`);
  }
  if (failures.length) throw new Error(failures.join('\n'));
  console.log(`Reference transport oracle passed: ${legacyMinimum} -> ${wideMinimum} retained bits at point 1.`);
} finally {
  if (browser) await browser.close();
  server.kill();
}
