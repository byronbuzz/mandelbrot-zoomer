import '../app/style.css';
import { BUILD_LABEL } from '../app/build';
import type { PresentationView } from './geometry';
import { Phase1PresentationKernel } from './phase1Kernel';

const root = document.querySelector<HTMLElement>('#app');
if (!root) throw new Error('Missing app root');
root.innerHTML = `
  <main class="app-shell phase1-shell">
    <canvas id="phase1-canvas" aria-label="Phase 1 presentation kernel"></canvas>
    <header class="top-bar">
      <div class="brand"><strong>Phase 1 Presentation Kernel</strong><span>${BUILD_LABEL}</span></div>
      <span class="phase1-badge">no fractal mathematics</span>
    </header>
    <aside class="phase1-panel" aria-label="Presentation kernel diagnostics">
      <strong>Executable gate</strong>
      <p>Persistent A/B history, affine reprojection and a single instanced tile overlay.</p>
      <div class="button-row">
        <button id="phase1-zoom">Navigation trace</button>
        <button id="phase1-suspend">Suspend / resume</button>
        <button id="phase1-loss">Force device loss</button>
      </div>
      <dl class="phase1-stats">
        <dt>State</dt><dd data-testid="phase1-state">initializing</dd>
        <dt>GPU errors</dt><dd data-testid="phase1-errors">0</dd>
        <dt>Resource epoch</dt><dd data-testid="phase1-resource-epoch">0</dd>
        <dt>Device epoch</dt><dd data-testid="phase1-device-epoch">1</dd>
        <dt>History</dt><dd data-testid="phase1-history">0</dd>
        <dt>Tile overlay</dt><dd data-testid="phase1-tiles">0 / 48</dd>
        <dt>Promotions</dt><dd data-testid="phase1-promotions">0</dd>
        <dt>Frame CPU p95</dt><dd data-testid="phase1-p95">0 ms</dd>
        <dt>Physical size</dt><dd data-testid="phase1-size">0 × 0</dd>
      </dl>
      <p id="phase1-status" role="status">Starting WebGPU…</p>
    </aside>
  </main>`;

const canvas = document.querySelector<HTMLCanvasElement>('#phase1-canvas');
if (!canvas) throw new Error('Missing presentation canvas');
const canvasElement = canvas;

let view: PresentationView = {
  centerX: -0.18,
  centerY: 0.11,
  height: 2.4,
  aspect: 1
};
const kernel = await Phase1PresentationKernel.create(canvasElement, view);
const status = document.querySelector<HTMLElement>('#phase1-status');
kernel.onError(message => { if (status) status.textContent = message; });

let traceStartedAt = 0;
let traceActive = false;
let suspendUntil = 0;
let animationFrame = 0;

function setText(testId: string, value: string): void {
  const element = document.querySelector<HTMLElement>(`[data-testid="${testId}"]`);
  if (element) element.textContent = value;
}

function updateDiagnostics(): void {
  const diagnostics = kernel.diagnostics;
  setText('phase1-state', diagnostics.state);
  setText('phase1-errors', String(diagnostics.validationErrors));
  setText('phase1-resource-epoch', String(diagnostics.resourceEpoch));
  setText('phase1-device-epoch', String(diagnostics.deviceEpoch));
  setText('phase1-history', `${diagnostics.historyFrames} reprojected / ${diagnostics.fallbackFrames} fallback`);
  setText('phase1-tiles', `${diagnostics.acceptedTiles} / ${diagnostics.tileCount}`);
  setText('phase1-promotions', String(diagnostics.anchorPromotions));
  setText('phase1-p95', `${diagnostics.frameCpuP95Ms.toFixed(2)} ms`);
  setText('phase1-size', `${diagnostics.width} × ${diagnostics.height}`);
  if (status && diagnostics.state === 'ready') {
    status.textContent = diagnostics.validationErrors === 0
      ? 'PASS candidate · zero WebGPU errors'
      : 'FAIL · WebGPU errors captured';
  }
}

function tick(now: number): void {
  animationFrame = requestAnimationFrame(tick);
  try {
    if (traceActive) {
      const elapsed = (now - traceStartedAt) / 1000;
      if (elapsed >= 4) {
        traceActive = false;
      } else {
        const phase = elapsed / 4;
        view = {
          centerX: -0.18 + 0.24 * Math.sin(phase * Math.PI * 2),
          centerY: 0.11 + 0.15 * Math.sin(phase * Math.PI * 4),
          height: 2.4 * Math.exp(-0.9 * Math.sin(phase * Math.PI)),
          aspect: Math.max(1, canvasElement.clientWidth) / Math.max(1, canvasElement.clientHeight)
        };
        kernel.setView(view);
      }
    } else {
      const nextAspect = Math.max(1, canvasElement.clientWidth) / Math.max(1, canvasElement.clientHeight);
      if (nextAspect !== view.aspect) {
        view = { ...view, aspect: nextAspect };
        kernel.setView(view);
      }
    }
    if (now < suspendUntil) kernel.render(0, 0, Math.min(2, devicePixelRatio));
    else kernel.render(canvasElement.clientWidth, canvasElement.clientHeight, Math.min(2, devicePixelRatio));
  } finally {
    updateDiagnostics();
  }
}

document.querySelector<HTMLButtonElement>('#phase1-zoom')?.addEventListener('click', () => {
  traceStartedAt = performance.now();
  traceActive = true;
});
document.querySelector<HTMLButtonElement>('#phase1-loss')?.addEventListener('click', () => {
  kernel.forceDeviceLossForTest();
});
document.querySelector<HTMLButtonElement>('#phase1-suspend')?.addEventListener('click', () => {
  suspendUntil = performance.now() + 350;
});

window.addEventListener('pagehide', () => {
  cancelAnimationFrame(animationFrame);
  kernel.destroy();
}, { once: true });

if (status) status.textContent = 'WebGPU ready';
animationFrame = requestAnimationFrame(tick);
