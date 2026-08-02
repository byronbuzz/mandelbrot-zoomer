import '../style.css';
import { BUILD_LABEL } from './build';
import { CameraModel } from '../v4/camera';
import { createUi, iterationCount, resetControlValues } from '../v4/ui';
import { ProgressiveRenderer } from './progressiveRenderer';

const app = document.querySelector<HTMLElement>('#app');
if (!app) throw new Error('Missing app root');

const ui = createUi(app);
const camera = new CameraModel();
const renderer = await ProgressiveRenderer.create(ui.canvas);

const MOVING_ANCHOR_INTERVAL_MS = 90;
const PAN_ANCHOR_INTERVAL_MS = 75;
const ZOOM_EASE_SECONDS = 0.12;
const VELOCITY_EPSILON = 0.002;
const WHEEL_SETTLE_MS = 150;

let anchorGeneration = 0;
let lastAnchorAt = -Infinity;
let targetDirection = 0;
let zoomVelocity = 0;
let pointerX = 0.5;
let pointerY = 0.5;
let panning = false;
let panPointer = -1;
let panX = 0;
let panY = 0;
let pumping = false;
let lastTick = performance.now();
let wheelTimer = 0;
let copyFeedbackTimer = 0;
let lastReadoutUpdate = 0;
let wasMoving = false;
const presentationTimestamps: number[] = [];

function requestedIterations(): number {
  return iterationCount(ui.iterations.value);
}

function canvasAspect(): number {
  return Math.max(1, ui.canvas.clientWidth) / Math.max(1, ui.canvas.clientHeight);
}

function renderDevicePixelRatio(): number {
  return Math.min(2, Math.max(1, devicePixelRatio));
}

function motionPressure(): number {
  const selectedSpeed = Math.max(0.25, Number(ui.speed.value));
  return Math.min(1, Math.abs(zoomVelocity) / selectedSpeed + (panning ? 0.5 : 0));
}

function startAnchor(pressure: number): void {
  const cssWidth = Math.max(1, ui.canvas.clientWidth);
  const cssHeight = Math.max(1, ui.canvas.clientHeight);
  anchorGeneration++;
  lastAnchorAt = performance.now();
  renderer.startAnchor({
    generation: anchorGeneration,
    camera: camera.snapshot(),
    cssWidth,
    cssHeight,
    devicePixelRatio: renderDevicePixelRatio(),
    iterations: requestedIterations(),
    palettePhase: Number(ui.palette.value),
    focusX: pointerX,
    focusY: pointerY,
    motionPressure: pressure
  });
  ensurePump();
}

function ensurePump(): void {
  if (pumping) return;
  pumping = true;
  void (async () => {
    try {
      while (renderer.hasWork) {
        await renderer.step();
      }
    } catch (error) {
      ui.status.textContent = error instanceof Error ? error.message : String(error);
      console.error(error);
    } finally {
      pumping = false;
      if (renderer.hasWork) ensurePump();
    }
  })();
}

function notePresentation(now: number): void {
  presentationTimestamps.push(now);
  const cutoff = now - 1000;
  while (presentationTimestamps.length > 0 && (presentationTimestamps[0] ?? now) < cutoff) {
    presentationTimestamps.shift();
  }
}

function displayRate(now = performance.now()): number {
  const cutoff = now - 1000;
  while (presentationTimestamps.length > 0 && (presentationTimestamps[0] ?? now) < cutoff) {
    presentationTimestamps.shift();
  }
  return presentationTimestamps.length;
}

function updatePointer(event: { clientX: number; clientY: number }): void {
  const bounds = ui.canvas.getBoundingClientRect();
  pointerX = Math.min(1, Math.max(0, (event.clientX - bounds.left) / Math.max(1, bounds.width)));
  pointerY = Math.min(1, Math.max(0, (event.clientY - bounds.top) / Math.max(1, bounds.height)));
}

function updateReadouts(): void {
  const stats = renderer.stats;
  const zoomLabel = `10^${camera.log10Magnification().toFixed(2)}`;
  const moving = Math.abs(zoomVelocity) > VELOCITY_EPSILON || panning;
  const blockLabel = stats.lastBlockSize > 0 ? `${stats.lastBlockSize}×${stats.lastBlockSize}` : 'waiting';

  ui.zoomOut.value = zoomLabel;
  ui.hudZoomOut.value = zoomLabel;
  ui.precisionOut.value = renderer.precisionLabel;
  ui.orbitOut.value = 'reference atlas arrives in V6.1';
  ui.healthOut.value = `${stats.completedJobs}/${stats.totalJobs} progressive tile jobs · ${stats.phase}`;
  ui.failureOut.value = 'iteration-cap pixels remain provisional; analytic cardioid + period-2 bulb enabled';
  ui.maxExponentOut.value = 'direct-render foundation';
  ui.bitsOut.value = String(camera.coordinateBits);
  ui.exponentOut.value = String(camera.snapshot().scale.exponent);
  ui.stateOut.value = moving ? `motion pressure ${(motionPressure() * 100).toFixed(0)}%` : stats.phase;
  ui.displayFpsOut.value = `${displayRate()} Hz`;
  ui.fpsOut.value = `${stats.tileRate} tiles/s`;
  ui.qualityOut.value = `${blockLabel} blocks · ${requestedIterations().toLocaleString()} iter target`;
  ui.timingOut.value = `${stats.lastTileMs.toFixed(1)} ms tile`;
  ui.batchesOut.value = `${stats.completedJobs}/${stats.totalJobs}`;
  ui.anchorMoveOut.value = `generation ${stats.anchorGeneration}`;
  ui.referenceTimeOut.value = 'not active in V6.0';
  ui.adapterOut.value = renderer.adapterLabel;
  ui.buildOut.value = BUILD_LABEL;
  ui.speedOut.value = `${Number(ui.speed.value).toFixed(2)}×/s`;
  ui.iterOut.value = requestedIterations().toLocaleString();
  ui.palOut.value = Number(ui.palette.value).toFixed(2);
  ui.hudFpsOut.value = `${displayRate()} Hz · ${stats.tileRate} tiles/s`;
  ui.setStatsWarning(false);
}

function diagnosticsReport(): string {
  const snapshot = camera.snapshot();
  const stats = renderer.stats;
  return [
    `Mandelbrot Zoomer ${BUILD_LABEL}`,
    `Engine: V6 progressive foundation`,
    `Captured: ${new Date().toISOString()}`,
    `GPU: ${renderer.adapterLabel}`,
    `URL: ${location.href}`,
    `Zoom depth: 10^${camera.log10Magnification().toFixed(6)}`,
    `Center X raw: ${snapshot.centerX.raw.toString()} (bits ${snapshot.centerX.bits})`,
    `Center Y raw: ${snapshot.centerY.raw.toString()} (bits ${snapshot.centerY.bits})`,
    `Scale: ${snapshot.scale.mantissa} * 2^${snapshot.scale.exponent}`,
    `Precision: ${renderer.precisionLabel}`,
    `Iterations target: ${requestedIterations()}`,
    `Progressive phase: ${stats.phase}`,
    `Tile progress: ${stats.completedJobs}/${stats.totalJobs}`,
    `Last block size: ${stats.lastBlockSize}`,
    `Last tile time: ${stats.lastTileMs.toFixed(3)} ms`,
    `Tile publication rate: ${stats.tileRate}/s`,
    `Display presentation rate: ${displayRate()} Hz`,
    `Motion pressure: ${motionPressure().toFixed(4)}`,
    `Analytic interior tests: ${stats.analyticInteriorEnabled}`
  ].join('\n');
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.append(textarea);
  textarea.select();
  document.execCommand('copy');
  textarea.remove();
}

function showCopyFeedback(message: string): void {
  if (copyFeedbackTimer) window.clearTimeout(copyFeedbackTimer);
  ui.copyFeedback.textContent = message;
  copyFeedbackTimer = window.setTimeout(() => { ui.copyFeedback.textContent = ''; }, 2200);
}

ui.canvas.addEventListener('pointermove', event => {
  updatePointer(event);
  if (!panning || event.pointerId !== panPointer) return;
  const bounds = ui.canvas.getBoundingClientRect();
  camera.panByCssPixels(event.clientX - panX, event.clientY - panY, bounds.height);
  panX = event.clientX;
  panY = event.clientY;
  const now = performance.now();
  if (now - lastAnchorAt >= PAN_ANCHOR_INTERVAL_MS) startAnchor(1);
});

ui.canvas.addEventListener('pointerdown', event => {
  ui.canvas.focus({ preventScroll: true });
  updatePointer(event);
  ui.canvas.setPointerCapture(event.pointerId);
  if (event.button === 1) {
    panning = true;
    panPointer = event.pointerId;
    panX = event.clientX;
    panY = event.clientY;
    targetDirection = 0;
  } else {
    targetDirection = event.button === 2 ? -1 : 1;
  }
  event.preventDefault();
});

function finishPointer(event: PointerEvent): void {
  if (event.pointerId === panPointer) {
    panning = false;
    panPointer = -1;
    startAnchor(0);
  }
  targetDirection = 0;
  if (ui.canvas.hasPointerCapture(event.pointerId)) ui.canvas.releasePointerCapture(event.pointerId);
}

ui.canvas.addEventListener('pointerup', finishPointer);
ui.canvas.addEventListener('pointercancel', finishPointer);
ui.canvas.addEventListener('contextmenu', event => event.preventDefault());
ui.canvas.addEventListener('auxclick', event => event.preventDefault());
ui.canvas.addEventListener('wheel', event => {
  updatePointer(event);
  camera.zoomAbout(pointerX, pointerY, Math.exp(event.deltaY * 0.0012), canvasAspect());
  startAnchor(0.5);
  if (wheelTimer) window.clearTimeout(wheelTimer);
  wheelTimer = window.setTimeout(() => startAnchor(0), WHEEL_SETTLE_MS);
  event.preventDefault();
}, { passive: false });

ui.speed.addEventListener('input', updateReadouts);
ui.iterations.addEventListener('input', () => startAnchor(0));
ui.palette.addEventListener('input', () => {
  renderer.recolour(Number(ui.palette.value));
  updateReadouts();
});
ui.resetControls.addEventListener('click', () => {
  resetControlValues(ui);
  startAnchor(0);
});
ui.resetLocation.addEventListener('click', () => {
  targetDirection = 0;
  zoomVelocity = 0;
  panning = false;
  panPointer = -1;
  pointerX = 0.5;
  pointerY = 0.5;
  camera.reset();
  startAnchor(0);
});
ui.copyDiagnostics.addEventListener('click', () => {
  void copyText(diagnosticsReport())
    .then(() => showCopyFeedback('Copied'))
    .catch(error => {
      console.error(error);
      showCopyFeedback('Copy failed');
    });
});

new ResizeObserver(() => startAnchor(0)).observe(ui.canvas);

document.addEventListener('keydown', event => {
  if (event.key === 'Tab' && !event.ctrlKey && !event.metaKey && !event.altKey) {
    const active = document.activeElement;
    const focusInsidePanel = active instanceof Node && ui.panel.contains(active);
    if (!ui.panelVisible() || !focusInsidePanel) {
      event.preventDefault();
      ui.togglePanel();
      if (ui.panelVisible()) ui.focusSelectedTab();
      else ui.canvas.focus({ preventScroll: true });
    }
    return;
  }
  if (event.key === '?' && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault();
    ui.setPanelVisible(true);
    ui.showTab('help');
    ui.focusSelectedTab();
    return;
  }
  if (event.key === 'Escape' && ui.panelVisible()) {
    event.preventDefault();
    ui.setPanelVisible(false);
    ui.canvas.focus({ preventScroll: true });
  }
});

renderer.onDeviceError(message => { ui.status.textContent = message; });
ui.status.textContent = `${renderer.adapterLabel} · ${BUILD_LABEL} · progressive direct engine`;
const anchorRateLabel = ui.fpsOut.previousElementSibling;
if (anchorRateLabel) anchorRateLabel.textContent = 'Tile publication rate';
const helpNote = ui.panel.querySelector<HTMLElement>('.help-panel .panel-note');
if (helpNote) helpNote.textContent = 'V6 uses eased zoom motion and continuous progressive tile refinement.';

function tick(now: number): void {
  const deltaSeconds = Math.min(0.05, Math.max(0, (now - lastTick) / 1000));
  lastTick = now;
  const selectedSpeed = Number(ui.speed.value);
  const targetVelocity = targetDirection * selectedSpeed;
  const easing = 1 - Math.exp(-deltaSeconds / ZOOM_EASE_SECONDS);
  zoomVelocity += (targetVelocity - zoomVelocity) * easing;
  if (targetDirection === 0 && Math.abs(zoomVelocity) < VELOCITY_EPSILON) zoomVelocity = 0;

  const moving = Math.abs(zoomVelocity) > VELOCITY_EPSILON || panning;
  if (Math.abs(zoomVelocity) > VELOCITY_EPSILON) {
    camera.zoomAbout(
      pointerX,
      pointerY,
      Math.exp(-zoomVelocity * deltaSeconds),
      canvasAspect()
    );
    if (now - lastAnchorAt >= MOVING_ANCHOR_INTERVAL_MS) startAnchor(motionPressure());
  }
  if (wasMoving && !moving) startAnchor(0);
  wasMoving = moving;

  if (renderer.present(
    camera.snapshot(),
    Math.max(1, ui.canvas.clientWidth),
    Math.max(1, ui.canvas.clientHeight),
    renderDevicePixelRatio()
  )) {
    notePresentation(now);
  }

  if (now - lastReadoutUpdate >= 250) {
    lastReadoutUpdate = now;
    updateReadouts();
  }
  requestAnimationFrame(tick);
}

startAnchor(0);
updateReadouts();
requestAnimationFrame(tick);
