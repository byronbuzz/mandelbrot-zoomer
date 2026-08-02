import './style.css';
import { CameraModel } from '../camera/cameraModel';
import { FieldRenderer } from '../presentation/fieldRenderer';
import type { InteractionState } from '../tiles/types';
import { APP_NAME, APP_VERSION, BUILD_LABEL } from './build';
import { createUi } from './ui';

const root = document.querySelector<HTMLElement>('#app');
if (!root) throw new Error('Missing app root');

const ui = createUi(root);
const camera = new CameraModel();
const renderer = await FieldRenderer.create(ui.canvas);

const MOVING_REQUEST_INTERVAL_MS = 85;
const PAN_REQUEST_INTERVAL_MS = 70;
const SETTLING_DURATION_MS = 180;
const ZOOM_EASE_SECONDS = 0.12;
const VELOCITY_EPSILON = 0.002;

let requestId = 0;
let interaction: InteractionState = 'settled';
let targetDirection = 0;
let zoomVelocity = 0;
let pointerX = 0.5;
let pointerY = 0.5;
let panning = false;
let panPointer = -1;
let panX = 0;
let panY = 0;
let lastRequestAt = -Infinity;
let lastTick = performance.now();
let wasMoving = false;
let settleTimer = 0;
let wheelTimer = 0;
let copyFeedbackTimer = 0;
let lastReadoutAt = 0;
const presentationTimestamps: number[] = [];

function targetIterations(): number {
  return Math.max(1, Math.floor(Number(ui.iterations.value)));
}

function palettePhase(): number {
  return Number(ui.palette.value);
}

function renderDevicePixelRatio(): number {
  return Math.min(2, Math.max(1, devicePixelRatio));
}

function canvasAspect(): number {
  return Math.max(1, ui.canvas.clientWidth) / Math.max(1, ui.canvas.clientHeight);
}

function clearSettleTimers(): void {
  if (settleTimer) window.clearTimeout(settleTimer);
  if (wheelTimer) window.clearTimeout(wheelTimer);
  settleTimer = 0;
  wheelTimer = 0;
}

function requestField(nextInteraction: InteractionState): void {
  interaction = nextInteraction;
  requestId++;
  lastRequestAt = performance.now();
  renderer.request({
    requestId,
    camera: camera.snapshot(),
    cssWidth: Math.max(1, ui.canvas.clientWidth),
    cssHeight: Math.max(1, ui.canvas.clientHeight),
    devicePixelRatio: renderDevicePixelRatio(),
    targetIterations: targetIterations(),
    palettePhase: palettePhase(),
    focusX: pointerX,
    focusY: pointerY,
    interaction: nextInteraction
  });
}

function beginMotion(): void {
  clearSettleTimers();
  if (interaction !== 'moving') requestField('moving');
  else interaction = 'moving';
}

function beginSettling(): void {
  clearSettleTimers();
  requestField('settling');
  settleTimer = window.setTimeout(() => {
    settleTimer = 0;
    if (targetDirection === 0 && !panning && Math.abs(zoomVelocity) <= VELOCITY_EPSILON) {
      requestField('settled');
    }
  }, SETTLING_DURATION_MS);
}

function updatePointer(event: { clientX: number; clientY: number }): void {
  const bounds = ui.canvas.getBoundingClientRect();
  pointerX = Math.min(1, Math.max(0, (event.clientX - bounds.left) / Math.max(1, bounds.width)));
  pointerY = Math.min(1, Math.max(0, (event.clientY - bounds.top) / Math.max(1, bounds.height)));
}

function notePresentation(now: number): void {
  presentationTimestamps.push(now);
  trimPresentationTimes(now);
}

function trimPresentationTimes(now: number): void {
  const cutoff = now - 1000;
  while (presentationTimestamps.length > 0 && (presentationTimestamps[0] ?? now) < cutoff) {
    presentationTimestamps.shift();
  }
}

function displayRate(now = performance.now()): number {
  trimPresentationTimes(now);
  return presentationTimestamps.length;
}

function updateReadouts(): void {
  const stats = renderer.stats;
  const depth = camera.log10Magnification();
  const precision = stats.precision === 'f32-direct'
    ? 'native f32 direct'
    : depth >= 6
      ? 'double-float direct · perturbation stage pending'
      : 'double-float direct';
  ui.zoomOut.value = `10^${depth.toFixed(3)}`;
  ui.stateOut.value = `${interaction}${renderer.isBusy ? ' · calculating' : ''}`;
  ui.precisionOut.value = precision;
  ui.fieldOut.value = `${stats.phase} · ${(stats.fieldAgeMs / 1000).toFixed(1)} s old · anchor ${stats.anchorGeneration}`;
  ui.jobsOut.value = `${stats.completedJobs}/${stats.totalJobs} · ${stats.publishedJobs} published`;
  ui.timingOut.value = `${stats.lastBatchMs.toFixed(1)} ms · ${stats.batchJobs} jobs`;
  ui.displayOut.value = `${displayRate()} Hz`;
  ui.renderSizeOut.value = `${stats.renderWidth} × ${stats.renderHeight}`;
  ui.navigationOut.value = `${stats.navigationBlockSize}×${stats.navigationBlockSize} · ${stats.navigationIterations} iter · ${(stats.navigationResolution * 100).toFixed(0)}%`;
  ui.gpuOut.value = renderer.adapterLabel;
  ui.buildOut.value = BUILD_LABEL;
}

function diagnosticsReport(): string {
  const snapshot = camera.snapshot();
  const stats = renderer.stats;
  return [
    `${APP_NAME} ${APP_VERSION} · ${BUILD_LABEL.split(' · ')[1] ?? 'dev'}`,
    `Captured: ${new Date().toISOString()}`,
    `GPU: ${renderer.adapterLabel}`,
    `URL: ${location.href}`,
    `Zoom depth: 10^${camera.log10Magnification().toFixed(6)}`,
    `Center X raw: ${snapshot.centerX.raw.toString()} (bits ${snapshot.centerX.bits})`,
    `Center Y raw: ${snapshot.centerY.raw.toString()} (bits ${snapshot.centerY.bits})`,
    `Scale: ${snapshot.scale.mantissa} * 2^${snapshot.scale.exponent}`,
    `Interaction: ${interaction}`,
    `Precision: ${stats.precision}`,
    `Field phase: ${stats.phase}`,
    `Field anchor generation: ${stats.anchorGeneration}`,
    `Field age: ${stats.fieldAgeMs.toFixed(3)} ms`,
    `Render size: ${stats.renderWidth} x ${stats.renderHeight}`,
    `Jobs: ${stats.completedJobs}/${stats.totalJobs}`,
    `Published jobs: ${stats.publishedJobs}`,
    `Last batch: ${stats.lastBatchMs.toFixed(3)} ms (${stats.batchJobs} jobs)`,
    `Navigation profile: ${stats.navigationBlockSize}x${stats.navigationBlockSize}, ${stats.navigationIterations} iterations, ${stats.navigationResolution.toFixed(4)} resolution`,
    `Display presentation rate: ${displayRate()} Hz`,
    `Iteration target: ${targetIterations()}`,
    `Palette phase: ${palettePhase().toFixed(4)}`
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

ui.canvas.addEventListener('pointerdown', event => {
  ui.canvas.focus({ preventScroll: true });
  updatePointer(event);
  ui.canvas.setPointerCapture(event.pointerId);
  beginMotion();
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

ui.canvas.addEventListener('pointermove', event => {
  updatePointer(event);
  if (!panning || event.pointerId !== panPointer) return;
  const bounds = ui.canvas.getBoundingClientRect();
  camera.panByCssPixels(event.clientX - panX, event.clientY - panY, bounds.height);
  panX = event.clientX;
  panY = event.clientY;
  const now = performance.now();
  if (now - lastRequestAt >= PAN_REQUEST_INTERVAL_MS) requestField('moving');
});

function finishPointer(event: PointerEvent): void {
  if (event.pointerId === panPointer) {
    panning = false;
    panPointer = -1;
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
  beginMotion();
  camera.zoomAbout(pointerX, pointerY, Math.exp(event.deltaY * 0.0012), canvasAspect());
  requestField('moving');
  if (wheelTimer) window.clearTimeout(wheelTimer);
  wheelTimer = window.setTimeout(() => {
    wheelTimer = 0;
    beginSettling();
  }, 130);
  event.preventDefault();
}, { passive: false });

ui.speed.addEventListener('input', updateReadouts);
ui.iterations.addEventListener('input', () => requestField('settled'));
ui.palette.addEventListener('input', () => requestField('settled'));
ui.resetLocation.addEventListener('click', () => {
  clearSettleTimers();
  targetDirection = 0;
  zoomVelocity = 0;
  panning = false;
  panPointer = -1;
  pointerX = 0.5;
  pointerY = 0.5;
  camera.reset();
  requestField('settled');
});
ui.copyDiagnostics.addEventListener('click', () => {
  void copyText(diagnosticsReport())
    .then(() => showCopyFeedback('Copied'))
    .catch(error => {
      console.error(error);
      showCopyFeedback('Copy failed');
    });
});

new ResizeObserver(() => requestField('settled')).observe(ui.canvas);

document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearSettleTimers();
  else requestField('settled');
});

document.addEventListener('keydown', event => {
  if (event.key === 'Escape') ui.panel.classList.add('is-hidden');
  if (event.key === '?' && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault();
    ui.panel.classList.remove('is-hidden');
  }
});

renderer.onDeviceError(message => { ui.status.textContent = message; });
renderer.onRuntimeError(message => { ui.status.textContent = `Renderer error: ${message}`; });
ui.status.textContent = `${renderer.adapterLabel} · ${APP_NAME} ${BUILD_LABEL}`;

function tick(now: number): void {
  const deltaSeconds = Math.min(0.05, Math.max(0, (now - lastTick) / 1000));
  lastTick = now;
  const speed = Number(ui.speed.value);
  const targetVelocity = targetDirection * speed;
  const easing = 1 - Math.exp(-deltaSeconds / ZOOM_EASE_SECONDS);
  zoomVelocity += (targetVelocity - zoomVelocity) * easing;
  if (targetDirection === 0 && Math.abs(zoomVelocity) < VELOCITY_EPSILON) zoomVelocity = 0;

  const moving = panning || Math.abs(zoomVelocity) > VELOCITY_EPSILON;
  if (Math.abs(zoomVelocity) > VELOCITY_EPSILON) {
    camera.zoomAbout(pointerX, pointerY, Math.exp(-zoomVelocity * deltaSeconds), canvasAspect());
    if (now - lastRequestAt >= MOVING_REQUEST_INTERVAL_MS) requestField('moving');
  }
  if (wasMoving && !moving) beginSettling();
  wasMoving = moving;

  if (renderer.present(
    camera.snapshot(),
    Math.max(1, ui.canvas.clientWidth),
    Math.max(1, ui.canvas.clientHeight),
    renderDevicePixelRatio()
  )) {
    notePresentation(now);
  }

  if (now - lastReadoutAt >= 200) {
    lastReadoutAt = now;
    updateReadouts();
  }
  requestAnimationFrame(tick);
}

requestField('settled');
updateReadouts();
requestAnimationFrame(tick);
