import './style.css';
import { fixedDifferenceToNumber } from './bigFixed';
import { scaleToNumber } from './binaryScale';
import { CameraModel } from './v4/camera';
import { ReferenceManager } from './v4/referenceManager';
import { RenderCoordinator, type PresentedFrame } from './v4/renderCoordinator';
import type {
  CpuReference,
  GpuReference,
  PrecisionMode,
  RenderQuality,
  RenderSnapshot,
  RenderStage,
  RenderTelemetry
} from './v4/types';
import { createUi, iterationCount } from './v4/ui';
import { WebGpuRenderer } from './v4/webGpuRenderer';

const app = document.querySelector<HTMLElement>('#app');
if (!app) throw new Error('Missing app root');
const ui = createUi(app);
const camera = new CameraModel();
const renderer = await WebGpuRenderer.create(ui.canvas);
const references = new ReferenceManager();

const TARGET_FRAME_MS = 1000 / 60;
const SETTLE_MS = 180;
const REFINE_DELAY_MS = 100;
const MIN_RESOLUTION = 0.28;
const DOUBLE_FLOAT_THRESHOLD_LOG10 = 4;
const REFERENCE_PREFETCH_LOG10 = 4;
const PERTURBATION_THRESHOLD_LOG10 = 4.5;
const DOUBLE_FLOAT_INTERACTIVE_RESOLUTION = 0.52;
const REFERENCE_FALLBACK_INTERACTIVE_RESOLUTION = 0.42;
const PERTURBATION_INTERACTIVE_RESOLUTION = 0.72;

let presentationEpoch = 1;
let stage: RenderStage = 'full-quality';
let refineStep = 0;
let adaptiveResolution = 1;
let smoothedFrameMs = TARGET_FRAME_MS;
let lastCompletedFps = 0;
let direction = 0;
let pointerX = 0.5;
let pointerY = 0.5;
let panning = false;
let panPointer = -1;
let panX = 0;
let panY = 0;
let settleTimer = 0;
let refineTimer = 0;
let activeReference: GpuReference | null = null;
let lastSnapshot: RenderSnapshot | null = null;
let lastTelemetry: RenderTelemetry | null = null;
let deferredReference: CpuReference | null = null;
const retiredReferences: GpuReference[] = [];

function requestedIterations(): number { return iterationCount(ui.iterations.value); }
function aspect(): number { return Math.max(1, ui.canvas.clientWidth) / Math.max(1, ui.canvas.clientHeight); }
function isInteracting(): boolean { return stage === 'interactive'; }

function clearTimers(): void {
  if (settleTimer) window.clearTimeout(settleTimer);
  if (refineTimer) window.clearTimeout(refineTimer);
  settleTimer = 0;
  refineTimer = 0;
}

function invalidatePresentation(): void {
  presentationEpoch++;
  lastTelemetry = null;
}

function setStage(next: RenderStage, nextRefineStep = 0): void {
  if (stage === next && refineStep === nextRefineStep) return;
  stage = next;
  refineStep = nextRefineStep;
  invalidatePresentation();
}

function beginInteraction(autoSettle: boolean): void {
  clearTimers();
  setStage('interactive');
  if (autoSettle) settleTimer = window.setTimeout(() => finishInteraction(), SETTLE_MS);
}

function finishInteraction(): void {
  clearTimers();
  references.cancelOlderThan(camera.generation);
  setStage('refining', 1);
  tryActivateDeferredReference();
  requestCurrentRender();
}

function scheduleNextRefinement(): void {
  if (stage !== 'refining') return;
  if (refineTimer) window.clearTimeout(refineTimer);
  refineTimer = window.setTimeout(() => {
    if (stage !== 'refining') return;
    if (refineStep < 2) setStage('refining', refineStep + 1);
    else setStage('full-quality');
    requestCurrentRender();
  }, REFINE_DELAY_MS);
}

function effectiveQuality(): RenderQuality {
  const iterations = requestedIterations();
  if (stage === 'interactive') {
    const depth = camera.log10Magnification();
    let resolution = adaptiveResolution;
    if (depth >= PERTURBATION_THRESHOLD_LOG10 && !activeReference) {
      resolution = Math.min(resolution, REFERENCE_FALLBACK_INTERACTIVE_RESOLUTION);
    } else if (depth >= PERTURBATION_THRESHOLD_LOG10) {
      resolution = Math.min(resolution, PERTURBATION_INTERACTIVE_RESOLUTION);
    } else if (depth >= DOUBLE_FLOAT_THRESHOLD_LOG10) {
      resolution = Math.min(resolution, DOUBLE_FLOAT_INTERACTIVE_RESOLUTION);
    }
    return { iterations, resolution: Math.max(MIN_RESOLUTION, resolution) };
  }
  if (stage === 'refining') return { iterations, resolution: refineStep === 1 ? 0.65 : 0.85 };
  return { iterations, resolution: 1 };
}

function referenceOffsetInViewports(reference: GpuReference | CpuReference): number {
  const snapshot = camera.snapshot();
  const dx = fixedDifferenceToNumber(snapshot.centerX, reference.centerX);
  const dy = fixedDifferenceToNumber(snapshot.centerY, reference.centerY);
  return Math.hypot(dx, dy) / Math.max(scaleToNumber(snapshot.scale), Number.MIN_VALUE);
}

function referenceIsWeak(reference: GpuReference, iterations: number): boolean {
  const offset = referenceOffsetInViewports(reference);
  return !Number.isFinite(offset)
    || offset > 1.5
    || (reference.escaped && reference.length < Math.min(iterations, 4096));
}

function requestLatestProvisional(cameraSnapshot: ReturnType<CameraModel['snapshot']>, iterations: number): void {
  references.request(
    cameraSnapshot,
    Math.min(iterations, 4096),
    'provisional',
    aspect(),
    true
  );
}

function manageReferenceRequests(cameraSnapshot: ReturnType<CameraModel['snapshot']>, iterations: number): void {
  const depth = camera.log10Magnification();
  if (depth < REFERENCE_PREFETCH_LOG10) return;
  const weak = !activeReference || referenceIsWeak(activeReference, iterations);

  if (depth < PERTURBATION_THRESHOLD_LOG10) {
    if (weak) requestLatestProvisional(cameraSnapshot, iterations);
    return;
  }

  if (stage === 'interactive') {
    if (weak) requestLatestProvisional(cameraSnapshot, iterations);
    return;
  }
  const currentSettled = Boolean(
    activeReference
    && activeReference.cameraGeneration === cameraSnapshot.generation
    && activeReference.purpose === 'settled'
    && activeReference.requestedIterations >= iterations
  );
  if (!currentSettled && references.pendingCount === 0) {
    references.request(cameraSnapshot, iterations, 'settled', aspect());
  }
}

function precisionFor(reference: GpuReference | null): PrecisionMode {
  const depth = camera.log10Magnification();
  if (depth < DOUBLE_FLOAT_THRESHOLD_LOG10) return 'f32';
  if (depth < PERTURBATION_THRESHOLD_LOG10 || !reference) return 'double-float';
  return 'perturbation';
}

function buildSnapshot(): RenderSnapshot {
  const cameraSnapshot = camera.snapshot();
  const quality = effectiveQuality();
  manageReferenceRequests(cameraSnapshot, quality.iterations);
  return {
    generation: presentationEpoch,
    camera: cameraSnapshot,
    stage,
    quality,
    palettePhase: Number(ui.palette.value),
    cssWidth: Math.max(1, ui.canvas.clientWidth),
    cssHeight: Math.max(1, ui.canvas.clientHeight),
    devicePixelRatio,
    reference: activeReference,
    precision: precisionFor(activeReference)
  };
}

function requestCurrentRender(): void {
  const snapshot = buildSnapshot();
  lastSnapshot = snapshot;
  coordinator.request(snapshot);
  updateReadouts(snapshot);
}

function invalidateAndRender(): void {
  invalidatePresentation();
  requestCurrentRender();
}

function updateController(frameMs: number, renderedStage: RenderStage): void {
  smoothedFrameMs = smoothedFrameMs * 0.78 + frameMs * 0.22;
  lastCompletedFps = 1000 / smoothedFrameMs;
  if (renderedStage !== 'interactive') return;
  if (frameMs > TARGET_FRAME_MS * 1.08) adaptiveResolution = Math.max(MIN_RESOLUTION, adaptiveResolution * 0.82);
  else if (frameMs < TARGET_FRAME_MS * 0.78) adaptiveResolution = Math.min(1, adaptiveResolution * 1.06 + 0.01);
}

function onPresented(frame: PresentedFrame): void {
  updateController(frame.computeMs + frame.presentMs, frame.snapshot.stage);
  if (frame.telemetry) lastTelemetry = frame.telemetry;
  updateReadouts(frame.snapshot);
  if (frame.snapshot.stage === 'refining' && frame.snapshot.generation === presentationEpoch) scheduleNextRefinement();
}

function onCoordinatorIdle(): void {
  while (retiredReferences.length) renderer.destroyReference(retiredReferences.shift()!);
}

const coordinator = new RenderCoordinator(
  renderer,
  () => presentationEpoch,
  onPresented,
  error => {
    ui.status.textContent = error instanceof Error ? error.message : String(error);
    console.error(error);
  },
  onCoordinatorIdle
);

function candidateIsUseful(candidate: CpuReference): boolean {
  const distance = referenceOffsetInViewports(candidate);
  if (!Number.isFinite(distance) || distance > 0.9) return false;
  if (!activeReference) return true;
  const currentWeak = referenceIsWeak(activeReference, requestedIterations());
  const candidateStronger = !candidate.escaped || candidate.length > activeReference.length;
  if (currentWeak && candidateStronger) return true;
  if (candidate.purpose === 'settled' && activeReference.purpose === 'provisional') return true;
  return candidate.requestedIterations >= activeReference.requestedIterations && candidateStronger;
}

function activateReference(candidate: CpuReference): void {
  if (!candidateIsUseful(candidate)) return;
  const next = renderer.createReference(candidate);
  if (activeReference) retiredReferences.push(activeReference);
  activeReference = next;
  invalidatePresentation();
  requestCurrentRender();
}

function tryActivateDeferredReference(): void {
  if (!deferredReference) return;
  const candidate = deferredReference;
  deferredReference = null;
  activateReference(candidate);
}

references.onReference(candidate => {
  if (!candidateIsUseful(candidate)) return;
  const currentHealthy = activeReference && !referenceIsWeak(activeReference, requestedIterations());
  if (isInteracting() && currentHealthy) {
    if (!deferredReference || candidate.length > deferredReference.length) deferredReference = candidate;
    return;
  }
  activateReference(candidate);
});

function updateReadouts(snapshot: RenderSnapshot): void {
  const reference = snapshot.reference;
  const offset = reference ? referenceOffsetInViewports(reference) : 0;
  const ended = reference?.escaped ? ` · escaped at ${reference.length - 1}` : '';
  const pending = references.pendingCount ? ` · ${references.pendingCount} reference queued` : '';
  ui.zoomOut.value = `10^${camera.log10Magnification().toFixed(2)}`;
  ui.precisionOut.value = snapshot.precision;
  ui.orbitOut.value = reference
    ? `~${reference.bits}-bit ${reference.purpose} · ${reference.length - 1} stored · ${offset.toFixed(2)} view offset${ended}${pending}`
    : `inactive${pending}`;
  if (snapshot.precision !== 'perturbation') {
    ui.healthOut.value = 'inactive';
  } else if (!lastTelemetry) {
    ui.healthOut.value = snapshot.stage === 'full-quality' ? 'measuring…' : 'measured on full-quality frames';
  } else {
    const percentage = lastTelemetry.totalPixels > 0
      ? (lastTelemetry.unresolvedPixels / lastTelemetry.totalPixels) * 100
      : 0;
    ui.healthOut.value = `${lastTelemetry.unresolvedPixels.toLocaleString()} unresolved (${percentage.toFixed(3)}%) · ${lastTelemetry.exhaustedPixels.toLocaleString()} orbit-exhausted`;
  }
  ui.bitsOut.value = String(camera.coordinateBits);
  ui.exponentOut.value = String(snapshot.camera.scale.exponent);
  ui.stateOut.value = snapshot.stage;
  ui.fpsOut.value = `${lastCompletedFps.toFixed(1)} FPS`;
  ui.qualityOut.value = `${snapshot.quality.iterations} / ${requestedIterations()} iter · ${Math.round(snapshot.quality.resolution * 100)}%`;
  ui.speedOut.value = `${Number(ui.speed.value).toFixed(2)}×/s`;
  ui.iterOut.value = requestedIterations().toLocaleString();
  ui.palOut.value = Number(ui.palette.value).toFixed(2);
}

function updatePointer(event: { clientX: number; clientY: number }): void {
  const bounds = ui.canvas.getBoundingClientRect();
  pointerX = (event.clientX - bounds.left) / Math.max(1, bounds.width);
  pointerY = (event.clientY - bounds.top) / Math.max(1, bounds.height);
}

ui.canvas.addEventListener('pointermove', event => {
  updatePointer(event);
  if (!panning || event.pointerId !== panPointer) return;
  const bounds = ui.canvas.getBoundingClientRect();
  camera.panByCssPixels(event.clientX - panX, event.clientY - panY, bounds.height);
  panX = event.clientX;
  panY = event.clientY;
  requestCurrentRender();
});

ui.canvas.addEventListener('pointerdown', event => {
  updatePointer(event);
  ui.canvas.setPointerCapture(event.pointerId);
  if (event.button === 1) {
    panning = true;
    panPointer = event.pointerId;
    panX = event.clientX;
    panY = event.clientY;
    direction = 0;
  } else {
    direction = event.button === 2 ? -1 : 1;
  }
  beginInteraction(false);
  requestCurrentRender();
  event.preventDefault();
});

function endPointer(event: PointerEvent): void {
  if (event.pointerId === panPointer) {
    panning = false;
    panPointer = -1;
  }
  direction = 0;
  finishInteraction();
  if (ui.canvas.hasPointerCapture(event.pointerId)) ui.canvas.releasePointerCapture(event.pointerId);
}

ui.canvas.addEventListener('pointerup', endPointer);
ui.canvas.addEventListener('pointercancel', endPointer);
ui.canvas.addEventListener('contextmenu', event => event.preventDefault());
ui.canvas.addEventListener('auxclick', event => event.preventDefault());
ui.canvas.addEventListener('wheel', event => {
  updatePointer(event);
  camera.zoomAbout(pointerX, pointerY, Math.exp(event.deltaY * 0.0012), aspect());
  beginInteraction(true);
  requestCurrentRender();
  event.preventDefault();
}, { passive: false });

ui.speed.addEventListener('input', () => {
  if (lastSnapshot) updateReadouts(lastSnapshot);
});
ui.iterations.addEventListener('input', () => {
  adaptiveResolution = 1;
  smoothedFrameMs = TARGET_FRAME_MS;
  clearTimers();
  setStage('full-quality');
  invalidateAndRender();
});
ui.palette.addEventListener('input', () => {
  clearTimers();
  setStage('full-quality');
  invalidateAndRender();
});
new ResizeObserver(() => {
  adaptiveResolution = 1;
  invalidateAndRender();
}).observe(ui.canvas);

renderer.onDeviceError(message => { ui.status.textContent = message; });
ui.status.textContent = `${renderer.adapterLabel} · numerical core V4.2`;

let previousTick = performance.now();
function tick(now: number): void {
  const deltaSeconds = Math.min(0.05, (now - previousTick) / 1000);
  previousTick = now;
  if (direction !== 0) {
    camera.zoomAbout(pointerX, pointerY, Math.exp(-direction * Number(ui.speed.value) * deltaSeconds), aspect());
    requestCurrentRender();
  }
  requestAnimationFrame(tick);
}

requestCurrentRender();
requestAnimationFrame(tick);
