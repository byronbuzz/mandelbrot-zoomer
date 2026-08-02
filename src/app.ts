import './style.css';
import { fixedDifferenceToNumber } from './bigFixed';
import { scaleToNumber } from './binaryScale';
import { BUILD_LABEL } from './v4/build';
import { CameraModel } from './v4/camera';
import { ReferenceManager } from './v4/referenceManager';
import {
  RenderCoordinator,
  type DroppedInteractiveFrame,
  type PresentedFrame
} from './v4/renderCoordinator';
import type {
  CpuReference,
  GpuReference,
  PrecisionMode,
  RenderQuality,
  RenderSnapshot,
  RenderStage,
  RenderTelemetry
} from './v4/types';
import { createUi, iterationCount, resetControlValues } from './v4/ui';
import { WebGpuRenderer } from './v4/webGpuRenderer';
import {
  NavigationQualityController,
  PresentationRateMeter
} from './v5/navigationQualityController';

const app = document.querySelector<HTMLElement>('#app');
if (!app) throw new Error('Missing app root');
const ui = createUi(app);
const camera = new CameraModel();
const renderer = await WebGpuRenderer.create(ui.canvas);
const references = new ReferenceManager();
const navigationQuality = new NavigationQualityController();
const presentationRate = new PresentationRateMeter();

const SETTLE_MS = 180;
const REFINE_DELAY_MS = 100;
const DOUBLE_FLOAT_THRESHOLD_LOG10 = 4;
const REFERENCE_PREFETCH_LOG10 = 4;
const PERTURBATION_THRESHOLD_LOG10 = 4.5;
const DOUBLE_FLOAT_INTERACTIVE_RESOLUTION = 0.9;
const REFERENCE_FALLBACK_INTERACTIVE_RESOLUTION = 0.65;
const PERTURBATION_INTERACTIVE_RESOLUTION = 0.85;
const MAX_GLOBAL_REPAIR_PASSES = 4;
const MAX_TILE_REPAIR_PASSES = 8;
const GLOBAL_REPAIR_THRESHOLD_FRACTION = 0.15;
const REPAIR_TARGET_FRACTION = 0.0005;
const MIN_TILE_REPAIR_RELATIVE_IMPROVEMENT = 0.005;
const MAX_STALLED_TILE_REPAIR_PASSES = 2;

type RepairMode = 'global' | 'tile';

let presentationEpoch = 1;
let stage: RenderStage = 'full-quality';
let refineStep = 0;
let lastAnchorFps = 0;
let lastDisplayFps = 0;
let lastComputeMs = 0;
let lastPresentMs = 0;
let lastComputeBatches = 0;
let lastReferenceAnchorMovePixels = 0;
let direction = 0;
let pointerX = 0.5;
let pointerY = 0.5;
let panning = false;
let panPointer = -1;
let panX = 0;
let panY = 0;
let settleTimer = 0;
let refineTimer = 0;
let copyFeedbackTimer = 0;
let activeReference: GpuReference | null = null;
let repairReference: GpuReference | null = null;
let lastSnapshot: RenderSnapshot | null = null;
let lastTelemetry: RenderTelemetry | null = null;
let deferredReference: CpuReference | null = null;
let repairPass = 0;
let globalRepairPass = 0;
let tileRepairPass = 0;
let repairRequestEpoch = -1;
let repairRequestCameraGeneration = -1;
let repairRequestMode: RepairMode | null = null;
let activeRepairMode: RepairMode | null = null;
let lastTileRepairUnresolved = Number.POSITIVE_INFINITY;
let stalledTileRepairPasses = 0;
let lastRateUiUpdate = 0;
const retiredReferences: GpuReference[] = [];

function requestedIterations(): number { return iterationCount(ui.iterations.value); }
function aspect(): number { return Math.max(1, ui.canvas.clientWidth) / Math.max(1, ui.canvas.clientHeight); }
function isInteracting(): boolean { return stage === 'interactive'; }

navigationQuality.reset(requestedIterations());

function clearTimers(): void {
  if (settleTimer) window.clearTimeout(settleTimer);
  if (refineTimer) window.clearTimeout(refineTimer);
  settleTimer = 0;
  refineTimer = 0;
}

function resetRepairState(): void {
  repairPass = 0;
  globalRepairPass = 0;
  tileRepairPass = 0;
  repairRequestEpoch = -1;
  repairRequestCameraGeneration = -1;
  repairRequestMode = null;
  activeRepairMode = null;
  lastTileRepairUnresolved = Number.POSITIVE_INFINITY;
  stalledTileRepairPasses = 0;
  if (repairReference) retiredReferences.push(repairReference);
  repairReference = null;
}

function invalidatePresentation(): void {
  presentationEpoch++;
  lastTelemetry = null;
  resetRepairState();
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

function interactiveResolutionCap(): number {
  const depth = camera.log10Magnification();
  if (depth >= PERTURBATION_THRESHOLD_LOG10 && !activeReference) {
    return REFERENCE_FALLBACK_INTERACTIVE_RESOLUTION;
  }
  if (depth >= PERTURBATION_THRESHOLD_LOG10) return PERTURBATION_INTERACTIVE_RESOLUTION;
  if (depth >= DOUBLE_FLOAT_THRESHOLD_LOG10) return DOUBLE_FLOAT_INTERACTIVE_RESOLUTION;
  return 1;
}

function effectiveQuality(): RenderQuality {
  const iterations = requestedIterations();
  if (stage === 'interactive') {
    return navigationQuality.quality(iterations, interactiveResolutionCap());
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

function buildSnapshot(
  reference: GpuReference | null = activeReference,
  repairPassValue = 0,
  manageReferences = true
): RenderSnapshot {
  const cameraSnapshot = camera.snapshot();
  const quality = effectiveQuality();
  if (manageReferences) manageReferenceRequests(cameraSnapshot, quality.iterations);
  return {
    generation: presentationEpoch,
    camera: cameraSnapshot,
    stage,
    quality,
    palettePhase: Number(ui.palette.value),
    cssWidth: Math.max(1, ui.canvas.clientWidth),
    cssHeight: Math.max(1, ui.canvas.clientHeight),
    devicePixelRatio,
    reference,
    precision: precisionFor(reference),
    repairPass: repairPassValue
  };
}

function notePresentation(): void {
  presentationRate.note();
  lastDisplayFps = presentationRate.rate();
}

function requestCurrentRender(): void {
  const snapshot = buildSnapshot();
  lastSnapshot = snapshot;
  if (coordinator.request(snapshot)) notePresentation();
  updateReadouts(snapshot);
}

function requestRepairRender(reference: GpuReference): void {
  const snapshot = buildSnapshot(reference, repairPass, false);
  lastSnapshot = snapshot;
  if (coordinator.request(snapshot)) notePresentation();
  updateReadouts(snapshot);
}

function invalidateAndRender(): void {
  invalidatePresentation();
  requestCurrentRender();
}

function worstUnresolvedTile(telemetry: RenderTelemetry): { x: number; y: number; count: number } | null {
  let bestIndex = -1;
  let bestCount = 0;
  for (let index = 0; index < telemetry.tileUnresolved.length; index++) {
    const count = telemetry.tileUnresolved[index] ?? 0;
    if (count <= bestCount) continue;
    bestCount = count;
    bestIndex = index;
  }
  if (bestIndex < 0 || bestCount <= 0) return null;
  return {
    x: bestIndex % telemetry.tileColumns,
    y: Math.floor(bestIndex / telemetry.tileColumns),
    count: bestCount
  };
}

function markRepairRequest(mode: RepairMode): void {
  repairRequestEpoch = presentationEpoch;
  repairRequestCameraGeneration = camera.generation;
  repairRequestMode = mode;
}

function clearRepairRequest(): void {
  repairRequestEpoch = -1;
  repairRequestCameraGeneration = -1;
  repairRequestMode = null;
}

function maybeScheduleRepair(frame: PresentedFrame): void {
  const telemetry = frame.telemetry;
  const snapshot = frame.snapshot;
  if (!telemetry || snapshot.stage !== 'full-quality' || snapshot.precision !== 'perturbation') return;
  if (snapshot.generation !== presentationEpoch || snapshot.camera.generation !== camera.generation) return;
  if (snapshot.repairPass === 0 && snapshot.reference?.purpose !== 'settled') return;
  if (repairRequestEpoch >= 0) return;

  const unresolvedFraction = telemetry.totalPixels > 0
    ? telemetry.unresolvedPixels / telemetry.totalPixels
    : 0;
  if (unresolvedFraction <= REPAIR_TARGET_FRACTION) return;

  if (unresolvedFraction > GLOBAL_REPAIR_THRESHOLD_FRACTION && globalRepairPass < MAX_GLOBAL_REPAIR_PASSES) {
    markRepairRequest('global');
    const requested = references.requestGlobalRepair(
      camera.snapshot(),
      requestedIterations(),
      aspect(),
      globalRepairPass
    );
    if (!requested) clearRepairRequest();
    return;
  }

  if (tileRepairPass >= MAX_TILE_REPAIR_PASSES || stalledTileRepairPasses >= MAX_STALLED_TILE_REPAIR_PASSES) return;
  if (!Number.isFinite(lastTileRepairUnresolved)) lastTileRepairUnresolved = telemetry.unresolvedPixels;
  const tile = worstUnresolvedTile(telemetry);
  if (!tile) return;
  const normalizedX = (tile.x + 0.5) / telemetry.tileColumns;
  const normalizedY = (tile.y + 0.5) / telemetry.tileRows;
  markRepairRequest('tile');
  const requested = references.requestRepair(
    camera.snapshot(),
    requestedIterations(),
    aspect(),
    normalizedX,
    normalizedY
  );
  if (!requested) clearRepairRequest();
}

function onPresented(frame: PresentedFrame): void {
  lastComputeMs = frame.computeMs;
  lastPresentMs = frame.presentMs;
  lastComputeBatches = frame.computeBatches;
  notePresentation();

  if (frame.snapshot.stage === 'interactive') {
    navigationQuality.observeCompleted(
      frame.computeMs + frame.presentMs,
      frame.snapshot.quality,
      requestedIterations(),
      interactiveResolutionCap()
    );
    lastAnchorFps = navigationQuality.anchorFps;
  }
  if (frame.telemetry) lastTelemetry = frame.telemetry;

  if (frame.snapshot.repairPass > 0 && activeRepairMode === 'tile' && frame.telemetry) {
    const relativeImprovement = (lastTileRepairUnresolved - frame.telemetry.unresolvedPixels)
      / Math.max(1, lastTileRepairUnresolved);
    if (relativeImprovement < MIN_TILE_REPAIR_RELATIVE_IMPROVEMENT) stalledTileRepairPasses++;
    else stalledTileRepairPasses = 0;
    lastTileRepairUnresolved = frame.telemetry.unresolvedPixels;
  }

  updateReadouts(frame.snapshot);
  if (frame.snapshot.stage === 'refining' && frame.snapshot.generation === presentationEpoch) {
    scheduleNextRefinement();
  }
  if (frame.snapshot.repairPass > 0 && repairReference) {
    retiredReferences.push(repairReference);
    repairReference = null;
  }
  activeRepairMode = null;
  maybeScheduleRepair(frame);
}

function onInteractiveDropped(frame: DroppedInteractiveFrame): void {
  navigationQuality.observeDropped(frame.snapshot.quality, requestedIterations());
  lastComputeMs = frame.elapsedMs;
  lastComputeBatches = 0;
  if (lastSnapshot) updateReadouts(lastSnapshot);
}

function onCoordinatorIdle(): void {
  while (retiredReferences.length) renderer.destroyReference(retiredReferences.shift()!);
}

const coordinator = new RenderCoordinator(
  renderer,
  () => presentationEpoch,
  onPresented,
  onInteractiveDropped,
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
  const cameraSnapshot = camera.snapshot();
  const viewportScale = Math.max(scaleToNumber(cameraSnapshot.scale), Number.MIN_VALUE);
  if (activeReference) {
    const dx = fixedDifferenceToNumber(candidate.centerX, activeReference.centerX);
    const dy = fixedDifferenceToNumber(candidate.centerY, activeReference.centerY);
    lastReferenceAnchorMovePixels = Math.hypot(dx, dy)
      / viewportScale
      * Math.max(1, ui.canvas.clientHeight);
  } else {
    lastReferenceAnchorMovePixels = 0;
  }

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
  if (candidate.purpose === 'repair') {
    const expectedEpoch = repairRequestEpoch;
    const expectedCameraGeneration = repairRequestCameraGeneration;
    const mode = repairRequestMode;
    clearRepairRequest();
    if (
      !mode
      || expectedEpoch < 0
      || expectedEpoch !== presentationEpoch
      || expectedCameraGeneration !== camera.generation
      || candidate.cameraGeneration !== camera.generation
      || stage !== 'full-quality'
    ) return;
    if (repairReference) retiredReferences.push(repairReference);
    repairReference = renderer.createReference(candidate);
    repairPass++;
    if (mode === 'global') globalRepairPass++;
    else tileRepairPass++;
    activeRepairMode = mode;
    requestRepairRender(repairReference);
    return;
  }

  if (!candidateIsUseful(candidate)) return;
  const currentHealthy = activeReference && !referenceIsWeak(activeReference, requestedIterations());
  if (isInteracting() && currentHealthy) {
    if (!deferredReference || candidate.length > deferredReference.length) deferredReference = candidate;
    return;
  }
  activateReference(candidate);
});

function updateRateReadouts(): void {
  ui.displayFpsOut.value = `${lastDisplayFps.toFixed(0)} Hz`;
  ui.fpsOut.value = `${lastAnchorFps.toFixed(1)} FPS`;
  ui.hudFpsOut.value = `${lastDisplayFps.toFixed(0)} Hz · ${lastAnchorFps.toFixed(1)} FPS`;
}

function updateReadouts(snapshot: RenderSnapshot): void {
  const reference = snapshot.reference;
  const offset = reference ? referenceOffsetInViewports(reference) : 0;
  const ended = reference?.escaped ? ` · escaped at ${reference.length - 1}` : '';
  const pending = references.pendingCount ? ` · ${references.pendingCount} reference queued` : '';
  const zoomLabel = `10^${camera.log10Magnification().toFixed(2)}`;

  ui.zoomOut.value = zoomLabel;
  ui.hudZoomOut.value = zoomLabel;
  ui.precisionOut.value = snapshot.precision;
  ui.orbitOut.value = reference
    ? `~${reference.bits}-bit ${reference.purpose} · ${reference.length - 1} stored · ${offset.toFixed(2)} view offset${ended}${pending}`
    : `inactive${pending}`;

  if (snapshot.precision !== 'perturbation') {
    ui.healthOut.value = 'inactive';
    ui.failureOut.value = 'inactive';
    ui.maxExponentOut.value = 'inactive';
  } else if (!lastTelemetry) {
    ui.healthOut.value = snapshot.stage === 'full-quality' ? 'measuring…' : 'measured on full-quality frames';
    ui.failureOut.value = 'measuring…';
    ui.maxExponentOut.value = 'measuring…';
  } else {
    const percentage = lastTelemetry.totalPixels > 0
      ? (lastTelemetry.unresolvedPixels / lastTelemetry.totalPixels) * 100
      : 0;
    const queued = repairRequestMode === 'global'
      ? ` · global rescue ${globalRepairPass + 1}/${MAX_GLOBAL_REPAIR_PASSES} queued`
      : repairRequestMode === 'tile'
        ? ` · tile repair ${tileRepairPass + 1}/${MAX_TILE_REPAIR_PASSES} queued`
        : '';
    const completed = globalRepairPass > 0 || tileRepairPass > 0
      ? ` · ${globalRepairPass} global + ${tileRepairPass} tile repairs`
      : '';
    ui.healthOut.value = `${lastTelemetry.unresolvedPixels.toLocaleString()} unresolved (${percentage.toFixed(3)}%)${queued}${completed}`;
    ui.failureOut.value = [
      `${lastTelemetry.exhaustedPixels.toLocaleString()} exhausted`,
      `${lastTelemetry.magnitudeGuardPixels.toLocaleString()} magnitude`,
      `${lastTelemetry.nonFinitePixels.toLocaleString()} non-finite`,
      `${lastTelemetry.rebaseFailurePixels.toLocaleString()} rebase`
    ].join(' · ');
    ui.maxExponentOut.value = lastTelemetry.maxPerturbationExponent === null
      ? 'below normal range'
      : `2^${lastTelemetry.maxPerturbationExponent}`;
  }

  ui.bitsOut.value = String(camera.coordinateBits);
  ui.exponentOut.value = String(snapshot.camera.scale.exponent);
  ui.stateOut.value = snapshot.stage;
  updateRateReadouts();
  ui.qualityOut.value = `${snapshot.quality.iterations} / ${requestedIterations()} iter · ${Math.round(snapshot.quality.resolution * 100)}%`;
  ui.timingOut.value = `${lastComputeMs.toFixed(1)} + ${lastPresentMs.toFixed(1)} ms`;
  ui.batchesOut.value = String(lastComputeBatches);
  ui.anchorMoveOut.value = `${lastReferenceAnchorMovePixels.toFixed(2)} px`;
  ui.referenceTimeOut.value = reference ? `${reference.generationMs.toFixed(1)} ms` : 'inactive';
  ui.adapterOut.value = renderer.adapterLabel;
  ui.buildOut.value = BUILD_LABEL;
  ui.speedOut.value = `${Number(ui.speed.value).toFixed(2)}×/s`;
  ui.iterOut.value = requestedIterations().toLocaleString();
  ui.palOut.value = Number(ui.palette.value).toFixed(2);
  ui.setStatsWarning(Boolean(lastTelemetry?.unresolvedPixels));
}

function diagnosticsReport(): string {
  const snapshot = lastSnapshot ?? buildSnapshot(activeReference, 0, false);
  const cameraSnapshot = camera.snapshot();
  const reference = snapshot.reference;
  const telemetry = lastTelemetry;
  const lines = [
    `Mandelbrot Zoomer ${BUILD_LABEL}`,
    `Captured: ${new Date().toISOString()}`,
    `GPU: ${renderer.adapterLabel}`,
    `URL: ${location.href}`,
    `Zoom depth: 10^${camera.log10Magnification().toFixed(6)}`,
    `Center X raw: ${cameraSnapshot.centerX.raw.toString()} (bits ${cameraSnapshot.centerX.bits})`,
    `Center Y raw: ${cameraSnapshot.centerY.raw.toString()} (bits ${cameraSnapshot.centerY.bits})`,
    `Scale: ${cameraSnapshot.scale.mantissa} * 2^${cameraSnapshot.scale.exponent}`,
    `Precision: ${snapshot.precision}`,
    `Stage: ${snapshot.stage}`,
    `Requested iterations: ${requestedIterations()}`,
    `Effective quality: ${snapshot.quality.iterations} iterations at ${(snapshot.quality.resolution * 100).toFixed(1)}%`,
    `Anchor-frame rate: ${lastAnchorFps.toFixed(2)} FPS`,
    `Display presentation rate: ${lastDisplayFps.toFixed(2)} Hz`,
    `Compute/present: ${lastComputeMs.toFixed(2)} / ${lastPresentMs.toFixed(2)} ms`,
    `GPU batches: ${lastComputeBatches}`,
    `Reference-anchor move: ${lastReferenceAnchorMovePixels.toFixed(4)} px`
  ];

  if (reference) {
    lines.push(
      `Reference: ${reference.purpose}, ${reference.bits} bits, ${reference.length - 1} stored, escaped=${reference.escaped}`,
      `Reference offset: ${referenceOffsetInViewports(reference).toFixed(8)} viewports`,
      `Reference generation: ${reference.generationMs.toFixed(3)} ms`
    );
  } else {
    lines.push('Reference: inactive');
  }

  if (telemetry) {
    const unresolvedPercentage = telemetry.totalPixels > 0
      ? telemetry.unresolvedPixels / telemetry.totalPixels * 100
      : 0;
    lines.push(
      `Unresolved: ${telemetry.unresolvedPixels}/${telemetry.totalPixels} (${unresolvedPercentage.toFixed(6)}%)`,
      `Failure causes: exhausted=${telemetry.exhaustedPixels}, magnitude=${telemetry.magnitudeGuardPixels}, non-finite=${telemetry.nonFinitePixels}, rebase=${telemetry.rebaseFailurePixels}`,
      `Maximum perturbation exponent: ${telemetry.maxPerturbationExponent ?? 'below normal range'}`,
      `Repairs: global=${globalRepairPass}, tile=${tileRepairPass}`
    );
  } else {
    lines.push('Perturbation telemetry: inactive');
  }

  return lines.join('\n');
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
  ui.canvas.focus({ preventScroll: true });
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
  navigationQuality.reset(requestedIterations());
  clearTimers();
  setStage('full-quality');
  invalidateAndRender();
});
ui.palette.addEventListener('input', () => {
  clearTimers();
  setStage('full-quality');
  invalidateAndRender();
});
ui.resetControls.addEventListener('click', () => {
  resetControlValues(ui);
  navigationQuality.reset(requestedIterations());
  clearTimers();
  setStage('full-quality');
  invalidateAndRender();
});
ui.resetLocation.addEventListener('click', () => {
  clearTimers();
  direction = 0;
  panning = false;
  panPointer = -1;
  pointerX = 0.5;
  pointerY = 0.5;
  camera.reset();
  references.cancelOlderThan(camera.generation);
  deferredReference = null;
  if (activeReference) retiredReferences.push(activeReference);
  activeReference = null;
  lastReferenceAnchorMovePixels = 0;
  navigationQuality.reset(requestedIterations());
  stage = 'full-quality';
  refineStep = 0;
  invalidatePresentation();
  requestCurrentRender();
});
ui.copyDiagnostics.addEventListener('click', () => {
  void copyText(diagnosticsReport())
    .then(() => showCopyFeedback('Copied'))
    .catch(error => {
      console.error(error);
      showCopyFeedback('Copy failed');
    });
});

new ResizeObserver(() => {
  navigationQuality.reset(requestedIterations());
  invalidateAndRender();
}).observe(ui.canvas);

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
ui.status.textContent = `${renderer.adapterLabel} · ${BUILD_LABEL}`;

let previousTick = performance.now();
function tick(now: number): void {
  const deltaSeconds = Math.min(0.05, (now - previousTick) / 1000);
  previousTick = now;
  if (direction !== 0) {
    camera.zoomAbout(pointerX, pointerY, Math.exp(-direction * Number(ui.speed.value) * deltaSeconds), aspect());
    requestCurrentRender();
  }
  lastDisplayFps = presentationRate.rate(now);
  if (now - lastRateUiUpdate >= 250) {
    lastRateUiUpdate = now;
    updateRateReadouts();
  }
  requestAnimationFrame(tick);
}

requestCurrentRender();
requestAnimationFrame(tick);
