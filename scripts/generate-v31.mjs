import { readFile, writeFile } from 'node:fs/promises';

const sourcePath = new URL('../src/mainV3.ts', import.meta.url);
const outputPath = new URL('../src/mainV31.generated.ts', import.meta.url);
let source = await readFile(sourcePath, 'utf8');

function replaceExact(label, before, after) {
  if (!source.includes(before)) throw new Error(`V3.1 generator could not find: ${label}`);
  source = source.replace(before, after);
}

replaceExact(
  'core label',
  '<p><b>Numerical core V3:</b> persistent high-precision references, double-single perturbation, rebasing and cancellable background refinement.</p>',
  '<p><b>Numerical core V3.1:</b> non-blocking reference states, useful out-of-order orbit acceptance, double-single perturbation and corrected rebasing.</p>'
);

replaceExact(
  'status label',
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3`;",
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.1`;"
);

replaceExact(
  'display precision type',
  "type DisplayPrecision = 'f32' | 'double-float' | 'perturbation' | 'awaiting reference';",
  "type DisplayPrecision = 'f32' | 'double-float' | 'provisional double-float' | 'perturbation';"
);

replaceExact(
  'reference types',
`type ReferenceCache = {
  key: string;
  centerX: BigFixed;
  centerY: BigFixed;
  iterations: number;
  length: number;
  escaped: boolean;
  bits: number;
  ms: number;
};`,
`type ReferencePurpose = 'provisional' | 'settled';
type ReferenceCache = {
  key: string;
  centerX: BigFixed;
  centerY: BigFixed;
  requestedIterations: number;
  length: number;
  escaped: boolean;
  bits: number;
  ms: number;
  purpose: ReferencePurpose;
};
type ReferenceRequestMeta = {
  key: string;
  centerX: BigFixed;
  centerY: BigFixed;
  requestedIterations: number;
  purpose: ReferencePurpose;
};`
);

replaceExact(
  'reference globals',
`let referenceCache: ReferenceCache | null = null;
let pendingReferenceKey = '';
let pendingCenterX: BigFixed | null = null;
let pendingCenterY: BigFixed | null = null;
let referenceRequestId = 0;`,
`let referenceCache: ReferenceCache | null = null;
let referenceRequestId = 0;
const pendingReferences = new Map<number, ReferenceRequestMeta>();`
);

replaceExact(
  'reference request lifecycle',
`function requestReference(iterationLimit: number): void {
  const key = referenceKey(centerX, centerY, iterationLimit);
  if (pendingReferenceKey === key) return;
  pendingReferenceKey = key;
  pendingCenterX = centerX;
  pendingCenterY = centerY;
  referenceRequestId++;
  const request: ReferenceRequest = {
    id: referenceRequestId,
    centerX: serializeFixed(centerX),
    centerY: serializeFixed(centerY),
    iterations: iterationLimit
  };
  worker.postMessage(request);
}
worker.addEventListener('message', event => {
  const response = event.data as ReferenceResponse;
  if (response.id !== referenceRequestId || !pendingCenterX || !pendingCenterY) return;
  const key = pendingReferenceKey;
  device.queue.writeBuffer(orbitBuffer, 0, response.orbit);
  referenceCache = {
    key,
    centerX: pendingCenterX,
    centerY: pendingCenterY,
    iterations: response.length - 1,
    length: response.length,
    escaped: response.escaped,
    bits: response.bits,
    ms: response.generationMs
  };
  pendingReferenceKey = '';
  pendingCenterX = null;
  pendingCenterY = null;
  requestRender();
});`,
`function hasPendingReference(key: string, purpose?: ReferencePurpose): boolean {
  for (const pending of pendingReferences.values()) {
    if (pending.key === key && (!purpose || pending.purpose === purpose)) return true;
  }
  return false;
}
function requestReference(iterationLimit: number, purpose: ReferencePurpose): void {
  const key = referenceKey(centerX, centerY, iterationLimit);
  if (hasPendingReference(key, purpose)) return;
  const id = ++referenceRequestId;
  pendingReferences.set(id, {
    key,
    centerX,
    centerY,
    requestedIterations: iterationLimit,
    purpose
  });
  const request: ReferenceRequest = {
    id,
    centerX: serializeFixed(centerX),
    centerY: serializeFixed(centerY),
    iterations: iterationLimit
  };
  worker.postMessage(request);
}
function referenceDistanceInViewports(x: BigFixed, y: BigFixed): number {
  const dx = fixedDifferenceToNumber(centerX, x);
  const dy = fixedDifferenceToNumber(centerY, y);
  const scale = viewportScale.mantissa * Math.pow(2, viewportScale.exponent);
  return Math.hypot(dx, dy) / Math.max(scale, Number.MIN_VALUE);
}
function shouldActivateReference(candidate: ReferenceCache): boolean {
  const candidateDistance = referenceDistanceInViewports(candidate.centerX, candidate.centerY);
  if (!Number.isFinite(candidateDistance) || candidateDistance > REFERENCE_REUSE_VIEWPORTS * 8) return false;
  if (!referenceCache) return true;
  const currentDistance = referenceDistanceInViewports(referenceCache.centerX, referenceCache.centerY);
  if (candidate.purpose === 'settled' && referenceCache.purpose === 'provisional') return true;
  if (candidate.purpose === referenceCache.purpose && candidate.requestedIterations > referenceCache.requestedIterations && candidateDistance <= currentDistance + .5) return true;
  return candidateDistance + .25 < currentDistance;
}
worker.addEventListener('message', event => {
  const response = event.data as ReferenceResponse;
  const meta = pendingReferences.get(response.id);
  if (!meta) return;
  pendingReferences.delete(response.id);
  const candidate: ReferenceCache = {
    key: meta.key,
    centerX: meta.centerX,
    centerY: meta.centerY,
    requestedIterations: meta.requestedIterations,
    length: response.length,
    escaped: response.escaped,
    bits: response.bits,
    ms: response.generationMs,
    purpose: meta.purpose
  };
  if (shouldActivateReference(candidate)) {
    device.queue.writeBuffer(orbitBuffer, 0, response.orbit);
    referenceCache = candidate;
  }
  requestRender();
});`
);

replaceExact(
  'rebasing recurrence',
`    if(refIndex>0u&&radius<deltaRadius){
      dzx=currentX; dzy=currentY; refIndex=0u;
    }

    let dzSq=complexSquare(dzx,dzy);`,
`    if(refIndex>0u&&radius<deltaRadius){
      dzx=currentX; dzy=currentY; refIndex=0u;
      continue;
    }

    let dzSq=complexSquare(dzx,dzy);`
);

replaceExact(
  'deep reference selection',
`  if (order >= PERTURBATION_THRESHOLD_LOG10) {
    const fullIterations = requestedIterations();
    const cachedOffset = referenceCache ? referenceOffset(referenceCache) : null;
    const cacheUsable = Boolean(referenceCache && cachedOffset && cachedOffset.viewports <= REFERENCE_REUSE_VIEWPORTS && referenceCache.iterations >= quality.iterations);

    if (!interactive) {
      const exactKey = referenceKey(centerX, centerY, fullIterations);
      if (referenceCache?.key !== exactKey || referenceCache.iterations < fullIterations) requestReference(fullIterations);
    } else if (!referenceCache && !pendingReferenceKey) {
      requestReference(Math.min(fullIterations, 2500));
    }

    if (cacheUsable && referenceCache && cachedOffset) {
      mode = 2;
      refOffsetX = cachedOffset.x;
      refOffsetY = cachedOffset.y;
      orbitLength = referenceCache.length;
      displayedPrecision = 'perturbation';
      const state = pendingReferenceKey ? ' · refining reference' : '';
      orbitStatus = `${referenceCache.bits}-bit · ${referenceCache.length - 1} iter · ${cachedOffset.viewports.toFixed(2)} view offset${state}`;
    } else if (referenceCache && cachedOffset && referenceCache.iterations >= quality.iterations) {
      mode = 2;
      refOffsetX = cachedOffset.x;
      refOffsetY = cachedOffset.y;
      orbitLength = referenceCache.length;
      displayedPrecision = 'perturbation';
      orbitStatus = `${referenceCache.bits}-bit · extended offset · rebuilding`;
      if (!pendingReferenceKey) requestReference(fullIterations);
    } else {
      displayedPrecision = 'awaiting reference';
      orbitStatus = `building ${coordinateBits}-bit reference…`;
      frameInFlight = false;
      updateReadouts();
      return;
    }
  } else {
    displayedPrecision = mode === 0 ? 'f32' : 'double-float';
  }`,
`  if (order >= PERTURBATION_THRESHOLD_LOG10) {
    const fullIterations = requestedIterations();
    const provisionalIterations = Math.min(fullIterations, 4096);
    const cachedOffset = referenceCache ? referenceOffset(referenceCache) : null;
    const spatiallyUsable = Boolean(referenceCache && cachedOffset && cachedOffset.viewports <= REFERENCE_REUSE_VIEWPORTS * 3);
    const iterationUsable = Boolean(referenceCache && (interactive || referenceCache.requestedIterations >= quality.iterations || referenceCache.escaped));
    const cacheUsable = spatiallyUsable && iterationUsable;

    if (interactive) {
      const provisionalKey = referenceKey(centerX, centerY, provisionalIterations);
      if (!cacheUsable && !hasPendingReference(provisionalKey, 'provisional')) requestReference(provisionalIterations, 'provisional');
    } else {
      const settledKey = referenceKey(centerX, centerY, fullIterations);
      const exactSettled = referenceCache?.key === settledKey && referenceCache.purpose === 'settled';
      if (!exactSettled && !hasPendingReference(settledKey, 'settled')) requestReference(fullIterations, 'settled');
    }

    if (cacheUsable && referenceCache && cachedOffset) {
      mode = 2;
      refOffsetX = cachedOffset.x;
      refOffsetY = cachedOffset.y;
      orbitLength = referenceCache.length;
      displayedPrecision = 'perturbation';
      const pending = pendingReferences.size ? ' · refining reference' : '';
      const ended = referenceCache.escaped ? ` · escaped at ${referenceCache.length - 1}` : '';
      orbitStatus = `~${referenceCache.bits}-bit ${referenceCache.purpose} · ${referenceCache.length - 1} stored · ${cachedOffset.viewports.toFixed(2)} view offset${ended}${pending}`;
    } else {
      mode = 1;
      displayedPrecision = 'provisional double-float';
      orbitStatus = `building reference in background · ${pendingReferences.size} queued`;
    }
  } else {
    displayedPrecision = mode === 0 ? 'f32' : 'double-float';
  }`
);

replaceExact(
  'fps label',
  '<label>GPU frame rate <output id="fpsOut">0.0 FPS</output></label>',
  '<label>Completed render rate <output id="fpsOut">0.0 FPS</output></label>'
);

await writeFile(outputPath, source);
console.log('Generated src/mainV31.generated.ts');
