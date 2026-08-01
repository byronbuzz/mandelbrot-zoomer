import { readFile, writeFile } from 'node:fs/promises';

await import('./generate-v34.mjs');

const mainSourcePath = new URL('../src/mainV34.generated.ts', import.meta.url);
const mainOutputPath = new URL('../src/mainV35.generated.ts', import.meta.url);
const workerSourcePath = new URL('../src/referenceWorker.ts', import.meta.url);
const workerOutputPath = new URL('../src/referenceWorkerV35.generated.ts', import.meta.url);

let main = await readFile(mainSourcePath, 'utf8');
let worker = await readFile(workerSourcePath, 'utf8');

function replaceExact(target, label, before, after) {
  if (!target.includes(before)) throw new Error(`V3.5 generator could not find: ${label}`);
  return target.replace(before, after);
}

main = replaceExact(
  main,
  'core label',
  '<p><b>Numerical core V3.4:</b> adaptive double-double/triple-double CPU references, sticky perturbation and continuity-safe promotion.</p>',
  '<p><b>Numerical core V3.5:</b> adaptive high-precision references, nearby long-orbit selection and seamless background reference refresh.</p>'
);

main = replaceExact(
  main,
  'status label',
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.4`;",
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.5`;"
);

main = replaceExact(
  main,
  'remove obsolete reference reuse constant',
  'const REFERENCE_REUSE_VIEWPORTS = 2.5;\n',
  ''
);

main = main.replaceAll("new URL('./referenceWorker.ts', import.meta.url)", "new URL('./referenceWorkerV35.generated.ts', import.meta.url)");

main = replaceExact(
  main,
  'candidate request payload',
`  const request: ReferenceRequest = {
    id,
    centerX: serializeFixed(centerX),
    centerY: serializeFixed(centerY),
    iterations: iterationLimit
  };`,
`  const candidateOffsets = [
    [0, 0], [-.3, 0], [.3, 0], [0, -.3], [0, .3],
    [-.22, -.22], [.22, -.22], [-.22, .22], [.22, .22]
  ] as const;
  const candidates = candidateOffsets.map(([dx, dy]) => ({
    centerX: serializeFixed(fixedAddScaled(centerX, dx * viewportScale.mantissa, viewportScale.exponent)),
    centerY: serializeFixed(fixedAddScaled(centerY, dy * viewportScale.mantissa, viewportScale.exponent))
  }));
  const request = {
    id,
    centerX: serializeFixed(centerX),
    centerY: serializeFixed(centerY),
    iterations: iterationLimit,
    candidates
  } as ReferenceRequest & { candidates: Array<{ centerX: ReturnType<typeof serializeFixed>; centerY: ReturnType<typeof serializeFixed> }> };`
);

main = replaceExact(
  main,
  'activation policy',
`function shouldActivateReference(candidate: ReferenceCache): boolean {
  const candidateDistance = referenceDistanceInViewports(candidate.centerX, candidate.centerY);
  if (!Number.isFinite(candidateDistance)) return false;
  if (!referenceCache) return candidateDistance <= REFERENCE_REUSE_VIEWPORTS * 8;
  // Never replace a visually stable local frame with a quantized reference that
  // represents a noticeably different viewport centre.
  if (candidateDistance > .125) return false;
  if (candidate.purpose === 'settled' && referenceCache.purpose === 'provisional') return true;
  return candidate.requestedIterations >= referenceCache.requestedIterations;
}`,
`function shouldActivateReference(candidate: ReferenceCache): boolean {
  const candidateDistance = referenceDistanceInViewports(candidate.centerX, candidate.centerY);
  if (!Number.isFinite(candidateDistance) || candidateDistance > .75) return false;
  if (!referenceCache) return true;
  const currentDistance = referenceDistanceInViewports(referenceCache.centerX, referenceCache.centerY);
  const currentWeak = currentDistance > 2 || (referenceCache.escaped && referenceCache.length < Math.min(referenceCache.requestedIterations, 4096));
  const candidateStronger = !candidate.escaped || candidate.length > referenceCache.length;
  if (currentWeak && candidateStronger) return true;
  if (candidate.purpose === 'settled' && referenceCache.purpose === 'provisional') return true;
  return candidate.requestedIterations >= referenceCache.requestedIterations && candidateStronger;
}`
);

main = replaceExact(
  main,
  'interactive refresh',
`      const provisionalKey = referenceKey(centerX, centerY, provisionalIterations);
      if (!cacheUsable && !hasPendingReference(provisionalKey, 'provisional')) requestReference(provisionalIterations, 'provisional');`,
`      const provisionalKey = referenceKey(centerX, centerY, provisionalIterations);
      const refreshNeeded = !cacheUsable || (cachedOffset?.viewports ?? Infinity) > 2 || Boolean(referenceCache?.escaped && referenceCache.length < Math.min(quality.iterations, 4096));
      if (refreshNeeded && pendingReferences.size === 0 && !hasPendingReference(provisionalKey, 'provisional')) requestReference(provisionalIterations, 'provisional');`
);

main = replaceExact(
  main,
  'orbit diagnostic',
  '      orbitStatus = `~${referenceCache.bits}-bit ${referenceCache.purpose} · ${referenceCache.length - 1} stored · sticky ${cachedOffset.viewports.toFixed(2)} view offset${ended}${pending}`;',
  '      orbitStatus = `~${referenceCache.bits}-bit ${referenceCache.purpose} · ${referenceCache.length - 1} stored · ${cachedOffset.viewports.toFixed(2)} view offset${ended}${pending}`;'
);

worker = replaceExact(
  worker,
  'candidate request type',
  'type TD = readonly [number, number, number];',
`type TD = readonly [number, number, number];
type CandidateRequest = ReferenceRequest & { candidates?: Array<{ centerX: SerializedFixed; centerY: SerializedFixed }> };`
);

worker = replaceExact(
  worker,
  'reference selector',
`function buildReference(request: ReferenceRequest): void {
  const bits = Math.max(request.centerX.bits, request.centerY.bits);
  const response = bits > TRIPLE_DOUBLE_THRESHOLD_BITS
    ? buildTripleDoubleReference(request, bits)
    : buildDoubleDoubleReference(request, bits);
  worker.postMessage(response, [response.orbit.buffer]);
}`,
`function probeDoubleDouble(centerX: SerializedFixed, centerY: SerializedFixed, iterations: number): number {
  const cx = fixedToDD(BigInt(centerX.raw), centerX.bits);
  const cy = fixedToDD(BigInt(centerY.raw), centerY.bits);
  let zx: DD = [0, 0];
  let zy: DD = [0, 0];
  for (let index = 0; index < iterations; index++) {
    const zx2 = ddMul(zx, zx);
    const zy2 = ddMul(zy, zy);
    const zxy = ddMul(zx, zy);
    zx = ddAdd(ddSub(zx2, zy2), cx);
    zy = ddAdd(ddScale(zxy, 2), cy);
    const x = zx[0] + zx[1];
    const y = zy[0] + zy[1];
    if (!Number.isFinite(x) || !Number.isFinite(y) || x * x + y * y > 256) return index + 1;
  }
  return iterations + 1;
}

function probeTripleDouble(centerX: SerializedFixed, centerY: SerializedFixed, iterations: number): number {
  const cx = fixedToTD(BigInt(centerX.raw), centerX.bits);
  const cy = fixedToTD(BigInt(centerY.raw), centerY.bits);
  let zx: TD = [0, 0, 0];
  let zy: TD = [0, 0, 0];
  for (let index = 0; index < iterations; index++) {
    const zx2 = tdMul(zx, zx);
    const zy2 = tdMul(zy, zy);
    const zxy = tdMul(zx, zy);
    zx = tdAdd(tdSub(zx2, zy2), cx);
    zy = tdAdd(tdScale(zxy, 2), cy);
    const x = zx[0] + zx[1] + zx[2];
    const y = zy[0] + zy[1] + zy[2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || x * x + y * y > 256) return index + 1;
  }
  return iterations + 1;
}

function selectCandidate(request: CandidateRequest, bits: number): ReferenceRequest {
  const candidates = request.candidates?.length ? request.candidates : [{ centerX: request.centerX, centerY: request.centerY }];
  const probeIterations = Math.min(request.iterations, 2048);
  let selected = candidates[0];
  let bestScore = -1;
  for (const candidate of candidates) {
    const score = bits >= TRIPLE_DOUBLE_THRESHOLD_BITS
      ? probeTripleDouble(candidate.centerX, candidate.centerY, probeIterations)
      : probeDoubleDouble(candidate.centerX, candidate.centerY, probeIterations);
    if (score > bestScore) {
      selected = candidate;
      bestScore = score;
    }
    if (score > probeIterations) break;
  }
  return { id: request.id, centerX: selected.centerX, centerY: selected.centerY, iterations: request.iterations };
}

function buildReference(incoming: ReferenceRequest): void {
  const request = incoming as CandidateRequest;
  const bits = Math.max(request.centerX.bits, request.centerY.bits);
  const selected = selectCandidate(request, bits);
  const response = bits >= TRIPLE_DOUBLE_THRESHOLD_BITS
    ? buildTripleDoubleReference(selected, bits)
    : buildDoubleDoubleReference(selected, bits);
  worker.postMessage(response, [response.orbit.buffer]);
}`
);

await writeFile(mainOutputPath, main);
await writeFile(workerOutputPath, worker);
console.log('Generated src/mainV35.generated.ts and src/referenceWorkerV35.generated.ts');
