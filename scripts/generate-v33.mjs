import { readFile, writeFile } from 'node:fs/promises';

await import('./generate-v32.mjs');

const sourcePath = new URL('../src/mainV32.generated.ts', import.meta.url);
const outputPath = new URL('../src/mainV33.generated.ts', import.meta.url);
let source = await readFile(sourcePath, 'utf8');

function replaceExact(label, before, after) {
  if (!source.includes(before)) throw new Error(`V3.3 generator could not find: ${label}`);
  source = source.replace(before, after);
}

replaceExact(
  'core label',
  '<p><b>Numerical core V3.2:</b> quantization-corrected references, atomic GPU orbit promotion and a hard idle state after settled rendering.</p>',
  '<p><b>Numerical core V3.3:</b> sticky perturbation references, continuity-gated promotion and no deep double-float fallback.</p>'
);

replaceExact(
  'status label',
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.2`;",
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.3`;"
);

replaceExact(
  'earlier perturbation preparation',
  'const PERTURBATION_THRESHOLD_LOG10 = 5;',
  'const PERTURBATION_THRESHOLD_LOG10 = 4.5;'
);

replaceExact(
  'rejected-reference memory',
  "let lastCompletedRenderKey = '';",
  "let lastCompletedRenderKey = '';\nconst rejectedReferenceKeys = new Set<string>();"
);

replaceExact(
  'request rejection guard',
  '  if (hasPendingReference(key, purpose)) return;',
  '  if (hasPendingReference(key, purpose) || rejectedReferenceKeys.has(key)) return;'
);

replaceExact(
  'activation policy',
`function shouldActivateReference(candidate: ReferenceCache): boolean {
  const candidateDistance = referenceDistanceInViewports(candidate.centerX, candidate.centerY);
  if (!Number.isFinite(candidateDistance) || candidateDistance > REFERENCE_REUSE_VIEWPORTS * 8) return false;
  if (!referenceCache) return true;
  const currentDistance = referenceDistanceInViewports(referenceCache.centerX, referenceCache.centerY);
  if (candidate.purpose === 'settled' && referenceCache.purpose === 'provisional') return true;
  if (candidate.purpose === referenceCache.purpose && candidate.requestedIterations > referenceCache.requestedIterations && candidateDistance <= currentDistance + .5) return true;
  return candidateDistance + .25 < currentDistance;
}`,
`function shouldActivateReference(candidate: ReferenceCache): boolean {
  const candidateDistance = referenceDistanceInViewports(candidate.centerX, candidate.centerY);
  if (!Number.isFinite(candidateDistance)) return false;
  if (!referenceCache) return candidateDistance <= REFERENCE_REUSE_VIEWPORTS * 8;
  // Never replace a visually stable local frame with a quantized reference that
  // represents a noticeably different viewport centre.
  if (candidateDistance > .125) return false;
  if (candidate.purpose === 'settled' && referenceCache.purpose === 'provisional') return true;
  return candidate.requestedIterations >= referenceCache.requestedIterations;
}`
);

replaceExact(
  'remember rejected candidates',
`  if (!shouldActivateReference(candidate)) {
    candidateBuffer.destroy();
    return;
  }`,
`  if (!shouldActivateReference(candidate)) {
    rejectedReferenceKeys.add(meta.key);
    candidateBuffer.destroy();
    return;
  }`
);

replaceExact(
  'sticky spatial validity',
  '    const spatiallyUsable = Boolean(referenceCache && cachedOffset && cachedOffset.viewports <= REFERENCE_REUSE_VIEWPORTS * 3);',
  '    const spatiallyUsable = Boolean(referenceCache && cachedOffset && Number.isFinite(cachedOffset.viewports));'
);

replaceExact(
  'sticky orbit diagnostic',
  '      orbitStatus = `~${referenceCache.bits}-bit ${referenceCache.purpose} · ${referenceCache.length - 1} stored · ${cachedOffset.viewports.toFixed(2)} view offset${ended}${pending}`;',
  '      orbitStatus = `~${referenceCache.bits}-bit ${referenceCache.purpose} · ${referenceCache.length - 1} stored · sticky ${cachedOffset.viewports.toFixed(2)} view offset${ended}${pending}`;'
);

await writeFile(outputPath, source);
console.log('Generated src/mainV33.generated.ts');
