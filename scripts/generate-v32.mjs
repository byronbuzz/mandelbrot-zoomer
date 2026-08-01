import { readFile, writeFile } from 'node:fs/promises';

await import('./generate-v31.mjs');

const sourcePath = new URL('../src/mainV31.generated.ts', import.meta.url);
const outputPath = new URL('../src/mainV32.generated.ts', import.meta.url);
let source = await readFile(sourcePath, 'utf8');

function lines(items) {
  return items.join('\n');
}

function replaceExact(label, before, after) {
  if (!source.includes(before)) throw new Error(`V3.2 generator could not find: ${label}`);
  source = source.replace(before, after);
}

function replaceBetween(label, start, end, replacement) {
  const startIndex = source.indexOf(start);
  if (startIndex < 0) throw new Error(`V3.2 generator could not find start: ${label}`);
  const endIndex = source.indexOf(end, startIndex + start.length);
  if (endIndex < 0) throw new Error(`V3.2 generator could not find end: ${label}`);
  source = source.slice(0, startIndex) + replacement + source.slice(endIndex + end.length);
}

replaceExact(
  'fixed-point deserializer import',
  '  fixedAddScaled,',
  '  deserializeFixed,\n  fixedAddScaled,'
);

replaceExact(
  'core label',
  '<p><b>Numerical core V3.1:</b> non-blocking reference states, useful out-of-order orbit acceptance, double-single perturbation and corrected rebasing.</p>',
  '<p><b>Numerical core V3.2:</b> quantization-corrected references, atomic GPU orbit promotion and a hard idle state after settled rendering.</p>'
);

replaceExact(
  'status label',
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.1`;",
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.2`;"
);

replaceExact(
  'fallback orbit buffer',
  'const orbitBuffer = device.createBuffer({ size: (ITERATION_MAX + 1) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });',
  'const fallbackOrbitBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });'
);

replaceExact(
  'mutable worker',
  "const worker = new Worker(new URL('./referenceWorker.ts', import.meta.url), { type: 'module' });",
  "let worker = new Worker(new URL('./referenceWorker.ts', import.meta.url), { type: 'module' });"
);

replaceExact(
  'reference buffer field',
  lines([
    '  ms: number;',
    '  purpose: ReferencePurpose;',
    '};'
  ]),
  lines([
    '  ms: number;',
    '  purpose: ReferencePurpose;',
    '  buffer: GPUBuffer;',
    '};'
  ])
);

replaceExact(
  'idle render key global',
  'const pendingReferences = new Map<number, ReferenceRequestMeta>();',
  'const pendingReferences = new Map<number, ReferenceRequestMeta>();\nlet lastCompletedRenderKey = \'\';'
);

replaceBetween(
  'atomic reference lifecycle',
  'function hasPendingReference(key: string, purpose?: ReferencePurpose): boolean {',
  'function updateReadouts(): void {',
  lines([
    'function attachReferenceWorker(): void {',
    "  worker.addEventListener('message', handleReferenceResponse);",
    '}',
    'function resetReferenceWorker(): void {',
    '  worker.terminate();',
    '  pendingReferences.clear();',
    "  worker = new Worker(new URL('./referenceWorker.ts', import.meta.url), { type: 'module' });",
    '  attachReferenceWorker();',
    '}',
    'function hasPendingReference(key: string, purpose?: ReferencePurpose): boolean {',
    '  for (const pending of pendingReferences.values()) {',
    '    if (pending.key === key && (!purpose || pending.purpose === purpose)) return true;',
    '  }',
    '  return false;',
    '}',
    'function requestReference(iterationLimit: number, purpose: ReferencePurpose): void {',
    '  const key = referenceKey(centerX, centerY, iterationLimit);',
    '  if (hasPendingReference(key, purpose)) return;',
    '  const id = ++referenceRequestId;',
    '  pendingReferences.set(id, { key, centerX, centerY, requestedIterations: iterationLimit, purpose });',
    '  const request: ReferenceRequest = {',
    '    id,',
    '    centerX: serializeFixed(centerX),',
    '    centerY: serializeFixed(centerY),',
    '    iterations: iterationLimit',
    '  };',
    '  worker.postMessage(request);',
    '}',
    'function referenceDistanceInViewports(x: BigFixed, y: BigFixed): number {',
    '  const dx = fixedDifferenceToNumber(centerX, x);',
    '  const dy = fixedDifferenceToNumber(centerY, y);',
    '  const scale = viewportScale.mantissa * Math.pow(2, viewportScale.exponent);',
    '  return Math.hypot(dx, dy) / Math.max(scale, Number.MIN_VALUE);',
    '}',
    'function shouldActivateReference(candidate: ReferenceCache): boolean {',
    '  const candidateDistance = referenceDistanceInViewports(candidate.centerX, candidate.centerY);',
    '  if (!Number.isFinite(candidateDistance) || candidateDistance > REFERENCE_REUSE_VIEWPORTS * 8) return false;',
    '  if (!referenceCache) return true;',
    '  const currentDistance = referenceDistanceInViewports(referenceCache.centerX, referenceCache.centerY);',
    "  if (candidate.purpose === 'settled' && referenceCache.purpose === 'provisional') return true;",
    '  if (candidate.purpose === referenceCache.purpose && candidate.requestedIterations > referenceCache.requestedIterations && candidateDistance <= currentDistance + .5) return true;',
    '  return candidateDistance + .25 < currentDistance;',
    '}',
    'function handleReferenceResponse(event: MessageEvent): void {',
    '  const response = event.data as ReferenceResponse;',
    '  const meta = pendingReferences.get(response.id);',
    '  if (!meta) return;',
    '  pendingReferences.delete(response.id);',
    '  const candidateBuffer = device.createBuffer({',
    '    size: Math.max(16, response.orbit.byteLength),',
    '    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST',
    '  });',
    '  device.queue.writeBuffer(candidateBuffer, 0, response.orbit);',
    '  const candidate: ReferenceCache = {',
    '    key: meta.key,',
    '    centerX: deserializeFixed(response.referenceCenterX),',
    '    centerY: deserializeFixed(response.referenceCenterY),',
    '    requestedIterations: meta.requestedIterations,',
    '    length: response.length,',
    '    escaped: response.escaped,',
    '    bits: response.bits,',
    '    ms: response.generationMs,',
    '    purpose: meta.purpose,',
    '    buffer: candidateBuffer',
    '  };',
    '  if (!shouldActivateReference(candidate)) {',
    '    candidateBuffer.destroy();',
    '    return;',
    '  }',
    '  const previousBuffer = referenceCache?.buffer;',
    '  referenceCache = candidate;',
    "  if (candidate.purpose === 'settled') resetReferenceWorker();",
    '  if (previousBuffer) void device.queue.onSubmittedWorkDone().then(() => previousBuffer.destroy());',
    '  lastCompletedRenderKey = \'\';',
    '  requestRender();',
    '}',
    'attachReferenceWorker();',
    '',
    'function updateReadouts(): void {'
  ])
);

replaceExact(
  'active orbit buffer binding',
  '{binding:2,resource:{buffer:orbitBuffer}}',
  '{binding:2,resource:{buffer:mode === 2 && referenceCache ? referenceCache.buffer : fallbackOrbitBuffer}}'
);

replaceExact(
  'render key insertion',
  '  canvas.width = w;',
  lines([
    '  const renderKey = [',
    '    centerX.raw, centerX.bits, centerY.raw, centerY.bits,',
    '    viewportScale.mantissa, viewportScale.exponent,',
    '    w, h, quality.iterations, Number(palette.value), mode,',
    "    mode === 2 && referenceCache ? `${referenceCache.key}:${referenceCache.length}` : 'direct'",
    "  ].join('|');",
    "  if (!interactive && activeStage === 'full-quality' && pendingReferences.size === 0 && renderKey === lastCompletedRenderKey) {",
    '    frameInFlight = false;',
    '    updateReadouts();',
    '    return;',
    '  }',
    '',
    '  canvas.width = w;'
  ])
);

replaceExact(
  'completed render key',
  '  texture.destroy();\n\n  smoothedFrameMs=',
  '  texture.destroy();\n  lastCompletedRenderKey = renderKey;\n\n  smoothedFrameMs='
);

await writeFile(outputPath, source);
console.log('Generated src/mainV32.generated.ts');
