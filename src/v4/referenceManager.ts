import { deserializeFixed, fixedAddScaled, serializeFixed } from '../bigFixed';
import type { CameraSnapshot, CpuReference, ReferencePurpose, ReferenceRequest, ReferenceResponse } from './types';

type PendingRequest = Readonly<{
  cameraGeneration: number;
  purpose: ReferencePurpose;
  requestedIterations: number;
  key: string;
  startedAt: number;
}>;

type ReferenceListener = (reference: CpuReference) => void;

const SETTLED_CANDIDATE_OFFSETS = [
  [0, 0],
  [-0.3, 0], [0.3, 0], [0, -0.3], [0, 0.3],
  [-0.22, -0.22], [0.22, -0.22], [-0.22, 0.22], [0.22, 0.22]
] as const;
const PROVISIONAL_CANDIDATE_OFFSETS = [[0, 0]] as const;
const STALE_PROVISIONAL_MS = 260;

export class ReferenceManager {
  private worker: Worker;
  private readonly pending = new Map<number, PendingRequest>();
  private readonly pendingKeys = new Set<string>();
  private readonly listeners = new Set<ReferenceListener>();
  private nextRequestId = 0;

  constructor() {
    this.worker = this.createWorker();
  }

  get pendingCount(): number { return this.pending.size; }

  onReference(listener: ReferenceListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  request(
    camera: CameraSnapshot,
    iterations: number,
    purpose: ReferencePurpose,
    aspect: number,
    replaceStaleProvisional = false
  ): boolean {
    if (replaceStaleProvisional && this.pending.size > 0) {
      const oldestStartedAt = Math.min(...Array.from(this.pending.values(), pending => pending.startedAt));
      if (performance.now() - oldestStartedAt < STALE_PROVISIONAL_MS) return false;
      this.restartWorker();
    }

    const key = `${camera.generation}:${iterations}:${purpose}`;
    if (this.pendingKeys.has(key)) return false;
    const id = ++this.nextRequestId;
    const offsets = purpose === 'provisional' ? PROVISIONAL_CANDIDATE_OFFSETS : SETTLED_CANDIDATE_OFFSETS;
    const candidates = offsets.map(([dx, dy]) => ({
      centerX: serializeFixed(fixedAddScaled(camera.centerX, dx * aspect * camera.scale.mantissa, camera.scale.exponent)),
      centerY: serializeFixed(fixedAddScaled(camera.centerY, dy * camera.scale.mantissa, camera.scale.exponent))
    }));
    const request: ReferenceRequest = {
      id,
      cameraGeneration: camera.generation,
      purpose,
      centerX: serializeFixed(camera.centerX),
      centerY: serializeFixed(camera.centerY),
      iterations,
      probeIterations: purpose === 'provisional' ? 0 : 2048,
      candidates
    };
    this.pending.set(id, {
      cameraGeneration: camera.generation,
      purpose,
      requestedIterations: iterations,
      key,
      startedAt: performance.now()
    });
    this.pendingKeys.add(key);
    this.worker.postMessage(request);
    return true;
  }

  cancelOlderThan(cameraGeneration: number): void {
    let removed = false;
    for (const [id, pending] of this.pending) {
      if (pending.cameraGeneration >= cameraGeneration) continue;
      this.pending.delete(id);
      this.pendingKeys.delete(pending.key);
      removed = true;
    }
    if (removed) this.restartWorker();
  }

  dispose(): void {
    this.worker.terminate();
    this.pending.clear();
    this.pendingKeys.clear();
    this.listeners.clear();
  }

  private createWorker(): Worker {
    const worker = new Worker(new URL('./referenceWorker.ts', import.meta.url), { type: 'module' });
    worker.addEventListener('message', event => {
      const response = event.data as ReferenceResponse;
      const pending = this.pending.get(response.id);
      if (!pending) return;
      this.pending.delete(response.id);
      this.pendingKeys.delete(pending.key);
      if (response.cameraGeneration !== pending.cameraGeneration || response.purpose !== pending.purpose) return;
      const reference: CpuReference = {
        id: response.id,
        cameraGeneration: response.cameraGeneration,
        purpose: response.purpose,
        centerX: deserializeFixed(response.referenceCenterX),
        centerY: deserializeFixed(response.referenceCenterY),
        requestedIterations: pending.requestedIterations,
        length: response.length,
        escaped: response.escaped,
        bits: response.bits,
        generationMs: response.generationMs,
        orbit: response.orbit
      };
      for (const listener of this.listeners) listener(reference);
    });
    worker.addEventListener('error', event => {
      console.error('Reference worker failed', event.error ?? event.message);
    });
    return worker;
  }

  private restartWorker(): void {
    this.worker.terminate();
    this.pending.clear();
    this.pendingKeys.clear();
    this.worker = this.createWorker();
  }
}
