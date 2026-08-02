import { deserializeFixed, fixedAddScaled, serializeFixed } from '../bigFixed';
import type { CameraSnapshot, CpuReference, ReferenceCandidate, ReferencePurpose, ReferenceRequest, ReferenceResponse } from './types';

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
const REPAIR_LOCAL_OFFSETS = [
  [0, 0],
  [-0.035, 0], [0.035, 0], [0, -0.035], [0, 0.035],
  [-0.025, -0.025], [0.025, -0.025], [-0.025, 0.025], [0.025, 0.025]
] as const;
const GLOBAL_REPAIR_COLUMNS = 11;
const GLOBAL_REPAIR_ROWS = 9;
const GLOBAL_REPAIR_PHASES = [
  [0, 0],
  [0.5, 0.5],
  [0.25, 0.75],
  [0.75, 0.25]
] as const;
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
    purpose: Exclude<ReferencePurpose, 'repair'>,
    aspect: number,
    replaceStaleProvisional = false
  ): boolean {
    if (replaceStaleProvisional && this.pending.size > 0) {
      const oldestStartedAt = Math.min(...Array.from(this.pending.values(), pending => pending.startedAt));
      if (performance.now() - oldestStartedAt < STALE_PROVISIONAL_MS) return false;
      this.restartWorker();
    }

    const offsets = purpose === 'provisional' ? PROVISIONAL_CANDIDATE_OFFSETS : SETTLED_CANDIDATE_OFFSETS;
    const candidates = offsets.map(([dx, dy]) => this.candidateAt(camera, dx * aspect, dy));
    return this.enqueue(
      camera,
      iterations,
      purpose,
      candidates,
      purpose === 'provisional' ? 0 : Math.min(iterations, 8192),
      `${camera.generation}:${iterations}:${purpose}`
    );
  }

  requestGlobalRepair(
    camera: CameraSnapshot,
    iterations: number,
    aspect: number,
    pass: number
  ): boolean {
    const phase = GLOBAL_REPAIR_PHASES[pass % GLOBAL_REPAIR_PHASES.length];
    const candidates: ReferenceCandidate[] = [this.candidateAt(camera, 0, 0)];
    for (let row = 0; row < GLOBAL_REPAIR_ROWS; row++) {
      const normalizedY = ((row + 0.5 + phase[1]) % GLOBAL_REPAIR_ROWS) / GLOBAL_REPAIR_ROWS;
      for (let column = 0; column < GLOBAL_REPAIR_COLUMNS; column++) {
        const normalizedX = ((column + 0.5 + phase[0]) % GLOBAL_REPAIR_COLUMNS) / GLOBAL_REPAIR_COLUMNS;
        candidates.push(this.candidateAt(
          camera,
          (normalizedX - 0.5) * aspect,
          normalizedY - 0.5
        ));
      }
    }
    return this.enqueue(
      camera,
      iterations,
      'repair',
      candidates,
      Math.min(iterations, 8192),
      `${camera.generation}:${iterations}:repair:global:${pass}`
    );
  }

  requestRepair(
    camera: CameraSnapshot,
    iterations: number,
    aspect: number,
    normalizedX: number,
    normalizedY: number
  ): boolean {
    const targetX = (normalizedX - 0.5) * aspect;
    const targetY = normalizedY - 0.5;
    const candidates = REPAIR_LOCAL_OFFSETS.map(([dx, dy]) =>
      this.candidateAt(camera, targetX + dx * aspect, targetY + dy)
    );
    const key = `${camera.generation}:${iterations}:repair:tile:${normalizedX.toFixed(4)}:${normalizedY.toFixed(4)}`;
    return this.enqueue(
      camera,
      iterations,
      'repair',
      candidates,
      Math.min(iterations, 8192),
      key
    );
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

  private candidateAt(camera: CameraSnapshot, xMultiplier: number, yMultiplier: number): ReferenceCandidate {
    return {
      centerX: serializeFixed(fixedAddScaled(
        camera.centerX,
        xMultiplier * camera.scale.mantissa,
        camera.scale.exponent
      )),
      centerY: serializeFixed(fixedAddScaled(
        camera.centerY,
        yMultiplier * camera.scale.mantissa,
        camera.scale.exponent
      ))
    };
  }

  private enqueue(
    camera: CameraSnapshot,
    iterations: number,
    purpose: ReferencePurpose,
    candidates: readonly ReferenceCandidate[],
    probeIterations: number,
    key: string
  ): boolean {
    if (this.pendingKeys.has(key)) return false;
    const id = ++this.nextRequestId;
    const request: ReferenceRequest = {
      id,
      cameraGeneration: camera.generation,
      purpose,
      centerX: serializeFixed(camera.centerX),
      centerY: serializeFixed(camera.centerY),
      iterations,
      probeIterations,
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
