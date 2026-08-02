import {
  deserializeFixed,
  fixedAddScaled,
  serializeFixed
} from '../bigFixed';
import type { CameraSnapshot, ReferenceRequest, ReferenceResponse } from '../v4/types';
import type { ProgressiveSurfaceSnapshot } from './types';

const OFFSETS = [-0.4, -0.2, 0, 0.2, 0.4] as const;

function align4(value: number): number {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

export type DeepReference = Readonly<{
  buffer: GPUBuffer;
  centerX: CameraSnapshot['centerX'];
  centerY: CameraSnapshot['centerY'];
  length: number;
  bits: number;
  generationMs: number;
}>;

export class ReferenceService {
  private worker: Worker | null = null;
  private requestId = 0;
  private cancelPending: (() => void) | null = null;

  constructor(private readonly device: GPUDevice) {}

  cancel(): void {
    this.worker?.terminate();
    this.worker = null;
    this.cancelPending?.();
    this.cancelPending = null;
  }

  request(snapshot: ProgressiveSurfaceSnapshot, width: number, height: number): Promise<DeepReference> {
    this.cancel();
    const id = ++this.requestId;
    const worker = new Worker(new URL('../v4/referenceWorker.ts', import.meta.url), { type: 'module' });
    this.worker = worker;

    return new Promise<DeepReference>((resolve, reject) => {
      let settled = false;
      const finish = (): boolean => {
        if (settled) return false;
        settled = true;
        if (this.worker === worker) this.worker = null;
        this.cancelPending = null;
        worker.terminate();
        return true;
      };
      this.cancelPending = () => {
        if (finish()) reject(new Error('Reference request superseded'));
      };
      worker.addEventListener('error', event => {
        if (finish()) reject(new Error(event.message || 'Reference worker failed'));
      });
      worker.addEventListener('message', event => {
        const response = event.data as ReferenceResponse;
        if (response.id !== id || response.cameraGeneration !== snapshot.generation) return;
        if (!finish()) return;
        try {
          const buffer = this.device.createBuffer({
            size: align4(response.orbit.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
          });
          this.device.queue.writeBuffer(buffer, 0, response.orbit);
          resolve({
            buffer,
            centerX: deserializeFixed(response.referenceCenterX),
            centerY: deserializeFixed(response.referenceCenterY),
            length: response.length,
            bits: response.bits,
            generationMs: response.generationMs
          });
        } catch (error) {
          reject(error);
        }
      });

      const request: ReferenceRequest = {
        id,
        cameraGeneration: snapshot.generation,
        purpose: 'settled',
        centerX: serializeFixed(snapshot.camera.centerX),
        centerY: serializeFixed(snapshot.camera.centerY),
        iterations: snapshot.iterations,
        probeIterations: Math.min(snapshot.iterations, 3072),
        candidates: this.candidates(snapshot, width, height)
      };
      worker.postMessage(request);
    });
  }

  private candidates(
    snapshot: ProgressiveSurfaceSnapshot,
    width: number,
    height: number
  ): ReferenceRequest['candidates'] {
    const camera = snapshot.camera;
    const aspect = width / Math.max(1, height);
    const result: ReferenceRequest['candidates'][number][] = [];
    const seen = new Set<string>();
    const add = (x: number, y: number): void => {
      const centerX = fixedAddScaled(
        camera.centerX,
        camera.scale.mantissa * x * aspect,
        camera.scale.exponent
      );
      const centerY = fixedAddScaled(
        camera.centerY,
        camera.scale.mantissa * y,
        camera.scale.exponent
      );
      const sx = serializeFixed(centerX);
      const sy = serializeFixed(centerY);
      const key = `${sx.raw}:${sy.raw}`;
      if (seen.has(key)) return;
      seen.add(key);
      result.push({ centerX: sx, centerY: sy });
    };
    add(0, 0);
    for (const y of OFFSETS) for (const x of OFFSETS) add(x, y);
    return result;
  }
}
