import {
  deserializeFixed,
  fixedAddScaled,
  serializeFixed,
  type BigFixed
} from '../bigFixed';
import type { ReferenceCandidate, ReferenceRequest, ReferenceResponse } from '../v4/types';
import type { PersistentTileDescriptor, PersistentTileKey } from '../tiles/persistentTileTypes';
import { tileSpanExponent } from '../tiles/worldTilePlanner';

const ORBIT_BYTES_PER_POINT = 8 * Float32Array.BYTES_PER_ELEMENT;
const MAX_REFERENCE_ITERATIONS = 65_536;
const REQUEST_TIMEOUT_MS = 30_000;
const REFERENCE_GROUP_TILE_SPAN = 2n;

const INITIAL_CANDIDATE_GRID = [-0.25, 0, 0.25] as const;
const REPAIR_CANDIDATE_GRID = [-0.4, -0.2, 0, 0.2, 0.4] as const;
const REPAIR_PHASES = [
  [0, 0],
  [0.0625, 0.0625],
  [-0.0625, 0.0625],
  [0.0625, -0.0625],
  [-0.0625, -0.0625]
] as const;

export type TileGpuReference = Readonly<{
  cacheKey: string;
  tileKey: PersistentTileKey;
  requestedIterations: number;
  repairPass: number;
  centerX: BigFixed;
  centerY: BigFixed;
  length: number;
  escaped: boolean;
  bits: number;
  generationMs: number;
  buffer: GPUBuffer;
}>;

type QueuedRequest = {
  id: number;
  cacheKey: string;
  groupKey: PersistentTileKey;
  tile: PersistentTileDescriptor;
  iterations: number;
  repairPass: number;
  priority: number;
  resolve: (reference: TileGpuReference) => void;
  reject: (error: Error) => void;
};

function floorDiv(value: bigint, divisor: bigint): bigint {
  const quotient = value / divisor;
  const remainder = value % divisor;
  return remainder < 0n ? quotient - 1n : quotient;
}

function groupKeyForParts(
  sampleExponent: number,
  tileX: bigint,
  tileY: bigint
): PersistentTileKey {
  return `${sampleExponent}:${floorDiv(tileX, REFERENCE_GROUP_TILE_SPAN).toString()}:${floorDiv(tileY, REFERENCE_GROUP_TILE_SPAN).toString()}`;
}

function groupKeyForTile(tile: PersistentTileDescriptor): PersistentTileKey {
  return groupKeyForParts(tile.sampleExponent, tile.tileX, tile.tileY);
}

function groupKeyFromTileKey(tileKey: PersistentTileKey): PersistentTileKey {
  const [sampleExponentValue, tileXValue, tileYValue] = tileKey.split(':');
  const sampleExponent = Number(sampleExponentValue);
  return groupKeyForParts(sampleExponent, BigInt(tileXValue), BigInt(tileYValue));
}

export class TileReferenceAtlas {
  private worker: Worker;
  private readonly cache = new Map<string, TileGpuReference>();
  private readonly pendingByKey = new Map<string, Promise<TileGpuReference>>();
  private readonly queue: QueuedRequest[] = [];
  private active: QueuedRequest | null = null;
  private timeout: ReturnType<typeof setTimeout> | null = null;
  private nextId = 0;

  constructor(private readonly device: GPUDevice) {
    this.worker = this.createWorker();
  }

  request(
    tile: PersistentTileDescriptor,
    iterations: number,
    priority: number,
    repairPass = 0
  ): Promise<TileGpuReference> {
    const boundedIterations = Math.max(1, Math.min(MAX_REFERENCE_ITERATIONS, Math.floor(iterations)));
    const groupKey = groupKeyForTile(tile);
    const cacheKey = `${groupKey}:${boundedIterations}:${repairPass}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return Promise.resolve(cached);
    const existing = this.pendingByKey.get(cacheKey);
    if (existing) return existing;

    const promise = new Promise<TileGpuReference>((resolve, reject) => {
      this.queue.push({
        id: ++this.nextId,
        cacheKey,
        groupKey,
        tile,
        iterations: boundedIterations,
        repairPass,
        priority,
        resolve,
        reject
      });
      this.queue.sort((left, right) => left.priority - right.priority);
      this.pump();
    });
    this.pendingByKey.set(cacheKey, promise);
    void promise.then(
      () => this.pendingByKey.delete(cacheKey),
      () => this.pendingByKey.delete(cacheKey)
    );
    return promise;
  }

  findReusable(
    tileKey: PersistentTileKey,
    iterations: number
  ): TileGpuReference | null {
    const groupKey = groupKeyFromTileKey(tileKey);
    let best: TileGpuReference | null = null;
    for (const reference of this.cache.values()) {
      if (reference.tileKey !== groupKey) continue;
      if (reference.length < iterations + 1 || reference.escaped) continue;
      if (!best || reference.requestedIterations < best.requestedIterations) best = reference;
    }
    return best;
  }

  dispose(): void {
    this.worker.terminate();
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = null;
    this.active?.reject(new Error('Reference atlas disposed'));
    this.active = null;
    for (const queued of this.queue.splice(0)) queued.reject(new Error('Reference atlas disposed'));
    for (const reference of this.cache.values()) reference.buffer.destroy();
    this.cache.clear();
    this.pendingByKey.clear();
  }

  private pump(): void {
    if (this.active || this.queue.length === 0) return;
    const request = this.queue.shift();
    if (!request) return;
    this.active = request;
    this.timeout = setTimeout(() => {
      if (this.active?.id !== request.id) return;
      request.reject(new Error(`Tile reference generation exceeded ${REQUEST_TIMEOUT_MS} ms`));
      this.active = null;
      this.restartWorker();
      this.pump();
    }, REQUEST_TIMEOUT_MS);

    const workerRequest: ReferenceRequest = {
      id: request.id,
      cameraGeneration: request.id,
      purpose: 'settled',
      centerX: serializeFixed(request.tile.centerX),
      centerY: serializeFixed(request.tile.centerY),
      iterations: request.iterations,
      probeIterations: Math.min(request.iterations, request.repairPass === 0 ? 1024 : 4096),
      candidates: this.candidates(request.tile, request.repairPass)
    };
    this.worker.postMessage(workerRequest);
  }

  private candidates(
    tile: PersistentTileDescriptor,
    repairPass: number
  ): readonly ReferenceCandidate[] {
    const spanExponent = tileSpanExponent(tile.sampleExponent);
    const phase = REPAIR_PHASES[repairPass % REPAIR_PHASES.length] ?? REPAIR_PHASES[0];
    const grid = repairPass === 0 ? INITIAL_CANDIDATE_GRID : REPAIR_CANDIDATE_GRID;
    const result: ReferenceCandidate[] = [];
    const seen = new Set<string>();
    for (const y of grid) {
      for (const x of grid) {
        const centerX = fixedAddScaled(tile.centerX, x + phase[0], spanExponent);
        const centerY = fixedAddScaled(tile.centerY, y + phase[1], spanExponent);
        const serializedX = serializeFixed(centerX);
        const serializedY = serializeFixed(centerY);
        const key = `${serializedX.raw}:${serializedY.raw}`;
        if (seen.has(key)) continue;
        seen.add(key);
        result.push({ centerX: serializedX, centerY: serializedY });
      }
    }
    return result;
  }

  private createWorker(): Worker {
    const worker = new Worker(new URL('../v4/referenceWorker.ts', import.meta.url), { type: 'module' });
    worker.addEventListener('message', event => this.onMessage(event.data as ReferenceResponse));
    worker.addEventListener('error', event => {
      if (this.timeout) clearTimeout(this.timeout);
      this.timeout = null;
      const active = this.active;
      if (active) active.reject(new Error(event.message || 'Reference worker failed'));
      this.active = null;
      this.restartWorker();
      this.pump();
    });
    return worker;
  }

  private onMessage(response: ReferenceResponse): void {
    const active = this.active;
    if (!active || response.id !== active.id) return;
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = null;
    this.active = null;

    try {
      if (!(response.orbit instanceof Float32Array)) {
        throw new Error('Reference worker returned an invalid orbit payload');
      }
      if (!Number.isInteger(response.length) || response.length < 2) {
        throw new Error('Reference worker returned an empty orbit');
      }
      const expectedBytes = response.length * ORBIT_BYTES_PER_POINT;
      if (response.orbit.byteLength < expectedBytes) {
        throw new Error('Reference orbit payload is shorter than its reported length');
      }
      if (response.escaped || response.length < active.iterations + 1) {
        throw new Error(
          `No full-length local reference survived ${active.iterations} iterations for group ${active.groupKey}`
        );
      }
      const buffer = this.device.createBuffer({
        size: Math.max(32, response.orbit.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this.device.queue.writeBuffer(buffer, 0, response.orbit);
      const reference: TileGpuReference = {
        cacheKey: active.cacheKey,
        tileKey: active.groupKey,
        requestedIterations: active.iterations,
        repairPass: active.repairPass,
        centerX: deserializeFixed(response.referenceCenterX),
        centerY: deserializeFixed(response.referenceCenterY),
        length: response.length,
        escaped: response.escaped,
        bits: response.bits,
        generationMs: response.generationMs,
        buffer
      };
      this.cache.set(active.cacheKey, reference);
      active.resolve(reference);
    } catch (error) {
      active.reject(error instanceof Error ? error : new Error(String(error)));
    } finally {
      this.pump();
    }
  }

  private restartWorker(): void {
    this.worker.terminate();
    this.worker = this.createWorker();
  }
}
