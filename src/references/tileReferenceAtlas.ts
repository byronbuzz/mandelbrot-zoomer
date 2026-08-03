import {
  deserializeFixed,
  fixed,
  fixedAddScaled,
  fixedDifferenceToNumber,
  serializeFixed,
  type BigFixed
} from '../bigFixed';
import type { ReferenceCandidate, ReferenceRequest, ReferenceResponse } from '../v4/types';
import {
  PERSISTENT_TILE_SIZE,
  type PersistentTileDescriptor,
  type PersistentTileKey
} from '../tiles/persistentTileTypes';
import { tileSpanExponent } from '../tiles/worldTilePlanner';

const ORBIT_BYTES_PER_POINT = 8 * Float32Array.BYTES_PER_ELEMENT;
const MAX_REFERENCE_ITERATIONS = 65_536;
const REQUEST_TIMEOUT_MS = 30_000;
const REFERENCE_GROUP_TILE_SPAN = 2n;
const MAX_REUSE_COVERAGE_DISTANCE = 0.75;

const INITIAL_CANDIDATE_GRID = [-0.34, 0, 0.34] as const;
const REPAIR_CANDIDATE_GRID = [-0.45, -0.225, 0, 0.225, 0.45] as const;
const REPAIR_PHASES = [
  [0, 0],
  [0.055, 0.055],
  [-0.055, 0.055],
  [0.055, -0.055],
  [-0.055, -0.055]
] as const;

export type TileGpuReference = Readonly<{
  cacheKey: string;
  tileKey: PersistentTileKey;
  requestedIterations: number;
  repairPass: number;
  centerX: BigFixed;
  centerY: BigFixed;
  coverageCenterX: BigFixed;
  coverageCenterY: BigFixed;
  coverageExponent: number;
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
  groupCenterX: BigFixed;
  groupCenterY: BigFixed;
  groupSpanExponent: number;
  tile: PersistentTileDescriptor;
  iterations: number;
  repairPass: number;
  priority: number;
  resolve: (reference: TileGpuReference) => void;
  reject: (error: Error) => void;
};

type WorkerSlot = {
  worker: Worker;
  active: QueuedRequest | null;
  timeout: ReturnType<typeof setTimeout> | null;
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

function parseTileKey(tileKey: PersistentTileKey): {
  sampleExponent: number;
  tileX: bigint;
  tileY: bigint;
} {
  const [sampleExponentValue, tileXValue, tileYValue] = tileKey.split(':');
  return {
    sampleExponent: Number(sampleExponentValue),
    tileX: BigInt(tileXValue),
    tileY: BigInt(tileYValue)
  };
}

function groupKeyFromTileKey(tileKey: PersistentTileKey): PersistentTileKey {
  const { sampleExponent, tileX, tileY } = parseTileKey(tileKey);
  return groupKeyForParts(sampleExponent, tileX, tileY);
}

function dyadicFixed(integer: bigint, exponent: number, bits: number): BigFixed {
  const shift = bits + exponent;
  return fixed(
    shift >= 0 ? integer << BigInt(shift) : integer >> BigInt(-shift),
    bits
  );
}

function centerForTileKey(tileKey: PersistentTileKey, bits: number): readonly [BigFixed, BigFixed] {
  const { sampleExponent, tileX, tileY } = parseTileKey(tileKey);
  const halfTile = BigInt(PERSISTENT_TILE_SIZE / 2);
  return [
    dyadicFixed(tileX * BigInt(PERSISTENT_TILE_SIZE) + halfTile, sampleExponent, bits),
    dyadicFixed(tileY * BigInt(PERSISTENT_TILE_SIZE) + halfTile, sampleExponent, bits)
  ];
}

function groupGeometry(tile: PersistentTileDescriptor): {
  key: PersistentTileKey;
  centerX: BigFixed;
  centerY: BigFixed;
  spanExponent: number;
} {
  const groupX = floorDiv(tile.tileX, REFERENCE_GROUP_TILE_SPAN);
  const groupY = floorDiv(tile.tileY, REFERENCE_GROUP_TILE_SPAN);
  const groupStartX = groupX * REFERENCE_GROUP_TILE_SPAN;
  const groupStartY = groupY * REFERENCE_GROUP_TILE_SPAN;
  const tileSpan = tileSpanExponent(tile.sampleExponent);
  const offsetX = Number(groupStartX + 1n - tile.tileX) - 0.5;
  const offsetY = Number(groupStartY + 1n - tile.tileY) - 0.5;
  return {
    key: groupKeyForParts(tile.sampleExponent, tile.tileX, tile.tileY),
    centerX: fixedAddScaled(tile.centerX, offsetX, tileSpan),
    centerY: fixedAddScaled(tile.centerY, offsetY, tileSpan),
    spanExponent: tileSpan + 1
  };
}

function workerCount(): number {
  if (typeof navigator === 'undefined') return 1;
  const hardwareThreads = Math.max(1, navigator.hardwareConcurrency || 2);
  return Math.max(1, Math.min(2, Math.floor(hardwareThreads / 2)));
}

export class TileReferenceAtlas {
  private readonly workers: WorkerSlot[] = [];
  private readonly cache = new Map<string, TileGpuReference>();
  private readonly pendingByKey = new Map<string, Promise<TileGpuReference>>();
  private readonly queue: QueuedRequest[] = [];
  private nextId = 0;
  private failureCountValue = 0;

  constructor(private readonly device: GPUDevice) {
    for (let index = 0; index < workerCount(); index++) {
      const slot = { worker: null as unknown as Worker, active: null, timeout: null };
      slot.worker = this.createWorker(slot);
      this.workers.push(slot);
    }
  }

  get failureCount(): number {
    return this.failureCountValue;
  }

  get pendingCount(): number {
    return this.queue.length + this.workers.filter(slot => slot.active !== null).length;
  }

  get cacheSize(): number {
    return this.cache.size;
  }

  request(
    tile: PersistentTileDescriptor,
    iterations: number,
    priority: number,
    repairPass = 0
  ): Promise<TileGpuReference> {
    const boundedIterations = Math.max(1, Math.min(MAX_REFERENCE_ITERATIONS, Math.floor(iterations)));
    const geometry = groupGeometry(tile);
    const cacheKey = `${geometry.key}:${boundedIterations}:${repairPass}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return Promise.resolve(cached);
    const existing = this.pendingByKey.get(cacheKey);
    if (existing) return existing;

    const promise = new Promise<TileGpuReference>((resolve, reject) => {
      this.queue.push({
        id: ++this.nextId,
        cacheKey,
        groupKey: geometry.key,
        groupCenterX: geometry.centerX,
        groupCenterY: geometry.centerY,
        groupSpanExponent: geometry.spanExponent,
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
    const exactGroup = groupKeyFromTileKey(tileKey);
    let best: TileGpuReference | null = null;
    let bestScore = Number.POSITIVE_INFINITY;

    for (const reference of this.cache.values()) {
      if (reference.length < 2) continue;
      const [tileCenterX, tileCenterY] = centerForTileKey(tileKey, reference.coverageCenterX.bits);
      const coverageSpan = Math.pow(2, reference.coverageExponent);
      if (!Number.isFinite(coverageSpan) || coverageSpan <= 0) continue;
      const dx = fixedDifferenceToNumber(tileCenterX, reference.coverageCenterX) / coverageSpan;
      const dy = fixedDifferenceToNumber(tileCenterY, reference.coverageCenterY) / coverageSpan;
      const coverageDistance = Math.max(Math.abs(dx), Math.abs(dy));
      if (!Number.isFinite(coverageDistance) || coverageDistance > MAX_REUSE_COVERAGE_DISTANCE) continue;

      const sameGroupBonus = reference.tileKey === exactGroup ? -0.5 : 0;
      const iterationPenalty = reference.requestedIterations >= iterations ? 0 : 0.35;
      const escapedPenalty = reference.escaped ? 0.04 : 0;
      const score = coverageDistance + iterationPenalty + escapedPenalty + sameGroupBonus;
      if (
        score < bestScore
        || (score === bestScore && reference.length > (best?.length ?? 0))
      ) {
        best = reference;
        bestScore = score;
      }
    }
    return best;
  }

  dispose(): void {
    for (const slot of this.workers) {
      slot.worker.terminate();
      if (slot.timeout) clearTimeout(slot.timeout);
      slot.timeout = null;
      slot.active?.reject(new Error('Reference atlas disposed'));
      slot.active = null;
    }
    for (const queued of this.queue.splice(0)) queued.reject(new Error('Reference atlas disposed'));
    for (const reference of this.cache.values()) reference.buffer.destroy();
    this.cache.clear();
    this.pendingByKey.clear();
  }

  private pump(): void {
    for (const slot of this.workers) {
      if (slot.active || this.queue.length === 0) continue;
      const request = this.queue.shift();
      if (!request) continue;
      this.start(slot, request);
    }
  }

  private start(slot: WorkerSlot, request: QueuedRequest): void {
    slot.active = request;
    slot.timeout = setTimeout(() => {
      if (slot.active?.id !== request.id) return;
      this.failureCountValue++;
      request.reject(new Error(`Tile reference generation exceeded ${REQUEST_TIMEOUT_MS} ms`));
      slot.active = null;
      slot.timeout = null;
      this.restartWorker(slot);
      this.pump();
    }, REQUEST_TIMEOUT_MS);

    const workerRequest: ReferenceRequest = {
      id: request.id,
      cameraGeneration: request.id,
      purpose: 'settled',
      centerX: serializeFixed(request.groupCenterX),
      centerY: serializeFixed(request.groupCenterY),
      iterations: request.iterations,
      probeIterations: Math.min(request.iterations, request.repairPass === 0 ? 1024 : 4096),
      maxTransportBits: 96,
      candidates: this.candidates(request)
    };
    slot.worker.postMessage(workerRequest);
  }

  private candidates(request: QueuedRequest): readonly ReferenceCandidate[] {
    const phase = REPAIR_PHASES[request.repairPass % REPAIR_PHASES.length] ?? REPAIR_PHASES[0];
    const grid = request.repairPass === 0 ? INITIAL_CANDIDATE_GRID : REPAIR_CANDIDATE_GRID;
    const result: ReferenceCandidate[] = [];
    const seen = new Set<string>();
    for (const y of grid) {
      for (const x of grid) {
        const centerX = fixedAddScaled(
          request.groupCenterX,
          x + phase[0],
          request.groupSpanExponent
        );
        const centerY = fixedAddScaled(
          request.groupCenterY,
          y + phase[1],
          request.groupSpanExponent
        );
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

  private createWorker(slot: WorkerSlot): Worker {
    const worker = new Worker(new URL('../v4/referenceWorker.ts', import.meta.url), { type: 'module' });
    worker.addEventListener('message', event => this.onMessage(slot, event.data as ReferenceResponse));
    worker.addEventListener('error', event => this.onWorkerError(slot, event.message));
    return worker;
  }

  private onWorkerError(slot: WorkerSlot, message: string): void {
    if (slot.timeout) clearTimeout(slot.timeout);
    slot.timeout = null;
    const active = slot.active;
    if (active) {
      this.failureCountValue++;
      active.reject(new Error(message || 'Reference worker failed'));
    }
    slot.active = null;
    this.restartWorker(slot);
    this.pump();
  }

  private onMessage(slot: WorkerSlot, response: ReferenceResponse): void {
    const active = slot.active;
    if (!active || response.id !== active.id) return;
    if (slot.timeout) clearTimeout(slot.timeout);
    slot.timeout = null;
    slot.active = null;

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

      // A short escaped orbit is still a valid local reference. Pixels that outlive it
      // are identified by the perturbation kernel and repaired with another local reference.
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
        coverageCenterX: active.groupCenterX,
        coverageCenterY: active.groupCenterY,
        coverageExponent: active.groupSpanExponent,
        length: response.length,
        escaped: response.escaped,
        bits: response.bits,
        generationMs: response.generationMs,
        buffer
      };
      this.cache.set(active.cacheKey, reference);
      active.resolve(reference);
    } catch (error) {
      this.failureCountValue++;
      active.reject(error instanceof Error ? error : new Error(String(error)));
    } finally {
      this.pump();
    }
  }

  private restartWorker(slot: WorkerSlot): void {
    slot.worker.terminate();
    slot.worker = this.createWorker(slot);
  }
}
