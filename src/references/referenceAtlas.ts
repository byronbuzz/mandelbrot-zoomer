import type { BigFixed } from '../bigFixed';
import type { PersistentTile, TileKey } from '../tiles/persistentTile';

export type ReferenceHealth = Readonly<{
  usableIterations: number;
  escaped: boolean;
  generationMs: number;
  validationMismatchRate: number;
}>;

export type ReferenceOrbit = {
  id: number;
  centerX: BigFixed;
  centerY: BigFixed;
  bits: number;
  length: number;
  buffer: GPUBuffer;
  health: ReferenceHealth;
  tileKeys: Set<string>;
  lastUsedAt: number;
};

export type ReferenceRequest = Readonly<{
  tileKey: TileKey;
  centerX: BigFixed;
  centerY: BigFixed;
  iterations: number;
  minimumBits: number;
}>;

export type ReferenceFactory = (request: ReferenceRequest) => Promise<ReferenceOrbit>;

function tileKeyString(key: TileKey): string {
  return `${key.level}:${key.x.toString()}:${key.y.toString()}`;
}

export class ReferenceAtlas {
  private readonly references = new Map<number, ReferenceOrbit>();
  private readonly tileAssignments = new Map<string, number>();
  private readonly pending = new Map<string, Promise<ReferenceOrbit>>();

  constructor(
    private readonly createReference: ReferenceFactory,
    private readonly maxReferences = 32
  ) {}

  assigned(tile: PersistentTile): ReferenceOrbit | null {
    const id = this.tileAssignments.get(tileKeyString(tile.key));
    if (id === undefined) return null;
    const reference = this.references.get(id) ?? null;
    if (reference) reference.lastUsedAt = performance.now();
    return reference;
  }

  async ensure(tile: PersistentTile, minimumBits: number): Promise<ReferenceOrbit> {
    const key = tileKeyString(tile.key);
    const assigned = this.assigned(tile);
    if (assigned
        && assigned.bits >= minimumBits
        && assigned.health.usableIterations >= tile.targetIterations
        && assigned.health.validationMismatchRate <= 0.001) {
      return assigned;
    }

    const inFlight = this.pending.get(key);
    if (inFlight) return inFlight;

    const request: ReferenceRequest = {
      tileKey: tile.key,
      centerX: tile.camera.centerX,
      centerY: tile.camera.centerY,
      iterations: tile.targetIterations,
      minimumBits
    };
    const promise = this.createReference(request).then(reference => {
      this.references.set(reference.id, reference);
      reference.tileKeys.add(key);
      this.tileAssignments.set(key, reference.id);
      this.pending.delete(key);
      this.evictUnused();
      return reference;
    }, error => {
      this.pending.delete(key);
      throw error;
    });
    this.pending.set(key, promise);
    return promise;
  }

  releaseTile(key: TileKey): void {
    const serialized = tileKeyString(key);
    const id = this.tileAssignments.get(serialized);
    this.tileAssignments.delete(serialized);
    if (id !== undefined) this.references.get(id)?.tileKeys.delete(serialized);
  }

  dispose(): void {
    for (const reference of this.references.values()) reference.buffer.destroy();
    this.references.clear();
    this.tileAssignments.clear();
    this.pending.clear();
  }

  private evictUnused(): void {
    if (this.references.size <= this.maxReferences) return;
    const candidates = [...this.references.values()]
      .filter(reference => reference.tileKeys.size === 0)
      .sort((left, right) => left.lastUsedAt - right.lastUsedAt);
    while (this.references.size > this.maxReferences && candidates.length > 0) {
      const reference = candidates.shift();
      if (!reference) break;
      reference.buffer.destroy();
      this.references.delete(reference.id);
    }
  }
}
