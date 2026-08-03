import type { InteractionState } from '../tiles/types';
import type { PersistentTile } from '../tiles/persistentTile';

export type TileWorkKind =
  | 'seed'
  | 'direct-chunk'
  | 'compact-active'
  | 'request-reference'
  | 'perturbation-chunk'
  | 'validate-sentinels'
  | 'recolour';

export type TileWork = Readonly<{
  tile: PersistentTile;
  kind: TileWorkKind;
  priority: number;
  iterationBudget: number;
}>;

const NAVIGATION_MAX_CHUNK = 128;
const SETTLING_MAX_CHUNK = 256;
const SETTLED_MAX_CHUNK = 512;

function iterationChunk(interaction: InteractionState, remaining: number): number {
  const cap = interaction === 'moving'
    ? NAVIGATION_MAX_CHUNK
    : interaction === 'settling' ? SETTLING_MAX_CHUNK : SETTLED_MAX_CHUNK;
  return Math.max(0, Math.min(cap, remaining));
}

function numericalUrgency(tile: PersistentTile): number {
  const missing = 1 - Math.min(tile.numericalCoverage, tile.spatialCoverage);
  const conditioning = Math.min(4, tile.health.coordinateErrorPixels * 2)
    + Math.min(4, tile.health.sentinelMismatchRate * 1000)
    + Math.min(4, tile.health.glitchFraction * 1000)
    + Math.min(2, tile.health.nonFiniteCount);
  return missing * 8 + conditioning;
}

function freshnessUrgency(tile: PersistentTile, now: number): number {
  const newestAge = Math.max(0, now - tile.freshness.newestNumericalSampleAt);
  const oldestAge = Math.max(0, now - tile.freshness.oldestVisibleNumericalSampleAt);
  return Math.min(4, newestAge / 250) + Math.min(4, oldestAge / 1000);
}

export function chooseTileWork(
  tile: PersistentTile,
  interaction: InteractionState,
  now = performance.now()
): TileWork | null {
  const remaining = Math.max(0, tile.targetIterations - tile.completedIterations);
  const priority = numericalUrgency(tile)
    + freshnessUrgency(tile, now)
    + Math.min(8, Math.max(0, tile.projectedPixelFootprint - 1));

  if (tile.lifecycle === 'empty') {
    return { tile, kind: 'seed', priority: priority + 20, iterationBudget: 0 };
  }

  if (tile.health.nonFiniteCount > 0
      || tile.health.sentinelMismatchRate > 0.001
      || tile.health.coordinateErrorPixels > 0.125) {
    if (tile.precision === 'f32-direct' || tile.precision === 'double-float-direct') {
      return { tile, kind: 'request-reference', priority: priority + 30, iterationBudget: 0 };
    }
    return { tile, kind: 'validate-sentinels', priority: priority + 25, iterationBudget: 0 };
  }

  if (remaining > 0 && tile.activePixels > 0) {
    const kind = tile.precision === 'perturbation' || tile.precision === 'scaled-perturbation'
      ? 'perturbation-chunk'
      : 'direct-chunk';
    return { tile, kind, priority: priority + 10, iterationBudget: iterationChunk(interaction, remaining) };
  }

  if (tile.activePixels > tile.width * tile.height * 0.08 && tile.activeIndexBuffer === null) {
    return { tile, kind: 'compact-active', priority: priority + 4, iterationBudget: 0 };
  }

  return null;
}

export function orderTileWork(work: readonly TileWork[]): TileWork[] {
  return [...work].sort((left, right) => right.priority - left.priority);
}
