import type { CameraSnapshot } from '../camera/types';

export type TilePrecision = 'f32-direct' | 'double-float-direct' | 'perturbation' | 'scaled-perturbation';
export type TileLifecycle = 'empty' | 'queued' | 'running' | 'provisional' | 'complete' | 'failed';

export type TileKey = Readonly<{
  level: number;
  x: bigint;
  y: bigint;
}>;

export type TileHealth = Readonly<{
  coordinateErrorPixels: number;
  sentinelMismatchRate: number;
  glitchFraction: number;
  rebaseRate: number;
  nonFiniteCount: number;
}>;

export type TileFreshness = Readonly<{
  newestNumericalSampleAt: number;
  oldestVisibleNumericalSampleAt: number;
  lastPresentationReuseAt: number;
}>;

export type PersistentTile = {
  key: TileKey;
  camera: CameraSnapshot;
  width: number;
  height: number;
  precision: TilePrecision;
  lifecycle: TileLifecycle;
  completedIterations: number;
  targetIterations: number;
  activePixels: number;
  escapedPixels: number;
  interiorPixels: number;
  numericalCoverage: number;
  spatialCoverage: number;
  projectedPixelFootprint: number;
  health: TileHealth;
  freshness: TileFreshness;
  resultTexture: GPUTexture;
  colourTexture: GPUTexture;
  stateBuffer: GPUBuffer | null;
  statusBuffer: GPUBuffer | null;
  activeIndexBuffer: GPUBuffer | null;
  referenceId: number | null;
};

export function tileKeyString(key: TileKey): string {
  return `${key.level}:${key.x.toString()}:${key.y.toString()}`;
}

export function emptyTileHealth(): TileHealth {
  return {
    coordinateErrorPixels: Number.POSITIVE_INFINITY,
    sentinelMismatchRate: 1,
    glitchFraction: 1,
    rebaseRate: 0,
    nonFiniteCount: 0
  };
}

export function tileIsNumericallyAcceptable(tile: PersistentTile): boolean {
  return tile.health.nonFiniteCount === 0
    && tile.health.coordinateErrorPixels <= 0.125
    && tile.health.sentinelMismatchRate <= 0.001
    && tile.health.glitchFraction <= 0.001;
}
