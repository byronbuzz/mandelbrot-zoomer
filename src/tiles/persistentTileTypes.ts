import type { BigFixed } from '../bigFixed';
import type { CameraSnapshot } from '../camera/types';
import type { InteractionState } from './types';

export const PERSISTENT_TILE_SIZE = 128;
export const PERSISTENT_TILE_BITS = 7;

export type PersistentTileKey = `${number}:${string}:${string}`;

export type PersistentTileDescriptor = Readonly<{
  key: PersistentTileKey;
  sampleExponent: number;
  tileX: bigint;
  tileY: bigint;
  centerX: BigFixed;
  centerY: BigFixed;
  distanceFromFocus: number;
}>;

export type PersistentTileRequest = Readonly<{
  requestId: number;
  camera: CameraSnapshot;
  cssWidth: number;
  cssHeight: number;
  devicePixelRatio: number;
  targetIterations: number;
  palettePhase: number;
  focusX: number;
  focusY: number;
  interaction: InteractionState;
}>;

export type PersistentTileWork = Readonly<{
  requestId: number;
  key: PersistentTileKey;
  targetIterations: number;
  chunkIterations: number;
  acceptIterationCap: boolean;
  priority: number;
}>;

export type PersistentTileHealth = Readonly<{
  activePixels: number;
  escapedPixels: number;
  analyticInteriorPixels: number;
  cappedPixels: number;
  nonFinitePixels: number;
}>;

export type PersistentFieldStats = Readonly<{
  requestId: number;
  interaction: InteractionState;
  visibleTiles: number;
  cachedTiles: number;
  activeTiles: number;
  convergedTiles: number;
  completedChunks: number;
  queuedChunks: number;
  lastBatchMs: number;
  numericalFreshnessMs: number;
  presentationHistoryMs: number;
  sampleExponent: number;
  tileSize: number;
}>;
