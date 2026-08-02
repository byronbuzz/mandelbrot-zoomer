import type { CameraSnapshot } from '../v4/types';

export type ProgressivePhase = 'coarse' | 'medium' | 'fine' | 'complete';

export type ProgressiveTileJob = Readonly<{
  generation: number;
  x: number;
  y: number;
  width: number;
  height: number;
  blockSize: 1 | 2 | 4 | 8;
  iterations: number;
  priority: number;
}>;

export type ProgressiveSurfaceSnapshot = Readonly<{
  generation: number;
  camera: CameraSnapshot;
  cssWidth: number;
  cssHeight: number;
  devicePixelRatio: number;
  iterations: number;
  palettePhase: number;
  focusX: number;
  focusY: number;
  motionPressure: number;
}>;

export type ProgressiveRendererStats = Readonly<{
  phase: ProgressivePhase;
  pendingJobs: number;
  completedJobs: number;
  totalJobs: number;
  lastBlockSize: number;
  lastTileMs: number;
  tileRate: number;
  anchorGeneration: number;
  analyticInteriorEnabled: boolean;
}>;

export type BenchmarkScene = Readonly<{
  id: string;
  label: string;
  centerX: number;
  centerY: number;
  viewportHeight: number;
  iterations: number;
  purpose: string;
}>;
