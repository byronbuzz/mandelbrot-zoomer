import type { CameraSnapshot } from '../camera/types';

export type InteractionState = 'moving' | 'settling' | 'settled';
export type PrecisionMode = 'f32-direct' | 'double-float-direct';
export type FieldPhase = 'seed' | 'coarse' | 'medium' | 'fine' | 'complete';

export type RenderRequest = Readonly<{
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

export type NavigationProfile = Readonly<{
  resolutionScale: number;
  blockSize: 2 | 4 | 8;
  iterations: number;
}>;

export type FieldJob = Readonly<{
  requestId: number;
  x: number;
  y: number;
  width: number;
  height: number;
  blockSize: 1 | 2 | 4 | 8;
  iterations: number;
  acceptIterationCap: boolean;
  phase: Exclude<FieldPhase, 'seed' | 'complete'>;
  priority: number;
}>;

export type FieldStats = Readonly<{
  requestId: number;
  interaction: InteractionState;
  phase: FieldPhase;
  completedJobs: number;
  totalJobs: number;
  lastBatchMs: number;
  batchJobs: number;
  publishedJobs: number;
  precision: PrecisionMode;
  renderWidth: number;
  renderHeight: number;
  navigationResolution: number;
  navigationIterations: number;
  navigationBlockSize: number;
  fieldAgeMs: number;
  anchorGeneration: number;
}>;
