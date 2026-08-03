import type { BigFixed } from '../bigFixed';
import type { BinaryScale } from '../binaryScale';

export type RenderStage = 'interactive' | 'refining' | 'full-quality';
export type ReferencePurpose = 'provisional' | 'settled' | 'repair';
export type PrecisionMode = 'f32' | 'double-float' | 'perturbation';

export type CameraSnapshot = Readonly<{
  generation: number;
  centerX: BigFixed;
  centerY: BigFixed;
  scale: BinaryScale;
}>;

export type SerializedFixed = Readonly<{ raw: string; bits: number }>;

export type ReferenceCandidate = Readonly<{
  centerX: SerializedFixed;
  centerY: SerializedFixed;
}>;

export type ReferenceRequest = Readonly<{
  id: number;
  cameraGeneration: number;
  purpose: ReferencePurpose;
  centerX: SerializedFixed;
  centerY: SerializedFixed;
  iterations: number;
  probeIterations: number;
  candidates: readonly ReferenceCandidate[];
}>;

export type ReferenceResponse = Readonly<{
  id: number;
  cameraGeneration: number;
  purpose: ReferencePurpose;
  bits: number;
  workingBits: number;
  transportBits: number;
  contractVersion: number;
  length: number;
  escaped: boolean;
  generationMs: number;
  referenceCenterX: SerializedFixed;
  referenceCenterY: SerializedFixed;
  orbit: Float32Array<ArrayBuffer>;
}>;

export type CpuReference = Readonly<{
  id: number;
  cameraGeneration: number;
  purpose: ReferencePurpose;
  centerX: BigFixed;
  centerY: BigFixed;
  requestedIterations: number;
  length: number;
  escaped: boolean;
  bits: number;
  generationMs: number;
  orbit: Float32Array<ArrayBuffer>;
}>;

export type GpuReference = Readonly<Omit<CpuReference, 'orbit'> & {
  buffer: GPUBuffer;
}>;

export type RenderQuality = Readonly<{
  iterations: number;
  resolution: number;
}>;

export type RenderSnapshot = Readonly<{
  generation: number;
  camera: CameraSnapshot;
  stage: RenderStage;
  quality: RenderQuality;
  palettePhase: number;
  cssWidth: number;
  cssHeight: number;
  devicePixelRatio: number;
  reference: GpuReference | null;
  precision: PrecisionMode;
  repairPass: number;
}>;

export type RenderTelemetry = Readonly<{
  unresolvedPixels: number;
  exhaustedPixels: number;
  magnitudeGuardPixels: number;
  nonFinitePixels: number;
  rebaseFailurePixels: number;
  maxPerturbationExponent: number | null;
  totalPixels: number;
  tileColumns: number;
  tileRows: number;
  tileUnresolved: readonly number[];
}>;

export type PreparedFrame = Readonly<{
  snapshot: RenderSnapshot;
  texture: GPUTexture;
  computeWidth: number;
  computeHeight: number;
  displayWidth: number;
  displayHeight: number;
  computeMs: number;
  computeBatches: number;
  telemetry: RenderTelemetry | null;
  retainAsSettled: boolean;
  accumulationKey: string;
}>;
