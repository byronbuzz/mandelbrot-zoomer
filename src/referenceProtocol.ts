export type ReferenceRequest = {
  id: number;
  centerX: { raw: string; bits: number };
  centerY: { raw: string; bits: number };
  iterations: number;
};

export type ReferenceResponse = {
  id: number;
  bits: number;
  length: number;
  escaped: boolean;
  generationMs: number;
  orbit: Float32Array;
};
