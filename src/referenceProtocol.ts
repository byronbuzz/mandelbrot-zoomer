export type SerializedFixed = { raw: string; bits: number };

export type ReferenceRequest = {
  id: number;
  centerX: SerializedFixed;
  centerY: SerializedFixed;
  iterations: number;
};

export type ReferenceResponse = {
  id: number;
  bits: number;
  length: number;
  escaped: boolean;
  generationMs: number;
  referenceCenterX: SerializedFixed;
  referenceCenterY: SerializedFixed;
  orbit: Float32Array<ArrayBuffer>;
};
