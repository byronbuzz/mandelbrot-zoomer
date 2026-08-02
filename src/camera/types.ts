import type { BigFixed } from '../bigFixed';
import type { BinaryScale } from '../binaryScale';

export type CameraSnapshot = Readonly<{
  generation: number;
  centerX: BigFixed;
  centerY: BigFixed;
  scale: BinaryScale;
}>;
