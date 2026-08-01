import {
  fixedAddScaled,
  fixedFromNumber,
  fixedRescale,
  requiredCoordinateBits,
  type BigFixed
} from '../bigFixed';
import {
  normalizeScale,
  scaleDeltaParts,
  scaleLog2,
  scaleLog10,
  scaleMultiply,
  type BinaryScale
} from '../binaryScale';
import type { CameraSnapshot } from './types';

export class CameraModel {
  private generationValue = 1;
  private coordinateBitsValue = 160;
  private centerXValue: BigFixed = fixedFromNumber(-0.5, this.coordinateBitsValue);
  private centerYValue: BigFixed = fixedFromNumber(0, this.coordinateBitsValue);
  private scaleValue: BinaryScale = normalizeScale(0.75, 2);

  get generation(): number { return this.generationValue; }
  get coordinateBits(): number { return this.coordinateBitsValue; }
  get scale(): BinaryScale { return this.scaleValue; }

  snapshot(): CameraSnapshot {
    return {
      generation: this.generationValue,
      centerX: this.centerXValue,
      centerY: this.centerYValue,
      scale: this.scaleValue
    };
  }

  log10Magnification(): number {
    return Math.log10(3) - scaleLog10(this.scaleValue);
  }

  zoomAbout(normalizedX: number, normalizedY: number, factor: number, aspect: number): void {
    const xMultiplier = (normalizedX - 0.5) * aspect * (1 - factor);
    const yMultiplier = (normalizedY - 0.5) * (1 - factor);
    this.moveByScaleMultipliers(xMultiplier, yMultiplier);
    this.scaleValue = scaleMultiply(this.scaleValue, factor);
    this.ensurePrecision();
    this.generationValue++;
  }

  panByCssPixels(deltaX: number, deltaY: number, cssHeight: number): void {
    if (!(cssHeight > 0)) return;
    this.moveByScaleMultipliers(-deltaX / cssHeight, -deltaY / cssHeight);
    this.ensurePrecision();
    this.generationValue++;
  }

  private moveByScaleMultipliers(xMultiplier: number, yMultiplier: number): void {
    const dx = scaleDeltaParts(this.scaleValue, xMultiplier);
    const dy = scaleDeltaParts(this.scaleValue, yMultiplier);
    this.centerXValue = fixedAddScaled(this.centerXValue, dx.mantissa, dx.exponent);
    this.centerYValue = fixedAddScaled(this.centerYValue, dy.mantissa, dy.exponent);
  }

  private ensurePrecision(): void {
    const needed = requiredCoordinateBits(-scaleLog2(this.scaleValue) + Math.log2(3));
    if (needed <= this.coordinateBitsValue) return;
    this.coordinateBitsValue = needed;
    this.centerXValue = fixedRescale(this.centerXValue, needed);
    this.centerYValue = fixedRescale(this.centerYValue, needed);
  }
}
