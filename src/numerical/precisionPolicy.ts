import type { InteractionState } from '../tiles/types';

export const REFERENCE_CONTRACT_VERSION = 1;
export const REFERENCE_TRANSPORT_GUARD_BITS = 32;
export const PERTURBATION_OVERLAP_SAMPLE_EXPONENT = -23;
export const PREFERRED_REFERENCE_TILE_BUDGET = 4;

export type PrecisionDecision = Readonly<{
  required: boolean;
  preferred: boolean;
  reason: 'coordinate-collapse' | 'non-finite' | 'focus-overlap' | 'direct';
  requiredTransportBits: number;
}>;

export function requiredReferenceTransportBits(sampleExponent: number): number {
  return Math.max(48, -Math.floor(sampleExponent) + REFERENCE_TRANSPORT_GUARD_BITS);
}

export function precisionDecision(input: Readonly<{
  sampleExponent: number;
  interaction: InteractionState;
  levelOffset: number;
  distanceFromFocus: number;
  coordinateCollapsed: boolean;
  nonFinitePixels: number;
  doubleFloat: boolean;
}>): PrecisionDecision {
  const required = input.coordinateCollapsed || input.nonFinitePixels > 0;
  const preferred = !required
    && input.doubleFloat
    && input.interaction === 'settled'
    && input.levelOffset === 0
    && input.sampleExponent <= PERTURBATION_OVERLAP_SAMPLE_EXPONENT
    && input.distanceFromFocus <= 1.5;
  return {
    required,
    preferred,
    reason: input.coordinateCollapsed
      ? 'coordinate-collapse'
      : input.nonFinitePixels > 0
        ? 'non-finite'
        : preferred ? 'focus-overlap' : 'direct',
    requiredTransportBits: requiredReferenceTransportBits(input.sampleExponent)
  };
}
