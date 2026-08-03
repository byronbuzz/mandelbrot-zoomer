import type { InteractionState } from '../tiles/types';

export const REFERENCE_CONTRACT_VERSION = 2;
export const LEGACY_REFERENCE_TRANSPORT_BITS = 96;
export const WIDE_REFERENCE_TRANSPORT_BITS = 192;
export const REFERENCE_TRANSPORT_GUARD_BITS = 32;
export const PERTURBATION_OVERLAP_SAMPLE_EXPONENT = -23;
export const PREFERRED_REFERENCE_SEED_BUDGET = 16;
export const MAX_PENDING_REFERENCE_DEMAND = 16;

export type PrecisionDecision = Readonly<{
  required: boolean;
  preferred: boolean;
  reason: 'coordinate-collapse' | 'non-finite' | 'deep-overlap' | 'direct';
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
    && input.levelOffset === 0
    && input.sampleExponent <= PERTURBATION_OVERLAP_SAMPLE_EXPONENT
    && Number.isFinite(input.distanceFromFocus);
  return {
    required,
    preferred,
    reason: input.coordinateCollapsed
      ? 'coordinate-collapse'
      : input.nonFinitePixels > 0
        ? 'non-finite'
        : preferred ? 'deep-overlap' : 'direct',
    requiredTransportBits: requiredReferenceTransportBits(input.sampleExponent)
  };
}
