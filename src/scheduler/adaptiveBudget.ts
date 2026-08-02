import type { NavigationProfile } from '../tiles/types';

const TARGET_BATCH_MS = 68;
const MIN_NAVIGATION_ITERATIONS = 96;
const INITIAL_NAVIGATION_ITERATIONS = 192;
const MIN_RESOLUTION_SCALE = 0.62;
const MAX_RESOLUTION_SCALE = 1;

export class AdaptiveNavigationBudget {
  private iterations = INITIAL_NAVIGATION_ITERATIONS;
  private resolutionScale = 1;
  private blockSize: 2 | 4 | 8 = 4;
  private smoothedMs = TARGET_BATCH_MS;

  profile(targetIterations: number): NavigationProfile {
    return {
      resolutionScale: Math.max(MIN_RESOLUTION_SCALE, Math.min(MAX_RESOLUTION_SCALE, this.resolutionScale)),
      blockSize: this.blockSize,
      iterations: Math.max(
        Math.min(targetIterations, MIN_NAVIGATION_ITERATIONS),
        Math.min(targetIterations, Math.round(this.iterations))
      )
    };
  }

  observe(batchMs: number, targetIterations: number): void {
    if (!(batchMs > 0) || !Number.isFinite(batchMs)) return;
    this.smoothedMs = this.smoothedMs * 0.72 + batchMs * 0.28;
    const minimumIterations = Math.min(targetIterations, MIN_NAVIGATION_ITERATIONS);

    if (this.smoothedMs > TARGET_BATCH_MS * 1.18) {
      const ratio = TARGET_BATCH_MS / this.smoothedMs;
      if (this.iterations > minimumIterations + 8) {
        this.iterations = Math.max(minimumIterations, Math.floor(this.iterations * Math.max(0.55, ratio)));
        return;
      }
      if (this.blockSize < 8) {
        this.blockSize = this.blockSize === 2 ? 4 : 8;
        return;
      }
      this.resolutionScale = Math.max(
        MIN_RESOLUTION_SCALE,
        this.resolutionScale * Math.max(0.82, Math.sqrt(ratio))
      );
      return;
    }

    if (this.smoothedMs < TARGET_BATCH_MS * 0.62) {
      if (this.resolutionScale < MAX_RESOLUTION_SCALE - 0.01) {
        this.resolutionScale = Math.min(MAX_RESOLUTION_SCALE, this.resolutionScale * 1.08 + 0.01);
        return;
      }
      if (this.blockSize > 2) {
        this.blockSize = this.blockSize === 8 ? 4 : 2;
        return;
      }
      if (this.iterations < targetIterations) {
        this.iterations = Math.min(targetIterations, Math.ceil(Math.max(this.iterations + 24, this.iterations * 1.16)));
      }
    }
  }
}
