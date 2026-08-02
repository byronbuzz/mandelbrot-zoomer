import type { RenderQuality } from '../v4/types';

export const TARGET_ANCHOR_FRAME_MS = 100;
export const MIN_NAVIGATION_RESOLUTION = 0.5;

const INITIAL_NAVIGATION_ITERATIONS = 2048;
const MIN_NAVIGATION_ITERATIONS = 384;

export class NavigationQualityController {
  private adaptiveResolution = 1;
  private adaptiveIterations = INITIAL_NAVIGATION_ITERATIONS;
  private smoothedFrameMs = TARGET_ANCHOR_FRAME_MS;
  private completedFps = 0;

  reset(requestedIterations: number): void {
    this.adaptiveResolution = 1;
    this.adaptiveIterations = Math.min(requestedIterations, INITIAL_NAVIGATION_ITERATIONS);
    this.smoothedFrameMs = TARGET_ANCHOR_FRAME_MS;
    this.completedFps = 0;
  }

  quality(requestedIterations: number, maximumResolution: number): RenderQuality {
    const minimumIterations = Math.min(requestedIterations, MIN_NAVIGATION_ITERATIONS);
    const iterations = Math.max(
      minimumIterations,
      Math.min(requestedIterations, Math.round(this.adaptiveIterations))
    );
    const resolution = Math.max(
      MIN_NAVIGATION_RESOLUTION,
      Math.min(maximumResolution, this.adaptiveResolution)
    );
    return { iterations, resolution };
  }

  observeCompleted(
    frameMs: number,
    quality: RenderQuality,
    requestedIterations: number,
    maximumResolution: number
  ): void {
    this.smoothedFrameMs = this.smoothedFrameMs * 0.72 + frameMs * 0.28;
    this.completedFps = 1000 / Math.max(1, this.smoothedFrameMs);

    const minimumIterations = Math.min(requestedIterations, MIN_NAVIGATION_ITERATIONS);
    if (frameMs > TARGET_ANCHOR_FRAME_MS * 1.12) {
      const ratio = TARGET_ANCHOR_FRAME_MS / Math.max(1, frameMs);
      if (quality.iterations > minimumIterations) {
        const factor = Math.max(0.4, Math.min(0.9, ratio * 0.96));
        this.adaptiveIterations = Math.max(minimumIterations, Math.floor(quality.iterations * factor));
      } else {
        this.adaptiveResolution = Math.max(
          MIN_NAVIGATION_RESOLUTION,
          quality.resolution * Math.max(0.75, Math.sqrt(ratio))
        );
      }
      return;
    }

    if (frameMs < TARGET_ANCHOR_FRAME_MS * 0.68) {
      if (quality.resolution < maximumResolution - 0.01) {
        const ratio = TARGET_ANCHOR_FRAME_MS / Math.max(1, frameMs);
        this.adaptiveResolution = Math.min(
          1,
          quality.resolution * Math.min(1.12, Math.sqrt(ratio)) + 0.015
        );
      } else if (quality.iterations < requestedIterations) {
        const factor = Math.min(1.24, TARGET_ANCHOR_FRAME_MS / Math.max(1, frameMs));
        this.adaptiveIterations = Math.min(
          requestedIterations,
          Math.ceil(Math.max(quality.iterations + 32, quality.iterations * factor))
        );
      }
    }
  }

  observeDropped(quality: RenderQuality, requestedIterations: number): void {
    const minimumIterations = Math.min(requestedIterations, MIN_NAVIGATION_ITERATIONS);
    if (quality.iterations > minimumIterations) {
      this.adaptiveIterations = Math.max(
        minimumIterations,
        Math.floor(quality.iterations * 0.62)
      );
    } else {
      this.adaptiveResolution = Math.max(
        MIN_NAVIGATION_RESOLUTION,
        quality.resolution * 0.86
      );
    }
  }

  get anchorFps(): number { return this.completedFps; }
  get navigationIterations(): number { return Math.round(this.adaptiveIterations); }
}

export class PresentationRateMeter {
  private readonly timestamps: number[] = [];

  note(now = performance.now()): void {
    this.timestamps.push(now);
    this.trim(now);
  }

  rate(now = performance.now()): number {
    this.trim(now);
    return this.timestamps.length;
  }

  private trim(now: number): void {
    const cutoff = now - 1000;
    while (this.timestamps.length > 0 && (this.timestamps[0] ?? now) < cutoff) {
      this.timestamps.shift();
    }
  }
}
