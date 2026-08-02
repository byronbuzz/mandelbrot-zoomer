import type { ProgressivePhase, ProgressiveTileJob } from './types';

const DEFAULT_TILE_SIZE = 96;
const MOVING_BLOCK_SIZES = [8, 4] as const;
const SETTLED_BLOCK_SIZES = [8, 4, 2, 1] as const;

export class ProgressiveTileScheduler {
  private jobs: ProgressiveTileJob[] = [];
  private nextIndex = 0;
  private completed = 0;

  reset(
    generation: number,
    width: number,
    height: number,
    iterations: number,
    focusX: number,
    focusY: number,
    motionPressure: number,
    tileSize = DEFAULT_TILE_SIZE
  ): void {
    const safeWidth = Math.max(1, Math.floor(width));
    const safeHeight = Math.max(1, Math.floor(height));
    const safeTileSize = Math.max(32, Math.floor(tileSize / 8) * 8);
    const blocks = motionPressure > 0.2 ? MOVING_BLOCK_SIZES : SETTLED_BLOCK_SIZES;
    const focusPixelX = Math.min(1, Math.max(0, focusX)) * safeWidth;
    const focusPixelY = Math.min(1, Math.max(0, focusY)) * safeHeight;
    const diagonal = Math.max(1, Math.hypot(safeWidth, safeHeight));
    const queued: ProgressiveTileJob[] = [];

    for (let level = 0; level < blocks.length; level++) {
      const blockSize = blocks[level];
      for (let y = 0; y < safeHeight; y += safeTileSize) {
        const tileHeight = Math.min(safeTileSize, safeHeight - y);
        for (let x = 0; x < safeWidth; x += safeTileSize) {
          const tileWidth = Math.min(safeTileSize, safeWidth - x);
          const centerX = x + tileWidth * 0.5;
          const centerY = y + tileHeight * 0.5;
          const distance = Math.hypot(centerX - focusPixelX, centerY - focusPixelY) / diagonal;
          const edgePenalty = Math.min(
            centerX / safeWidth,
            centerY / safeHeight,
            1 - centerX / safeWidth,
            1 - centerY / safeHeight
          );
          queued.push({
            generation,
            x,
            y,
            width: tileWidth,
            height: tileHeight,
            blockSize,
            iterations,
            priority: level * 1_000_000 + distance * 10_000 - edgePenalty * 100
          });
        }
      }
    }

    queued.sort((left, right) => left.priority - right.priority);
    this.jobs = queued;
    this.nextIndex = 0;
    this.completed = 0;
  }

  next(): ProgressiveTileJob | null {
    const job = this.jobs[this.nextIndex];
    if (!job) return null;
    this.nextIndex++;
    return job;
  }

  markCompleted(): void {
    this.completed = Math.min(this.jobs.length, this.completed + 1);
  }

  get pendingJobs(): number { return Math.max(0, this.jobs.length - this.nextIndex); }
  get completedJobs(): number { return this.completed; }
  get totalJobs(): number { return this.jobs.length; }
  get hasWork(): boolean { return this.nextIndex < this.jobs.length; }

  get phase(): ProgressivePhase {
    const next = this.jobs[this.nextIndex];
    if (!next) return 'complete';
    if (next.blockSize >= 8) return 'coarse';
    if (next.blockSize >= 4) return 'medium';
    return 'fine';
  }
}
