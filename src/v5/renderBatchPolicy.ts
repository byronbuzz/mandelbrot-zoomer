const WORKGROUP_HEIGHT = 8;
const INITIAL_BATCH_ROWS = 64;
const MIN_BATCH_ROWS = WORKGROUP_HEIGHT;
const MAX_BATCH_ROWS = 256;
const TARGET_BATCH_MS = 36;

export class RenderBatchPolicy {
  private rows = INITIAL_BATCH_ROWS;

  reset(iterations: number, width: number): void {
    const nominalRows = Math.floor(420_000_000 / Math.max(1, iterations * width));
    this.rows = this.align(Math.max(MIN_BATCH_ROWS, Math.min(MAX_BATCH_ROWS, nominalRows)));
  }

  currentRows(remainingRows: number): number {
    return Math.min(remainingRows, this.rows);
  }

  observe(elapsedMs: number): void {
    if (!Number.isFinite(elapsedMs) || elapsedMs <= 0) return;
    const scale = Math.sqrt(TARGET_BATCH_MS / elapsedMs);
    const bounded = Math.max(0.6, Math.min(1.5, scale));
    this.rows = this.align(Math.max(MIN_BATCH_ROWS, Math.min(MAX_BATCH_ROWS, this.rows * bounded)));
  }

  private align(value: number): number {
    return Math.max(MIN_BATCH_ROWS, Math.floor(value / WORKGROUP_HEIGHT) * WORKGROUP_HEIGHT);
  }
}
