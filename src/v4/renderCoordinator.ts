import type { PreparedFrame, RenderSnapshot, RenderTelemetry } from './types';
import type { WebGpuRenderer } from './webGpuRenderer';

const INTERACTIVE_FRAME_DEADLINE_MS = 160;

export type PresentedFrame = Readonly<{
  snapshot: RenderSnapshot;
  computeMs: number;
  computeBatches: number;
  presentMs: number;
  telemetry: RenderTelemetry | null;
}>;

export type DroppedInteractiveFrame = Readonly<{
  snapshot: RenderSnapshot;
  elapsedMs: number;
}>;

export class RenderCoordinator {
  private latest: RenderSnapshot | null = null;
  private running = false;

  constructor(
    private readonly renderer: WebGpuRenderer,
    private readonly currentGeneration: () => number,
    private readonly onPresented: (frame: PresentedFrame) => void,
    private readonly onInteractiveDropped: (frame: DroppedInteractiveFrame) => void,
    private readonly onError: (error: unknown) => void,
    private readonly onIdle: () => void
  ) {}

  get isBusy(): boolean { return this.running; }

  request(snapshot: RenderSnapshot): boolean {
    this.latest = snapshot;
    const reprojected = this.renderer.reproject(snapshot);
    if (!this.running) void this.pump();
    return reprojected;
  }

  private async pump(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.latest) {
        const snapshot = this.latest;
        this.latest = null;
        let frame: PreparedFrame | null = null;
        const started = performance.now();
        try {
          frame = await this.renderer.prepare(
            snapshot,
            () => {
              if (snapshot.generation !== this.currentGeneration()) return true;
              if (snapshot.stage !== 'interactive') return false;
              const newerCameraWaiting = Boolean(
                this.latest
                && this.latest.camera.generation !== snapshot.camera.generation
              );
              return newerCameraWaiting
                && performance.now() - started >= INTERACTIVE_FRAME_DEADLINE_MS;
            }
          );
          if (!frame) {
            if (
              snapshot.stage === 'interactive'
              && snapshot.generation === this.currentGeneration()
            ) {
              this.onInteractiveDropped({
                snapshot,
                elapsedMs: Math.max(0.1, performance.now() - started)
              });
            }
            continue;
          }
          if (snapshot.generation !== this.currentGeneration()) {
            this.renderer.discard(frame);
            frame = null;
            continue;
          }
          const computeMs = frame.computeMs;
          const computeBatches = frame.computeBatches;
          const telemetry = frame.telemetry;
          const presentMs = await this.renderer.present(frame);
          frame = null;
          if (this.latest) this.renderer.reproject(this.latest);
          this.onPresented({ snapshot: frame?.snapshot ?? snapshot, computeMs, computeBatches, presentMs, telemetry });
        } catch (error) {
          if (frame) this.renderer.discard(frame);
          this.onError(error);
        }
      }
    } finally {
      this.running = false;
      this.onIdle();
      if (this.latest) void this.pump();
    }
  }
}
