import type { PreparedFrame, RenderSnapshot, RenderTelemetry } from './types';
import type { WebGpuRenderer } from './webGpuRenderer';

export type PresentedFrame = Readonly<{
  snapshot: RenderSnapshot;
  computeMs: number;
  presentMs: number;
  telemetry: RenderTelemetry | null;
}>;

export class RenderCoordinator {
  private latest: RenderSnapshot | null = null;
  private running = false;

  constructor(
    private readonly renderer: WebGpuRenderer,
    private readonly currentGeneration: () => number,
    private readonly onPresented: (frame: PresentedFrame) => void,
    private readonly onError: (error: unknown) => void,
    private readonly onIdle: () => void
  ) {}

  get isBusy(): boolean { return this.running; }

  request(snapshot: RenderSnapshot): void {
    this.latest = snapshot;
    this.renderer.reproject(snapshot);
    if (!this.running) void this.pump();
  }

  private async pump(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.latest) {
        const snapshot = this.latest;
        this.latest = null;
        let frame: PreparedFrame | null = null;
        try {
          frame = await this.renderer.prepare(snapshot);
          if (snapshot.generation !== this.currentGeneration()) {
            this.renderer.discard(frame);
            frame = null;
            continue;
          }
          const computeMs = frame.computeMs;
          const telemetry = frame.telemetry;
          const presentMs = await this.renderer.present(frame);
          frame = null;
          if (this.latest) this.renderer.reproject(this.latest);
          this.onPresented({ snapshot, computeMs, presentMs, telemetry });
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
