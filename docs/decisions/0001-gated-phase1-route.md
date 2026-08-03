# Decision 0001: isolate Phase 1 behind a deployed route

Date: 2026-08-03

## Decision

Implement and deploy the presentation kernel at `?mode=presentation-kernel` while leaving 1.3.1 as the default route.

## Reason

The specification forbids combining presentation refactoring with new numerical work and requires every PR to preserve a working Zoomer. The current renderer cannot exercise presentation without numerical tiles and has notification-only device loss. A standalone deterministic fixture makes persistent history, reprojection, overlay, resize and recovery executable and reviewable without changing fractal mathematics or destabilizing the production path.

## Consequences

- Phase 1 can be tested on real WebGPU independently of numerical scheduling.
- The draft PR is deployable for evidence while the public default remains 1.3.1.
- Integration into the numerical renderer requires a later decision after all Phase 1 gates pass.
