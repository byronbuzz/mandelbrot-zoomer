# Frozen implementation contract

Status: **frozen**. Contract changes require a numbered decision record stating the evidence, reason, affected gates and rollback.

The normative specification is `docs/webgpu-fractal-zoomer-technical-specification-and-roadmap.md`. Its source freeze is identified by SHA-256:

`7ECBD9E9875A05413094C1C3C9AFBEB8A4CEB43AD7F51455A16D7C4AC9100F54`

## Completed production milestone

Phase 1 and its 1.4 production integration are presentation-only. They contain no Mandelbrot recurrence, reference orbit, perturbation, series approximation, BLA, precision crossover or numerical scheduler change.

The executable Phase 1 route is:

`/mandelbrot-zoomer/?mode=presentation-kernel`

Decision 0002 authorizes the gated production integration as release 1.4. The atlas-history presenter is the default; `?presenter=legacy` preserves the prior presentation path for exact comparison and rollback.

## Current evidence milestone

Decision 0003 authorizes a test-only numerical contract and benchmark harness before numerical optimization begins. It imports the unchanged production direct WGSL into a standalone, canvas-free WebGPU fixture and compares stable shallow cases against an independent CPU-f64 oracle. It changes no renderer, scheduler, presentation, recurrence, precision-selection, perturbation or reference behavior.

This milestone must pass before Phase 3 optimization work starts:

- frozen direct semantics and explicit status/index encodings;
- strict CPU goldens and chunk-invariance properties;
- direct production-WGSL browser execution without the app or presenter;
- exact discrete GPU/oracle agreement on stable fixtures;
- zero compilation, scoped, uncaptured, console or device-loss errors;
- raw adapter/environment/correctness JSON and separately labelled performance telemetry.

## Phase 1 invariants

1. The last accepted viewport is stored in an offscreen `rgba8unorm` history texture with immutable source-view, size and epoch metadata.
2. Navigation first reprojects the accepted anchor, then overlays current accepted tiles. Reprojected output never becomes numerical state.
3. A/B history textures prevent read/write feedback. Promotion occurs only after complete direct fixture coverage.
4. All visible tiles are emitted by one instanced overlay draw with half-open grid ownership.
5. Reprojection uniforms are admitted only when f32 packing adds at most 0.01 source texel of coordinate error at the target centre and pixel-centre corners.
6. Canvas acquisition, command encoding and submission are synchronous. No promise suspension occurs after `getCurrentTexture()` and before `queue.submit()`.
7. Zero physical size suspends acquisition and submission. Positive size creates a new resource epoch.
8. Resize keeps the old anchor sampleable through the first new-size submission, then retires it after a completion watermark.
9. Device loss stops use of the old device, requests a fresh adapter and device, rebuilds all GPU resources, and resumes with preserved CPU view state.
10. The numerical Zoomer remains working and numerically unchanged by this milestone.

## Gates required before integration

- all source, WGSL, TypeScript and production builds pass;
- deterministic reprojection oracle passes at the stated 0.01 source-texel bound;
- a real WebGPU browser compiles pipelines and renders the fixture with zero validation or uncaptured errors;
- a navigation trace shows continuous history and tile overlay without clear holes;
- landscape/portrait/DPR resize increments the resource epoch and preserves output;
- forced `device.destroy()` recovers on a fresh device epoch;
- visual captures, diagnostics and CPU frame timing are attached to the phase PR;
- the normal 1.3.1 route receives a regression smoke test.
