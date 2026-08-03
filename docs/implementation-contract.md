# Frozen implementation contract

Status: **frozen**. Contract changes require a numbered decision record stating the evidence, reason, affected gates and rollback.

The normative specification is `docs/webgpu-fractal-zoomer-technical-specification-and-roadmap.md`. Its source freeze is identified by SHA-256:

`7ECBD9E9875A05413094C1C3C9AFBEB8A4CEB43AD7F51455A16D7C4AC9100F54`

## Current milestone

Phase 1 is presentation-only. It contains no Mandelbrot recurrence, reference orbit, perturbation, series approximation, BLA, precision crossover or numerical scheduler change.

The executable Phase 1 route is:

`/mandelbrot-zoomer/?mode=presentation-kernel`

The production 1.3.1 route remains the default until the presentation kernel passes its gates and a later recorded decision authorizes integration.

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
10. The default Zoomer remains working and unchanged by this milestone.

## Gates required before integration

- all source, WGSL, TypeScript and production builds pass;
- deterministic reprojection oracle passes at the stated 0.01 source-texel bound;
- a real WebGPU browser compiles pipelines and renders the fixture with zero validation or uncaptured errors;
- a navigation trace shows continuous history and tile overlay without clear holes;
- landscape/portrait/DPR resize increments the resource epoch and preserves output;
- forced `device.destroy()` recovers on a fresh device epoch;
- visual captures, diagnostics and CPU frame timing are attached to the phase PR;
- the normal 1.3.1 route receives a regression smoke test.
