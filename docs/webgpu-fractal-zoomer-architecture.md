# WebGPU Fractal Zoomer architecture

## Product identity

The application is **WebGPU Fractal Zoomer**. Versioning follows `major.minor.fix`:

- first digit: major architecture/product generation;
- second digit: coherent development stage;
- third digit: fixes within that stage.

The previous Mandelbrot Zoomer implementation is preserved on the `legacy/mandelbrot-zoomer-2026-08-03` branch. The active application does not import its renderer, scheduler, presentation or reference lifecycle.

## Governing model

The renderer is a persistent numerical field viewed through a continuously moving camera. It is not a sequence of replacement frames.

1. The exact fixed-point camera is authoritative.
2. Presentation runs independently at display cadence.
3. The visible field is always complete: a new anchor is seeded by GPU reprojection of the previous field before it becomes visible.
4. New calculations overwrite only pixels for which they have a useful result.
5. Iteration-cap results remain provisional during navigation and therefore retain reprojected history.
6. Numerical data, colour data and quality metadata are separate resources.
7. GPU work is bounded, coalesced and batched; obsolete requests are abandoned at batch boundaries.
8. Precision is selected from coordinate distinguishability and measured tile evidence, never from one global zoom threshold.
9. Deep rendering will use local references and local repair; no single reference decides the whole viewport.
10. The stopped image converges through the same field used during navigation.

## 1.0.0 — persistent-field foundation

- exact fixed-point camera and zoom-about-pointer navigation;
- WebGPU compute as the primary backend;
- native `f32` and emulated double-float direct pipelines;
- analytic main-cardioid and period-two-bulb rejection;
- smooth escape colouring after escape;
- separate `rgba32float` numerical, `rgba8unorm` colour and `rgba8unorm` quality textures;
- GPU reprojection seeds every new field before publication;
- provisional navigation pixels preserve history rather than becoming black or translucent;
- explicit moving, settling and settled states;
- adaptive navigation iteration, block-size and resolution budget;
- progressive 8×8 → 4×4 → 2×2 → 1×1 refinement;
- several tile jobs encoded per command buffer with one completion fence;
- bounded texture-resource pool;
- device-loss and runtime-error reporting;
- diagnostics that expose the field lifecycle and adaptive budget.

## 1.1.0 — persistent iteration and active work

- resumable direct recurrence state;
- fixed-size iteration chunks;
- active-tile reduction;
- active-pixel masks, compaction and indirect dispatch where profitable;
- periodicity checks as a specialised pipeline;
- GPU-resident completion counters with sparse asynchronous readback;
- benchmark-driven chunk and workgroup selection.

## 1.2.0 — local deep perturbation

- persistent worker/Wasm reference service;
- reference cache and per-tile reference atlas;
- four-limb GPU reference orbits;
- perturbation, scaled perturbation and rebasing pipelines;
- tile health from glitch flags, rebase rate, numerical range and sentinel disagreement;
- local subdivision, alternate-reference and repair jobs;
- precision escalation with hysteresis;
- no whole-viewport direct/perturbation crossover.

## 1.3.0 — deep acceleration and verification

- arbitrary-precision Wasm reference engine;
- dynamic reference precision;
- series approximation with conservative remainder bounds;
- experimental BLA with exact fallback;
- deterministic high-precision sentinel validation;
- device-profile tuning and sustained mobile thermal telemetry.

## Admission policy for earlier work

Retained mathematics or techniques must be small, independently testable and compatible with the persistent-field model. The fixed-point camera, binary-scale representation, double-float primitives, analytic interior tests and exact camera transforms qualify. Whole-frame current/stable lifecycle code, alpha-as-confidence, hard depth crossovers, single-reference viewport policy and per-job host synchronisation do not.
