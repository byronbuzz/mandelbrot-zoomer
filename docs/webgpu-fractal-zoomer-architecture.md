# WebGPU Fractal Zoomer architecture

## Product identity

The application is **WebGPU Fractal Zoomer**. Versioning follows `major.minor.fix`:

- first digit: major architecture/product generation;
- second digit: coherent development stage;
- third digit: fixes within that stage.

The previous Mandelbrot Zoomer implementation is preserved on the `legacy/mandelbrot-zoomer-2026-08-03` branch. The active application does not import its renderer, scheduler, presentation or reference lifecycle.

## Governing model

The renderer is a persistent numerical tile field viewed through a continuously moving camera. It is not a sequence of replacement frames.

1. The exact fixed-point camera is authoritative.
2. Presentation runs independently at display cadence.
3. Numerical state belongs to exact dyadic world-space tiles, not viewport frames.
4. Camera movement reprojects accepted tile samples for presentation without pretending that colour reprojection is numerical persistence.
5. Iteration-cap results remain provisional during navigation unless a numerically suitable pipeline reaches the settled target.
6. Recurrence state, numerical results, colour data and quality metadata are separate resources.
7. GPU work is bounded, coalesced and batched; obsolete requests are abandoned at batch boundaries.
8. Precision is selected from coordinate distinguishability, iteration demand and measured tile health, never one global zoom threshold.
9. Deep rendering uses local reference groups and local repair; no single reference decides the whole viewport.
10. Moving, settling and settled states advance the same cached tile population.

## 1.0.0 — greenfield presentation prototype

1.0.0 established the new product, exact camera, explicit interaction states, independent display-rate presentation and adaptive direct rendering. Its browser tests also exposed an important architectural error: the persistent object was reprojected colour rather than numerical state. That implementation remains valuable as a prototype but is superseded by 1.1.0 and 1.2.0.

## 1.1.0 — persistent numerical tiles

Implemented foundations:

- exact dyadic world-space tile identities independent of camera anchors;
- tile-local result, recurrence, metadata, quality and colour resources;
- resumable native-f32 and double-float direct recurrence;
- fixed-size iteration chunks;
- escaped and analytically interior pixels remain completed across later chunks;
- active, capped and non-finite counters consumed by the scheduler;
- focus-prioritised active-tile work queues;
- bounded tile cache with coarse and fine levels coexisting;
- accepted-sample coverage controls composition instead of colour brightness;
- separate numerical-freshness and presentation-history telemetry;
- no static full-viewport 8×8 → 4×4 → 2×2 → 1×1 restart ladder.

Active-pixel compaction and indirect dispatch remain performance work for a later stage; the numerical state model no longer depends on them for correctness.

## 1.2.0 — local deep perturbation

Implemented deep path:

- persistent high-precision reference worker and GPU orbit cache;
- references shared by small local tile groups rather than the whole viewport;
- four-limb reference-orbit representation;
- specialised direct and perturbation compute pipelines;
- direct work is bounded conservatively while a local reference is pending;
- full-length reference admission before a tile enters perturbation mode;
- per-pixel perturbation continuation across bounded chunks;
- cancellation-glitch, orbit-exhaustion and non-finite health signals;
- alternate local reference candidates and bounded repair passes;
- repair resets unresolved pixels while preserving accepted escaped/interior samples;
- weak or failed references affect only their local tiles;
- no depth-triggered viewport-wide direct/perturbation transition.

## 1.3.0 — deep acceleration and verification

Planned:

- arbitrary-precision Wasm reference engine;
- dynamic reference precision;
- series approximation with conservative remainder bounds;
- scaled perturbation where ordinary deltas approach exponent limits;
- experimental BLA with exact fallback;
- deterministic high-precision sentinel validation;
- active-pixel compaction and indirect dispatch where profiling demonstrates a win;
- device-profile tuning and sustained mobile thermal telemetry.

## Admission policy for earlier work

Retained mathematics or techniques must be small, independently testable and compatible with the persistent-field model. The fixed-point camera, binary-scale representation, double-float primitives, analytic interior tests and multi-limb reference arithmetic qualify. Whole-frame current/stable lifecycle code, colour-as-numerical-persistence, alpha-as-confidence, hard depth crossovers, single-reference viewport policy and per-job host synchronisation do not.
