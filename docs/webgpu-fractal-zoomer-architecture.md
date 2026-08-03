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
5. Spatial resolution and iteration depth are independent refinement axes.
6. Iteration-cap results remain provisional during navigation unless a numerically suitable pipeline reaches the settled target.
7. Recurrence state, numerical results, colour data and quality metadata are separate resources.
8. GPU work is bounded, coalesced and batched; obsolete requests are abandoned at batch boundaries.
9. Interaction state controls scheduling latency and priority, not a global numerical iteration ceiling.
10. Precision is selected from coordinate distinguishability, iteration demand and measured tile health, never one global zoom threshold.
11. Deep rendering uses local reference groups and genuinely tile-local repair; no single reference decides the whole viewport.
12. Moving, settling and settled states advance the same cached tile population.
13. A stopped view must converge to an authoritative lattice whose sample pitch is no larger than one display pixel.
14. Provisional coverage may improve navigation but must never be counted as numerical convergence.
15. Refinement is presentation-monotonic: unresolved work, precision escalation and repair may replace accepted display samples only with newer accepted samples, never with holes.
16. Spatial levels are admitted lazily: a finer lattice is created and scheduled only after the current coarser level drains and no newer camera request supersedes it.

## 1.0.0 — greenfield presentation prototype

1.0.0 established the new product, exact camera, explicit interaction states, independent display-rate presentation and adaptive direct rendering. Its browser tests also exposed an important architectural error: the persistent object was reprojected colour rather than numerical state. That implementation remains valuable as a prototype but is superseded by later stages.

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
- separate numerical-freshness and presentation-history telemetry.

The 1.1 implementation proved the numerical state model, but its production scheduler still selected one lattice per interaction state and retained too much host synchronisation.

## 1.2.0 — local deep perturbation

Implemented deep-path components:

- persistent high-precision reference workers and GPU orbit cache;
- references shared by small local tile groups rather than the whole viewport;
- finite short and escaped references admitted as useful local segments;
- four-limb reference-orbit representation;
- specialised direct and perturbation compute pipelines;
- direct work bounded conservatively while a local reference is pending;
- per-pixel perturbation continuation across bounded chunks;
- cancellation-glitch, orbit-exhaustion and non-finite health signals;
- alternate reference candidates and bounded repair passes;
- repair resets unresolved pixels while preserving accepted escaped/interior samples;
- weak or failed references affect only their local tiles;
- no depth-triggered viewport-wide direct/perturbation transition.

The 1.2 browser traces exposed that the scheduler—not only the arithmetic—was preventing these components from operating as one coherent field: global moving iteration caps, one-lattice requests, tiny synchronised batches, an undersampled settled lattice and group-scoped repair all violated the governing model.

## 1.3.0 — progressive spatial and iteration field

1.3.0 is the architectural completion of the interactive field scheduler:

- one stable numerical resolution target across moving, settling and settled states;
- an explicit persistent lattice pyramid, with coarse coverage followed by intermediate and authoritative pixel-level tiles;
- no global moving or settling iteration ceiling;
- every tile retains the user iteration target and advances through bounded chunks as compute time permits;
- provisional coarse and moving iteration-cap coverage is publishable without being counted as convergence;
- adaptive multi-tile GPU quanta replace fixed four-tile request bursts;
- scheduler yields occur between bounded quanta rather than after every logical tile decision;
- the authoritative level-zero lattice uses a sample pitch no larger than one display pixel;
- nearest sampling presents the finest accepted lattice without a final linear-filter blur;
- initial references remain amortised over small groups, while repair references are tile-local;
- reference working precision is raised from coordinate/scale demand rather than inherited blindly from the camera storage width;
- WebGPU resource creation is guarded by validation scopes, and host/shader uniform layouts are CI invariants.

Per-batch diagnostic readback remains temporarily present in 1.3.0. It is amortised over larger adaptive batches; GPU-resident active compaction and indirect continuation remain a measured optimisation rather than a prerequisite for correctness.

## 1.3.1 — monotonic refinement and lazy admission

The first production trace of 1.3.0 exposed three implementation mismatches inside the otherwise correct stage architecture:

- every double-float tile was escalated immediately to perturbation, even while adjacent sample coordinates remained distinguishable in the double-float representation;
- all three spatial levels were allocated and queued at once instead of being admitted progressively;
- numerical reset and unresolved shader paths erased accepted quality data, so precision escalation and repair created rectangular presentation holes.

1.3.1 corrects those mismatches without adding a new precision threshold or viewport fallback:

- double-float direct remains a complete numerical tier and escalates only when exact adjacent-coordinate splitting collapses or measured tile health fails;
- coarse, intermediate and authoritative levels are admitted one at a time, with newer camera requests allowed to supersede finer admission;
- resource creation and clearing are batched and yield to presentation between batches;
- recurrence resets do not clear result, colour or quality resources;
- active, failed and unaccepted-cap paths preserve prior accepted presentation until a newer accepted result is available;
- coverage accounting is monotonic while authoritative convergence remains independently tracked;
- fully covered child tiles suppress redundant parent draws, reducing per-frame command and uniform traffic.

## 1.4.0 — deep acceleration and verification

Planned:

- arbitrary-precision Wasm reference engine;
- series approximation with conservative remainder bounds;
- true scaled perturbation where ordinary deltas approach exponent limits;
- experimental BLA with exact fallback;
- deterministic high-precision sentinel validation;
- active-pixel compaction and indirect dispatch where profiling demonstrates a win;
- GPU-resident health reduction that removes ordinary per-batch host readback;
- device-profile tuning and sustained mobile thermal telemetry.

## Admission policy for earlier work

Retained mathematics or techniques must be small, independently testable and compatible with the persistent-field model. The fixed-point camera, binary-scale representation, double-float primitives, analytic interior tests and multi-limb reference arithmetic qualify. Whole-frame current/stable lifecycle code, colour-as-numerical-persistence, alpha-as-confidence, hard depth crossovers, single-reference viewport policy and per-job host synchronisation do not.
