# Mandelbrot Zoomer V6 architecture

V6 replaces the whole-frame rendering hierarchy with a continuously improving numerical surface.

## Release groups

### V6.0 — Progressive foundation

- parallel engine entry point (`?engine=v6`)
- exact fixed-point camera retained from V4/V5
- separate numerical and colour textures
- Mandelbrot main-cardioid and period-two-bulb rejection
- bounded tile work
- 8×8 → 4×4 → 2×2 → 1×1 spatial refinement
- visible publication after each tile
- exact current/stable-surface reprojection
- eased zoom acceleration and deceleration
- palette recolouring without orbit recomputation
- benchmark scene corpus and explicit renderer contracts

### V6.1 — Persistent iteration and deep rendering

- resumable per-pixel recurrence state
- chunked iteration tiers
- cap-hit pixels remain provisional and continue later
- active-tile and active-pixel compaction
- existing perturbation recurrence ported into tile kernels
- per-tile reference atlas
- numerical repair represented as ordinary queue work

### V6.2 — Arbitrary precision and deep acceleration

- Rust/Wasm arbitrary-precision reference service
- dynamic precision policy
- sampled high-precision sentinel validation
- series approximation
- scaled perturbation
- experimental BLA with conservative validity and exact fallback

### V6.3 — Quality and explorer features

- final-image supersampling
- automatic final iteration continuation
- palette presets, logarithmic palette length and depth-driven colour
- exact URL state, bookmarks and history
- touch gestures and high-resolution export
- live Julia preview and Mandelbrot/Julia switching
- tile-telemetry-driven autopilot

## Governing rules

1. Avoid iterations before making iterations cheaper.
2. Camera motion and fractal calculation are independent schedules.
3. A low-quality new result must not globally replace a better old result.
4. Iteration-cap hits are provisional, not mathematical interior classifications.
5. Numerical results and colour presentation are separate resources.
6. Work is scheduled by visible value, not by whole-frame completion.
7. Every GPU submission is bounded and interruptible at a scheduling boundary.
8. Precision escalates per tile from measured evidence rather than zoom depth alone.
9. Palette-only changes never rerun the orbit.
10. The final image is the converged state of the same progressive surface used during motion.

## V6.0 surface model

Each anchor owns:

- a `rgba32float` numerical texture;
- a filterable `rgba8unorm` colour texture;
- an immutable exact camera snapshot;
- a priority-ordered tile queue.

Numerical texels encode:

- smooth escape value;
- status: uncomputed, escaped, analytically interior, or iteration-cap provisional;
- executed iteration count;
- current spatial block size.

A current surface is progressively filled. The preceding surface remains stable and is exactly reprojected into the display camera. The presentation pass chooses current calculated data first and stable reprojected data where the current surface has not yet supplied a trustworthy value.

## Development policy

V5 remains available only as a regression baseline while V6 is incomplete. The live default may stay on V5 during V6.0 and early V6.1, but new architecture work must not be added to the V5 scheduler. V6 becomes the sole default when its deep perturbation path reaches parity and its progressive surface is demonstrably more stable.
