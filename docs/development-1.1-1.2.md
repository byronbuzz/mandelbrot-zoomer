# WebGPU Fractal Zoomer 1.1–1.2 implementation contract

This development line implements the missing numerical architecture identified after the first 1.0.0 browser test. It is not a visual patch for one scene.

## Governing principles

1. Avoid iterations before making iterations cheaper.
2. The persistent object is a numerical tile, not a colour image.
3. Presentation reprojection is temporary history only and never counts as numerical progress.
4. Direct arithmetic is used only while measured coordinate and orbit error remain acceptable.
5. Deep rendering uses high-precision local references and low-precision perturbation deltas.
6. Precision and repair decisions are local to tiles and use hysteresis.
7. GPU work is chunked, batched and asynchronous; diagnostics use sparse readback.
8. Colouring occurs only after numerical results have been accepted.

## 1.1.0 deliverables

- stable tile identities in fractal/world space;
- persistent per-pixel recurrence state and status;
- bounded direct iteration chunks;
- active-tile counters and active-pixel compaction where profitable;
- numerical coverage, spatial footprint and freshness metadata;
- one scheduler shared by moving, settling and settled states;
- navigation work limited by freshness targets rather than final iteration target;
- presentation history age separated from numerical sample age;
- deterministic sentinel sampling hooks;
- benchmark scenes and replayable navigation traces.

## 1.2.0 deliverables

- persistent worker/Wasm reference service;
- reference cache and per-tile reference atlas;
- four-limb GPU reference-orbit storage;
- perturbation and scaled-perturbation pipelines;
- glitch, rebase, non-finite and sentinel telemetry;
- alternate-reference, subdivision and local repair work;
- evidence-driven precision escalation with hysteresis;
- no global direct/perturbation crossover.

## Deployment gate

The development branch does not replace the live app until all of the following are true:

- the recorded `10^5.3–10^5.6` navigation path remains structurally coherent;
- stopping continues existing tile recurrence rather than restarting all pixels;
- pending reference generation never suspends navigation calculation;
- repeated presentation reprojection does not alter accepted numerical colour;
- deterministic sentinels agree with a higher-precision oracle within the configured tolerance;
- weak references affect only their assigned tiles;
- diagnostics report presentation history, numerical freshness, active pixels and precision health separately;
- CI parses every WGSL pipeline and exercises scheduler/precision invariants.
