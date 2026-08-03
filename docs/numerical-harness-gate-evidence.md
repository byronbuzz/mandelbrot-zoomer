# Numerical contract and benchmark harness gate evidence

Date: 2026-08-03 (Australia/Sydney)  
Decision: `docs/decisions/0003-numerical-evidence-before-optimization.md`

## Scope

This milestone adds evidence infrastructure only. No file under `src/` changed. The renderer, presentation kernel, scheduler, direct and perturbation WGSL, reference service, precision-selection policy, recurrence state, palette path and lockfile remain unchanged.

The browser fixture imports only `src/numerical/tileDirectShader.ts`. It creates a private adapter/device, storage buffers and offscreen storage textures. It has no canvas and does not import or execute the app, presenter, scheduler, colour shader, reference service or perturbation shader.

## Frozen direct contract

- quadratic Mandelbrot recurrence with `z0 = 0`;
- inclusive main-cardioid and period-two-bulb analytic certificates;
- strict `|z|^2 > 256` escape condition;
- radius check before each recurrence update;
- iteration count equals completed updates;
- an escape observed exactly at the selected target wins over a provisional cap;
- pixel centres use `centre + (index + 0.5 - tileSize/2) * 2^sampleExponent`;
- recurrence metadata, accepted-result evidence and accepted quality must agree;
- smooth tolerance is `max(0.001, 8 f32 ULP)`.

The tested source commit is `ac9b38ed600e15fbe04ded7b1b4109cd379eea8a`. SHA-256 identities:

- fixture: `50dbc8e2b80520fb3448ef079e32ebe97863f3ab176f32155a418153901fe943`;
- unchanged production direct shader: `6f2b4110f3f1f790660bdb99d6116c86ac6dc8f3a59ab5e36ae6157955b63ec1`;
- browser runner: `268e6f03504ab3aea6831200613891b4657263431227aba5e1182fc0b30050b8`;
- WebGPU harness: `3270a81a8b73985263d476b9232cd1bc717941eede09b384267b81f87708db64`;
- CPU oracle contract: `2b1f71669d69a62203db93cefd5b63f76a0b011e31b9b3d23ccc5fb1958496c9`.

## Portable CPU and compile gates

- 16 strict stable cases passed their checked-in goldens and the executable oracle, including target zero;
- monolithic, repeated-one, repeated-seven, repeated-64, Fibonacci and escape-boundary chunk schedules produced identical oracle state;
- one 16x16 exact-dyadic coordinate grid independently tests half-pixel offsets, exponent mapping and both axes;
- analytic-boundary inclusion, strict bailout equality, escape-at-target precedence, nonanalytic bounded cap, short escape, `-2` boundary behavior and conjugacy are represented;
- four deliberately sensitive f32 boundary/signed-zero cases are recorded separately;
- CPU-f64 benchmark median: `5.9994 ms` for `2,442,855` explicitly executed iterations.

The CPU gate and a test-specific TypeScript compilation of the browser harness are part of `pnpm run build`. All existing release, WGSL, persistent-field, presentation, provenance, production TypeScript and Vite gates continued to pass.

## Real-browser production-WGSL gate

Browser: Chromium/Edge `150.0.4078.83`  
Adapter: AMD, architecture `rdna-4`  
Fixture: standalone Vite test page, no production app or presentation execution

- all 16 strict cases matched CPU-f64 discrete status and iteration exactly;
- every same-device chunk schedule produced bit-identical recurrence state, metadata, accepted result, quality and final counters;
- all 256 exact-dyadic grid pixels matched: 245 escaped, 7 analytic and 4 capped;
- maximum grid smooth-escape absolute difference: `0.0000003499`;
- cap-suppressed state remained active at iteration 2 with no accepted quality, then published a cap, then continued to escape at iteration 3 when the target increased;
- signed-zero results are bit-identical; cusp diagnostics retain the expected inside/outside analytic ordering;
- maximum strict smooth-escape absolute difference: `0.0000831661`;
- WGSL compilation, scoped validation/internal/out-of-memory, uncaptured, device-loss, browser and non-finite errors: `0`.

The existing real-browser presentation continuity gate was rerun on the same source tree and passed all three transformed/batch pairs.

## Baseline performance telemetry

Scene: 128x128 boundary tile at 1,000 target iterations, 64-iteration chunks. Seven measured runs followed two warmups.

- explicitly executed GPU recurrence iterations: `2,436,847`;
- final classes: 16,097 escaped, 287 capped, zero non-finite;
- fresh-run queue-completion latency: p50 `3.000 ms`, p95 `3.500 ms`, range `2.500â€“3.500 ms`;
- readback wall time p50: `3.000 ms`.

These are local before-optimization telemetry, not timestamp-query measurements, recurrence-only throughput, or a cross-adapter threshold. Each measured run creates fresh resources; the timed queue completion follows uniform uploads and can include upload completion, lazy GPU initialization, submission and browser scheduling overhead. Compilation, host allocation and readback are outside the interval.

## Evidence files

Every invocation writes an isolated run directory and one authoritative terminal manifest. The immutable bundle for source commit `ac9b38e` is checked in under `docs/evidence/numerical/ac9b38e-amd-rdna4/`:

- `cpu-oracle.json`;
- `browser-report.json`;
- `manifest.json`;
- `numerical-playwright-trace.zip`;
- `numerical-fixture.png`;
- `checksums.sha256`.

No claim is made that CPU-f64 is an MPFR oracle. MPFR/direct-interval stabilization remains required before deep perturbation, BLA or series approximation can become authoritative.
