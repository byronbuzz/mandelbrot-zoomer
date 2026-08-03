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
- recurrence metadata and accepted-result status encodings are tested independently.

The fixture SHA-256 is `8f9538a6d402b27c0c8df4d1fdb13c062c076271d62f502069f525d21a999666`. The unchanged production direct-shader source SHA-256 is `6f2b4110f3f1f790660bdb99d6116c86ac6dc8f3a59ab5e36ae6157955b63ec1`.

## Portable CPU gate

- 14 strict stable cases passed their checked-in status and iteration goldens;
- monolithic, repeated-one, repeated-seven, repeated-64, Fibonacci and escape-boundary chunk schedules produced identical oracle state;
- analytic-boundary inclusion, strict bailout equality, escape-at-target precedence, nonanalytic bounded cap, short escape, `-2` boundary behavior and conjugacy are represented;
- four deliberately sensitive f32 boundary/signed-zero cases are recorded separately and are not forced into f64 goldens;
- CPU-f64 benchmark median: `6.0663 ms` for `2,442,855` explicitly executed iterations on this machine.

The CPU gate is now part of `pnpm run build`. All existing release, WGSL, persistent-field, presentation, provenance, TypeScript and production Vite gates continued to pass.

## Real-browser production-WGSL gate

Browser: Chromium/Edge `150.0.4078.83`  
Adapter: AMD, architecture `rdna-4`  
Fixture: standalone Vite test page, no production app or presentation execution

- all 14 strict cases matched CPU-f64 discrete status exactly;
- all 14 strict cases matched escape/cap iteration exactly;
- every same-device chunk schedule produced bit-identical recurrence state, metadata, accepted result and final counters;
- maximum smooth-escape absolute difference from CPU-f64 was `0.0000831661`, below the initial portable bound of `0.02`;
- WGSL compilation errors: `0`;
- scoped validation/internal/out-of-memory errors: `0`;
- uncaptured WebGPU errors: `0`;
- non-finite pixels: `0`;
- premature device loss: `false`;
- browser console/page/HTTP errors: `0`.

The existing real-browser presentation continuity gate was rerun on the same source tree and passed all three transformed/batch pairs.

## Baseline performance telemetry

Scene: 128x128 boundary tile at 1,000 target iterations, 64-iteration chunks. Seven measured runs followed two warmups.

- explicitly executed GPU recurrence iterations: `2,436,847`;
- final classes: 16,097 escaped, 287 capped, zero non-finite;
- queue-completion wall time: p50 `3.500 ms`, p95 `4.200 ms`, range `2.500â€“4.500 ms`;
- readback wall time p50: `3.000 ms`;
- derived p50 throughput: approximately `696.2 million` explicit recurrence iterations/second.

These are local before-optimization telemetry, not timestamp-query measurements and not a cross-adapter threshold. Shader compilation, resource allocation and readback are excluded from the queue-completion interval; submission and browser scheduling overhead are not. The next scheduler PR must report the same scene and raw metric definitions, together with end-to-end latency and publication-work counts.

## Evidence files

The gate writes ignored, raw local artifacts under `test-results/numerical/`:

- `cpu-oracle.json`;
- `browser-report.json`;
- `numerical-playwright-trace.zip`;
- `numerical-fixture.png`.

No claim is made that CPU-f64 is an MPFR oracle. MPFR/direct-interval stabilization remains required before deep perturbation, BLA or series approximation can become authoritative.
