# Decision 0003: freeze numerical evidence before optimization

Date: 2026-08-03  
Status: accepted

## Context

Release 1.4 established presentation continuity and sharp immutable history. The next production change will remove the direct numerical scheduler's immediate queue-completion wait, per-tile counter mapping, and unconditional recolour/publication work. Those changes alter observation timing and work admission, so visual inspection alone cannot distinguish a throughput improvement from a numerical regression.

The production direct shader also differs from many textbook Mandelbrot implementations: it uses `z0 = 0`, analytic cardioid/period-two certificates, a strict `|z|^2 > 256` bailout, a check-before-update loop, and escape-at-target precedence over a cap.

## Decision

Add a test-only evidence milestone before changing production numerical code.

The milestone freezes the current direct contract in an independent CPU-f64 oracle and versioned fixture corpus. A standalone Vite test page imports only the production direct WGSL, creates its own adapter/device and offscreen buffers/textures, and exposes a browser runner. It does not import the app, canvas renderer, scheduler, presentation kernel, reference service, perturbation shader or colour pipeline.

Stable fixtures require exact discrete status and iteration agreement. Deliberately sensitive boundary fixtures are recorded but do not force f32/f64 parity. Same-device chunk schedules must produce identical recurrence metadata, state and result bits. Smooth escape is secondary and uses a documented tolerance. Performance telemetry is recorded separately from correctness and has no cross-adapter release threshold in this milestone.

## Evidence and gates

- CPU validator is part of `pnpm run build` and writes raw JSON.
- Real-browser WebGPU gate records the production shader hash, fixture hash, browser, OS, adapter information, features, limits, compilation messages, error scopes, uncaptured errors, device-loss state, per-case comparisons and raw benchmark samples.
- Passing requires zero discrete mismatches and zero WebGPU/browser errors.
- The browser benchmark excludes compilation, resource allocation and readback from its queue-completion wall interval. This remains telemetry, not an exact GPU timestamp claim.
- Production code and the lockfile remain unchanged by this decision.

## Rollback

Remove the test page, oracle files, browser runner, package commands and this decision. Production behavior is unaffected either way.
