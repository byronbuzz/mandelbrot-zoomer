# Release 1.4 gate evidence

Date: 2026-08-03 (Australia/Sydney)

## Scope

Release 1.4 integrates the presentation kernel with the existing accepted numerical tile field. Recurrence, perturbation, reference-orbit, precision-selection and numerical scheduling mathematics are unchanged. The old presenter remains executable with `?presenter=legacy`.

## Build and numerical-preservation gates

- release identity: UI `1.4`, package `1.4.0`;
- 25 WGSL template strings parsed by the source validator;
- all 1.x foundation, persistent-field, local-reference, repair and 1.3 invariants passed;
- 12 Phase 1 invariants and 15 original reprojection samples passed;
- 27 structural/oracle 1.4 invariants passed;
- TypeScript and the production Vite build passed;
- presentation-only deep-scale ratios passed at binary exponents -1022, -1074, -2048 and -4096;
- worst simulated f32 shader-coordinate error: `0.0001831` source texel (limit `0.01`).

## Real-browser WebGPU gates

Browser: current in-app Chromium WebGPU implementation. Adapter label: `amd`. Initial physical canvas: `1600x900`.

### Default atlas-history presenter

- 466 numerical tiles converged at the initial view;
- one atlas overlay draw presented 276 non-occluded instances;
- final navigation trace: 611 presentation frames, 609 history frames, 2 fallback frames;
- presentation host time at capture: `0.200 ms`;
- worst admitted reprojection error at capture: `0.000036` source texel;
- display readout: `60 Hz`;
- zero console warnings, console errors or WebGPU validation errors.

### Resize continuity

Landscape to portrait to landscape changed presentation resource epoch `1 -> 2 -> 3`. History continued on every post-initial frame. Captured host presentation time was `0.200 ms`, `0.200 ms`, then `0.100 ms`; validation errors remained zero.

### Forced device loss

The executable `?testDeviceLoss=1` hook destroyed the active device. Renderer epoch advanced `1 -> 2`; the renderer recreated the adapter/device, pipelines, atlases, history, tile resources and reference service, then rebuilt 466 converged tiles. Recovery capture recorded 270 frames, 268 history frames, `0.300 ms` host presentation time and zero validation errors.

### Legacy comparison

`?presenter=legacy` rendered the same initial Mandelbrot view with 466 converged tiles, 60 Hz and zero warnings, errors or validation failures.

### Boundary zoom stress trace

The browser harness accepts `?testIterations=1000`, which snaps and locks the native control to a valid step before every numerical request. This makes the prescribed trace repeatable despite browser form-state restoration and without synthetic range-control timing.

- anchor: a visually confirmed point on the main cardioid/bulb boundary at canvas CSS position `(300, 382)` in the recorded 756x927 viewport at DPR 1.375 (1039x1274 physical canvas);
- input: eleven wheel-in steps with requested `scrollY=-420`, spaced 420 ms apart at the same anchor, with no pan or cursor relocation, followed by two settling corrections at 1.8 s spacing to counter browser gesture momentum;
- atlas-history trace crossed `10^2` and `10^3.016`, then produced a stable settled capture at `10^3.074`;
- final atlas-history capture: exactly 1,000 requested iterations, 180 visible/converged tiles, 224 cached tiles, 60 Hz, zero validation errors and `0.000035` worst admitted source-texel error;
- independent legacy replay beyond `10^4`: 1,000 iterations, 446 visible/converged tiles, 60 Hz, zero validation errors;
- settled screenshot appeared visually complete, with continuous boundary detail and no visible horizontal cutoff, rectangular hole, or parent/child transition seam; this did not establish continuity before settlement.

The raw candidate record includes every observed zoom depth, exact fixed-point camera centre, binary scale, DPR, viewport/canvas sizes, presenter counters and the PNG filename in `outputs/v14-evidence/stress-boundary-stable-1000-depth3.json`.

### User-selected deep-boundary trace at 5,000 iterations

The final presentation candidate `c410bc7` was also tested from a user-selected, densely filamented boundary endpoint rather than an automatically chosen coarse-view coordinate.

- baseline: `10^5.818`, exactly 5,000 requested iterations, 512 visible/converged tiles, 60 Hz and zero validation errors;
- fixed continuation anchor: CSS `(300, 460)` in the 756x927 viewport at DPR 1.375;
- stable `10^7.131` capture: 334 visible/converged tiles, a visually complete settled screenshot, 60 Hz, `0.000035` worst admitted source-texel reprojection error and zero validation errors;
- stable `10^8.007` capture: 379 visible/converged tiles, a visually complete settled screenshot, 60 Hz, the same `0.000035` reprojection error and zero browser warnings, browser errors or WebGPU validation errors;
- numerical state at `10^8.007`: 379 direct tiles, zero perturbation tiles, zero references and zero reference failures.

Fine-grained speckling and structured banding emerge around `10^7` and become severe by `10^8`. The artifact remains in a settled frame with all visible tiles reported converged, low reprojection-coordinate error, zero validation errors and every affected tile still using direct mode. This strongly localizes the anomaly to numerical/render-quality content before presentation, but does not categorically exclude history or coverage effects because coverage was not independently measured. A later precision milestone must compare a frozen numerical buffer against an MPFR oracle and an independently rendered presentation baseline. No perturbation, series, BLA, reference-orbit or recurrence change is included in Release 1.4.

### Live zoom-out blocker

Settled captures concealed a separate presentation failure. From the same `10^8.007`, 5,000-iteration endpoint, a PNG was taken immediately after each zoom-out input burst at every decade down to approximately `10^2`. The corresponding post-transition diagnostics for orders `10^6` through `10^2` were sampled approximately 350 ms later, after multiple presentation frames and, in some cases, promotions. These records prove a live continuity failure but are not synchronized first-presentation-frame measurements.

- `10^8 -> 10^7`: the canvas became almost black with large rectangular retained islands before recovering at `10^7.131`;
- `10^7 -> 10^6`: only a small square of the old view survived; the post-transition packet had 16 atlas instances, 5 converged tiles and 31 active tiles;
- `10^5 -> 10^4`: the live frame showed a stale, block-edged plume while the UI reported `moving · calculating`, zero visible numerical tiles and zero converged tiles;
- `10^3 -> 10^2`: a large stale vertical band and block-shaped edge fill remained visible before refinement.

All captures retained 60 Hz presentation and zero WebGPU validation errors, so validation-clean submission is not a coverage guarantee. The root architectural defect is that the presenter retains only one full-screen history view. During zoom-out that source maps to a small rectangle inside the newly exposed viewport; the reprojection shader clears all source-UV misses to black, while the accepted atlas has too few retained coarser/outside tiles to cover the remainder. Subsequent 128x128 tile admission causes the visible block in-fill and jumps.

This is a Release 1.4 presentation blocker. The milestone must remain draft until a bounded multi-scale/overscanned history or equivalent retained coarse-coverage layer keeps the full target viewport continuously covered during zoom-out, and an executable pre-settle coverage gate passes the recorded `10^8 -> 10^2` trace. Evidence is stored in `outputs/v14-evidence/live-zoom-out-5000-order-{7..2}.{png,json}` and `live-zoom-out-5000-summary.json`.

An earlier pre-fix atlas capture at approximately `10^2` is retained as historical evidence of a specific parent-culling hole. The same class of hole appeared in the legacy path, localizing that earlier defect to shared tile admission. Fixing it did not establish general reverse-navigation coverage: the live `10^8 -> 10^2` gate above exposes additional unresolved history and coverage failures.

Release checklist status: the behavioral pre-settle full-coverage gate is **FAILED / PENDING**. Release 1.4 must not advance or deploy from this draft PR until a frame-synchronized validity gate passes both presenters across the recorded reverse-navigation trace.

## Failures found and fixed by the executable gates

1. A WGSL uniform padding layout required 48 bytes although the host allocated 32. The structure now uses four scalar `u32` fields and validates at 32 bytes.
2. A 256-slot atlas could transiently exhaust before the cache eviction pass. Allocation now retires the coldest non-visible tile before exhaustion while keeping the atlas bounded at 512 slots.
3. The overlay instance cap could omit non-occluded cached tiles. It now matches the bounded 512-slot atlas.
4. The legacy comparison path still allocated and populated atlas resources. Legacy mode now has no atlas allocation, publication copy, history frame or atlas-presenter dependency.
5. Zero-sized canvases were coerced to one pixel and physical presentation sizes ignored the adapter texture limit. Presentation now suspends at zero CSS extent and clamps allocations to `maxTextureDimension2D`.
6. Empty current-tile packets skipped presentation entirely. The atlas path now presents history with a zero-instance overlay, preserving continuity while numerical work catches up.
7. Slot and lease identifiers were packed through f32, becoming inexact above `2^24`. They are now host-packed and shader-consumed as exact `u32` values.
8. A candidate history anchor was promoted immediately after submission. Promotion now waits for queue completion and a clean validation error scope; the candidate cannot be reused while promotion is pending.
9. Parent tiles were culled as soon as four complete children existed in cache, even when those children were not all admitted to the current draw packet. That created the reproduced rectangular holes during cross-level transitions. The presentation loop now retains the parent and lets admitted children overlay it. This fixes that specific parent-culling hole; it does not prove full reverse-navigation coverage.
10. Zero CSS extent originally suspended presentation but not numerical submissions. The resize/request path now suspends the scheduler, clears queued work and resumes only after a non-zero request.
11. Display dimensions were independently clamped to the adapter limit, which could distort the mathematical aspect ratio. One common scale factor now constrains both axes.
12. A resize could reconfigure and clear the canvas while authoritative-anchor promotion was pending. Drawing-buffer resize is now deferred until promotion completes, preserving the last presented frame and pending-resource lifetime.
13. Browser form-state restoration could overwrite the test iteration slider after startup. Test mode now locks the snapped target on every numerical request; the recorded trace remained exactly 1,000 throughout.

Visual evidence is retained in the release handoff under `outputs/v14-evidence/`.
