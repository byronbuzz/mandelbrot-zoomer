# Release 1.4 gate evidence

Date: 2026-08-03 (Australia/Sydney)

## Scope

Release 1.4 integrates the presentation kernel with the existing accepted numerical tile field. Recurrence, perturbation, reference-orbit, precision-selection and numerical scheduling mathematics are unchanged. The old presenter remains executable with `?presenter=legacy`.

## Build and numerical-preservation gates

- release identity: UI `1.4`, package `1.4.0`;
- 25 WGSL template strings parsed by the source validator;
- all 1.x foundation, persistent-field, local-reference, repair and 1.3 invariants passed;
- 12 Phase 1 invariants and 15 original reprojection samples passed;
- 27 production 1.4 invariants passed;
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
- visual result: continuous boundary detail with complete canvas coverage; no horizontal cutoff, rectangular hole, or parent/child transition seam.

The raw candidate record includes every observed zoom depth, exact fixed-point camera centre, binary scale, DPR, viewport/canvas sizes, presenter counters and the PNG filename in `outputs/v14-evidence/stress-boundary-stable-1000-depth3.json`.

The failing pre-fix atlas capture at approximately `10^2` is retained alongside the passing captures. It reproduced a hard rectangular cutoff. The same class of hole appeared in the legacy path, proving that the fault was in shared tile admission rather than history sampling or atlas composition.

## Failures found and fixed by the executable gates

1. A WGSL uniform padding layout required 48 bytes although the host allocated 32. The structure now uses four scalar `u32` fields and validates at 32 bytes.
2. A 256-slot atlas could transiently exhaust before the cache eviction pass. Allocation now retires the coldest non-visible tile before exhaustion while keeping the atlas bounded at 512 slots.
3. The overlay instance cap could omit non-occluded cached tiles. It now matches the bounded 512-slot atlas.
4. The legacy comparison path still allocated and populated atlas resources. Legacy mode now has no atlas allocation, publication copy, history frame or atlas-presenter dependency.
5. Zero-sized canvases were coerced to one pixel and physical presentation sizes ignored the adapter texture limit. Presentation now suspends at zero CSS extent and clamps allocations to `maxTextureDimension2D`.
6. Empty current-tile packets skipped presentation entirely. The atlas path now presents history with a zero-instance overlay, preserving continuity while numerical work catches up.
7. Slot and lease identifiers were packed through f32, becoming inexact above `2^24`. They are now host-packed and shader-consumed as exact `u32` values.
8. A candidate history anchor was promoted immediately after submission. Promotion now waits for queue completion and a clean validation error scope; the candidate cannot be reused while promotion is pending.
9. Parent tiles were culled as soon as four complete children existed in cache, even when those children were not all admitted to the current draw packet. That created the reproduced rectangular holes during cross-level transitions. The presentation loop now retains the parent; ordinary draw ordering lets admitted children overlay it without risking uncovered pixels.
10. Zero CSS extent originally suspended presentation but not numerical submissions. The resize/request path now suspends the scheduler, clears queued work and resumes only after a non-zero request.
11. Display dimensions were independently clamped to the adapter limit, which could distort the mathematical aspect ratio. One common scale factor now constrains both axes.
12. A resize could reconfigure and clear the canvas while authoritative-anchor promotion was pending. Drawing-buffer resize is now deferred until promotion completes, preserving the last presented frame and pending-resource lifetime.
13. Browser form-state restoration could overwrite the test iteration slider after startup. Test mode now locks the snapped target on every numerical request; the recorded trace remained exactly 1,000 throughout.

Visual evidence is retained in the release handoff under `outputs/v14-evidence/`.
