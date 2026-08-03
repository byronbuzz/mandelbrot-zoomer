# Release 1.4 gate evidence

Date: 2026-08-03 (Australia/Sydney)

## Scope

Release 1.4 integrates the presentation kernel with the existing accepted numerical tile field. Recurrence, perturbation, reference-orbit, precision-selection and numerical scheduling mathematics are unchanged. The old presenter remains executable with `?presenter=legacy`.

## Build and numerical-preservation gates

- release identity: UI `1.4`, package `1.4.0`;
- 25 WGSL template strings parsed by the source validator;
- all 1.x foundation, persistent-field, local-reference, repair and 1.3 invariants passed;
- 12 Phase 1 invariants and 15 original reprojection samples passed;
- 17 production 1.4 invariants passed;
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

## Failures found and fixed by the executable gates

1. A WGSL uniform padding layout required 48 bytes although the host allocated 32. The structure now uses four scalar `u32` fields and validates at 32 bytes.
2. A 256-slot atlas could transiently exhaust before the cache eviction pass. Allocation now retires the coldest non-visible tile before exhaustion while keeping the atlas bounded at 512 slots.
3. The overlay instance cap could omit non-occluded cached tiles. It now matches the bounded 512-slot atlas.

Visual evidence is retained in the release handoff under `outputs/v14-evidence/`.
