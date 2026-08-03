# 1.4 asynchronous compute/direct-atlas performance evidence

## Scope

This evidence compares deployed build `501804c` with the performance-stage source on the same AMD adapter, browser, viewport, user-selected complex boundary, and 5,000-iteration moving request.

The numerical kernels are unchanged. The optimized build changes GPU submission/readback scheduling and accepted-atlas publication only.

## Same-GPU movement result

At approximately one second after the moving request:

- deployed baseline: 5,446 retired numerical tile-chunks;
- optimized build: 30,637 retired numerical tile-chunks;
- ratio: 5.625×.

The optimized trace observed the full three-batch submission window during active work and retired every submitted chunk. WebGPU validation errors were zero.
The gate asserts that the comparison page identifies itself as build `1.4 · 501804c`; it fails instead of silently comparing against a later Pages deployment.

After the moving trace, both builds were changed to `settled` at the identical camera. The optimized build reached 479/479 converged tiles with no queued or in-flight work. The synchronized continuity reduction reported:

- zero invalid pixels;
- zero provisional-cap pixels;
- zero quality regressions;
- zero escaped-to-provisional-black regressions;
- zero conflicts.

## Other release gates

- Frozen numerical browser oracle: 16 strict cases passed on AMD; 2,436,847 explicit iterations observed.
- User-positioned bidirectional continuity trace: three transformed/first-batch pairs passed.
- Device loss: renderer recovered to epoch 2 with no browser errors or warnings.
- Production build: all release, WGSL, persistent-field, Phase 1, provenance, numerical, TypeScript, and performance-pipeline invariants passed.

## Raw files

- `throughput-report.json`: timestamped baseline/optimized samples and settled transition results.
- `continuity-diagnostics.json`: synchronized transformed-frame and first-batch continuity evidence.
- `numerical-browser-report.json`: AMD direct-kernel oracle report.
