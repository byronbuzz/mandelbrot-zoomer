# Decision 0006: continuous moving perturbation and fine-state retention

## Status

Accepted. Supersedes the settled-only scheduling and four-tile seed limits in Decision 0005.

## Context

The first overlap deployment still rendered every tile directly during continuous zoom. Every camera request cancelled both queued and active reference workers, while preferred perturbation was restricted to the settled state. The result was deterministic starvation: at the user-reported `10^5.942447` failure position diagnostics showed 240 direct tiles, zero perturbation tiles, and zero references.

Fine perturbation state could also disappear between the coarse, middle, and fine admission passes of a new request. Cache eviction protected only the levels admitted so far, allowing still-relevant fine tiles from the preceding request to be evicted before the new fine level was admitted.

## Decision

- Allow preferred perturbation on the finest level during moving, settling, and settled interaction.
- Bound speculative reference work to 16 seeded requests and 16 total pending requests.
- On a new demand epoch, cancel obsolete queued work but let the bounded active workers finish and enter the reusable reference cache.
- Permit local reference reuse within eight tile spans.
- Protect the complete planned multiscale visible set from eviction before progressive level admission begins.
- Requeue the current request when a stale-demand reference completes so it can reuse the newly cached orbit.
- Retry deferred repair references when bounded demand capacity becomes available.
- Gate the stage at the exact user-reported failure coordinate, with request-correlated moving and settled diagnostics.

## Consequences

At the recorded AMD hardware-WebGPU canary, all 96 finest-level tiles use perturbation while interaction remains moving at `10^5.942447`. Coarser multiscale direct tiles remain as presentation fallback. Reference work remains bounded, reports 224-bit CPU working precision and 96-bit GPU transport precision separately, and drains after settling without WebGPU validation or reference failures.
