# Decision 0007: generic boundary rebasing and precision-safe multiscale coverage

## Status

Accepted. This is an evidence-driven numerical-kernel amendment to Decisions 0005 and 0006.

## Context

The periodic `c = i` oracle passed at a nominal `10^35` scale, but user captures at
`10^6`, `10^12`, `10^16`, `10^18`, `10^20`, and `10^24` exposed ordinary-boundary
failures. Coarser visible levels remained direct below the overlap threshold, while
the worker omitted the first escaped orbit sample. Short local references therefore
ended immediately before a classifiable escape and triggered hundreds of repair
waves. A plausible partial image could be reported as settled with 14-16 references
still pending and almost every tile active.

## Decision

- Prefer perturbation at every planned visible level below the overlap threshold.
- Cap precision-unsafe direct tiles at the direct safety horizon until a reference
  activates; never accept their full iteration result as final coverage.
- Include pending and coalesced reference work in the renderer busy contract.
- Store the first escaped reference sample so the GPU can classify that iteration.
- Rebase scaled perturbations to orbit index zero when the transported reference
  ends or the perturbation dominates the reference, preserving the current orbit
  state instead of immediately demanding another reference.
- Gate the production shader against both the periodic deep canary and the exact
  user-derived `10^16` boundary orbit, including pixels that escape one iteration
  after the original short reference.
- Treat reference drain, tile convergence, and a quiet final window as completion
  evidence; visual plausibility alone is not a precision pass.

## Consequences

The 10^12 mixed-mode regression no longer displays rectangular direct-tile
corruption. Orbit-end pixels can progress by mathematically valid rebasing, reducing
repair churn. Deep rendering remains feature-flag comparable against the legacy
96-bit transport, and the direct numerical kernel remains unchanged.
