# Decision 0004: bounded asynchronous compute and direct atlas publication

## Status

Accepted for the 1.4 performance stage.

## Context

The frozen direct numerical kernel was not the dominant shallow-navigation bottleneck. Production serialized every numerical batch with a queue-wide completion wait, mapped one counter buffer per tile, recoloured every accepted pixel after every chunk, and copied complete colour, quality, and evidence textures into the accepted atlas.

## Decision

- Keep the direct and perturbation numerical kernels and their accepted result/quality contract unchanged.
- Allow at most three FIFO GPU submissions in flight.
- Use one aggregate counter-readback buffer per submission and one `mapAsync` operation per batch.
- Permit at most one in-flight mutation for any tile.
- Advance iteration frontier, health, coverage, palette, cap mode, test revision, and authoritative state only when the corresponding FIFO submission retires.
- Drain submitted work before preparing a superseding request or activating a numerical reset.
- Publish result/quality directly into storage-capable accepted-atlas textures in the same command buffer as numerical work.
- Clear a newly leased atlas slot on its first publication, then write only newly accepted pixels; palette changes recolour accepted escaped pixels only.
- Materialize a previously suppressed cap through an acceptance-only numerical dispatch.
- Keep the legacy presenter path available for comparison.

## Consequences

The main thread no longer waits for each numerical batch, GPU work remains bounded, and the production path performs no full-tile atlas texture copies. A completed host-side tile update still implies that its matching atlas publication has completed. Precision features remain outside this change.
