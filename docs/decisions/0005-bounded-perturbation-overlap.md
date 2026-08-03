# Decision 0005: bounded perturbation overlap and explicit transport ceiling

## Status

Accepted for the first post-1.4 precision stage.

## Context

The 1.4 numerical field selected perturbation only after adjacent double-float coordinates became identical or a non-finite result appeared. At the user-observed loss region near `10^5`, direct coordinates remained distinguishable, so no reference was requested and the perturbation path was never exercised. Reference work also lacked demand epochs, and the worker advertised requested working precision through a four-f32 orbit transport whose actual ceiling is approximately 96 bits.

## Decision

- Preserve coordinate collapse and non-finite health as required escalation.
- Add a bounded, settled, finest-level, focus-first perturbation overlap beginning at sample exponent `-23`.
- Limit speculative overlap to four focus tiles; direct calculation continues while their references are pending.
- Give reference demand the renderer request epoch. Cancel queued and active older work when a newer camera request arrives, while retaining completed cache entries.
- Record working precision, transport precision, and a versioned reference contract separately.
- Require sample-exponent demand plus 32 guard bits. Reject a reference whose transport does not meet that demand.
- Keep both production numerical shaders byte-for-byte frozen in this stage.

## Consequences

The full reference activation path now runs before visible direct precision loss without creating a viewport-wide reference storm. The current 96-bit transport is valid in the `10^5` overlap canary but is explicitly reported insufficient around `10^35`; deeper authority now requires the planned arbitrary-precision service and higher-precision or extended-range transport.
