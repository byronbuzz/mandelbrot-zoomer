# Decision 0008: bound perturbation rebase acceptance

Status: accepted for the numerical-kernel phase.

## Evidence

Build `56abf9e` converged every visible tile with zero validation errors, but the
user's exact `10^9.340462` and `10^13.340777` traces became progressively fuzzy.
The new 512-bit oracle reproduces the defect: a short 402-sample reference in
the reported `10^9` region causes 13â€“30 rebases and four of nine pixels are
accepted with incorrect escape iterations.

## Decision

Retain the fast double-single rebase path only through twelve cumulative rebases.
The thirteenth rebase marks the pixel as a numerical glitch and invokes the existing
reference-repair path. Increase initial reference candidates from 9 to 25 and
repair candidates from 25 to 81 so repair more often finds a long-lived local
orbit on its first attempt. Order candidates from the tile or group centre
outward: the probe retains the first full-horizon survivor, so the former
corner-first enumeration could choose an unnecessarily distant orbit and cause
a large perturbation and prolonged repair wave. Previously accepted pixels
remain visible while repair runs.

This is deliberately conservative. A pixel may be delayed, but it may not be
published as final after the measured precision budget is exhausted.

## Gates

- production WGSL compilation and zero validation errors;
- exact agreement for the existing periodic and user-boundary scenarios;
- safe deferral, never final acceptance, for every repeated-rebase failure pixel;
- production build, reference-transport, numerical, precision-overlap and
  continuity browser gates;
- interactive AMD trace at the reported deep coordinate.

## Rollback

Restore the unbounded orbit-end rebase behavior and the prior candidate grids.
That rollback restores `56abf9e` performance but also restores the proven false
final classifications, so it is not acceptable as a correctness release.

