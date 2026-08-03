# ADR 0002: production presentation integration is release 1.4

Status: accepted for gated implementation.

## Decision

Release 1.4 integrates the Phase 1 presentation contract beneath the existing numerical tile field. It adds a bounded accepted-colour/quality atlas, slot leases, one instanced overlay, persistent exact-view history reprojection, transactional resize continuity, and whole-renderer device-loss recreation. The existing presenter remains available with `?presenter=legacy` for direct comparison.

The numerical recurrence, perturbation, reference-orbit selection, precision policy, and scheduler mathematics remain unchanged. The historical roadmap's former 1.4 numerical-acceleration label is superseded by this user-authorized release decision; numerical acceleration moves to a later gated milestone.

## Promotion gate

The atlas presenter becomes the default only after source invariants, TypeScript/WGSL validation, CPU transform oracles, built-artifact browser execution, real-GPU visual capture, resize/navigation/loss traces, and a legacy comparison pass. A reprojected-only candidate is never an accepted history anchor.
