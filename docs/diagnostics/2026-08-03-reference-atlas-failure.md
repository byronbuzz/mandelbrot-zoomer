# 1.2.1 deep-reference failure diagnosis

The 1.2.1 browser trace at approximately `10^5.52` and `10^13.84` showed the same architectural failure:

- all visible tiles remained on direct rendering;
- zero tiles entered perturbation;
- reference failures climbed from 84,137 to 143,106;
- the scheduler reported no active work even though many tiles contained unresolved pixels;
- the settled lattice used 390 visible 128×128 tiles and 599 cached tiles.

## Root causes

1. The CPU atlas rejected every escaped or target-short reference before the GPU could use it. Exterior tiles often have no reference point that survives the complete target iteration count, so full-length admission is not a valid local-perturbation requirement.
2. Rejected references restarted candidate passes on every camera request. Shared group failures were counted once per tile consumer, creating the runaway failure total.
3. The renderer considered an empty work queue equivalent to numerical convergence.
4. The lattice used `floor(pixelExponent)`, oversampling by up to two in each dimension and generating roughly four times the intended tile and reference workload.
5. References were keyed only to one lattice level, preventing useful parent-level references from seeding descendant tiles during continuous zoom.

## 1.2.2 correction

- admit finite short and escaped reference orbits;
- let GPU orbit-exhaustion/glitch telemetry select local repair references;
- make worker errors terminal for the requested target rather than recursively restarting;
- use interaction-bounded reference lengths during motion and upgrade unresolved tiles after settling;
- search cached reference coverage across lattice levels;
- generate candidates around the complete 2×2 reference group;
- use a small persistent worker pool;
- derive active/converged counts from accepted numerical coverage;
- choose the base lattice with `ceil(pixelExponent)` so the numerical budget corresponds to approximately one sample per display pixel.

This correction does not add a zoom threshold or a viewport-wide fallback.
