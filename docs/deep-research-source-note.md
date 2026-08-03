# Governing research source note

The 1.1–1.2 development line is based on the supplied report **High-Performance Mandelbrot and Julia Rendering on the Web**.

The implementation follows its central recommendations:

- avoid iterations before making iterations cheaper;
- use WebGPU compute as the primary backend;
- keep iterative state on the GPU;
- use bounded chunked compute for long-running pixels;
- schedule and escalate at tile granularity;
- use active-tile and active-pixel compaction where profitable;
- move deep precision into high-precision reference orbits;
- evaluate nearby pixels as perturbation deltas;
- use multiple local references rather than widening every pixel;
- use deterministic higher-precision sentinels to drive escalation;
- avoid CPU/GPU synchronisation in the ordinary path;
- retain series approximation and BLA as later conservative accelerators with exact fallback.
