import type { BenchmarkScene } from './types';

export const BENCHMARK_SCENES: readonly BenchmarkScene[] = [
  {
    id: 'whole-set',
    label: 'Whole Mandelbrot set',
    centerX: -0.5,
    centerY: 0,
    viewportHeight: 3,
    iterations: 512,
    purpose: 'Analytic interior rejection, fast exterior escape and baseline throughput.'
  },
  {
    id: 'cardioid-interior',
    label: 'Main-cardioid interior',
    centerX: -0.2,
    centerY: 0,
    viewportHeight: 0.7,
    iterations: 8192,
    purpose: 'Interior rejection, periodicity and high-iteration watchdog safety.'
  },
  {
    id: 'seahorse-valley',
    label: 'Seahorse Valley boundary',
    centerX: -0.743643887037151,
    centerY: 0.13182590420533,
    viewportHeight: 0.000014,
    iterations: 4096,
    purpose: 'Boundary divergence, progressive spatial refinement and coordinate mapping.'
  },
  {
    id: 'needle-boundary',
    label: 'High-complexity boundary',
    centerX: -1.25066,
    centerY: 0.02012,
    viewportHeight: 0.00008,
    iterations: 16_384,
    purpose: 'Long-tail active pixels and progressive iteration continuation.'
  }
];
