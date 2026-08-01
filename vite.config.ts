import { defineConfig } from 'vite';

export default defineConfig({
  base: '/mandelbrot-zoomer/',
  build: { target: 'es2022', sourcemap: true },
});
