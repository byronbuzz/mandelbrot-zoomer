import { defineConfig } from 'vite';

const buildSha = process.env.GITHUB_SHA?.slice(0, 7) ?? 'dev';

export default defineConfig({
  base: '/mandelbrot-zoomer/',
  define: {
    __BUILD_SHA__: JSON.stringify(buildSha)
  },
  build: { target: 'es2022', sourcemap: true }
});
