import { readFile, writeFile } from 'node:fs/promises';

await import('./generate-v33.mjs');

const sourcePath = new URL('../src/mainV33.generated.ts', import.meta.url);
const outputPath = new URL('../src/mainV34.generated.ts', import.meta.url);
let source = await readFile(sourcePath, 'utf8');

function replaceExact(label, before, after) {
  if (!source.includes(before)) throw new Error(`V3.4 generator could not find: ${label}`);
  source = source.replace(before, after);
}

replaceExact(
  'core label',
  '<p><b>Numerical core V3.3:</b> sticky perturbation references, continuity-gated promotion and no deep double-float fallback.</p>',
  '<p><b>Numerical core V3.4:</b> adaptive double-double/triple-double CPU references, sticky perturbation and continuity-safe promotion.</p>'
);

replaceExact(
  'status label',
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.3`;",
  "status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V3.4`;"
);

await writeFile(outputPath, source);
console.log('Generated src/mainV34.generated.ts');
