import { readFileSync } from 'node:fs';

const build = readFileSync(new URL('../src/app/build.ts', import.meta.url), 'utf8');
const packageJson = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const failures = [];
if (!build.includes("APP_VERSION = '1.4'")) failures.push('UI release must be 1.4.');
if (packageJson.version !== '1.4.0') failures.push('Package release must be 1.4.0.');
if (!packageJson.version.startsWith('1.4.')) failures.push('UI and package major/minor versions disagree.');
if (failures.length) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log('Validated release identity: UI 1.4, package 1.4.0.');
