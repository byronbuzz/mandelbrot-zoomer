import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import reflectModule from 'wgsl_reflect';

const WgslReflect = reflectModule.WgslReflect;
if (typeof WgslReflect !== 'function') {
  throw new Error('wgsl_reflect did not expose WgslReflect');
}

const root = new URL('../src/', import.meta.url);
const reservedIdentifiers = new Set([
  'asm', 'bf16', 'class', 'do', 'enum', 'extern', 'f16', 'f64', 'friend',
  'goto', 'handle', 'i8', 'i16', 'i64', 'long', 'mat', 'namespace', 'new',
  'operator', 'premerge', 'private', 'protected', 'public', 'regardless',
  'short', 'signed', 'smooth', 'static', 'template', 'this', 'throw',
  'typedef', 'u8', 'u16', 'u64', 'union', 'unless', 'unsigned', 'using',
  'vec', 'virtual', 'void', 'volatile'
]);

function sourceFiles(directory) {
  const files = [];
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) files.push(...sourceFiles(path));
    else if (path.endsWith('.ts')) files.push(path);
  }
  return files;
}

function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ');
}

const declarationPatterns = [
  /\b(?:let|const|override)\s+([A-Za-z_]\w*)/g,
  /\bvar(?:\s*<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  /\bfn\s+([A-Za-z_]\w*)/g,
  /^\s*([A-Za-z_]\w*)\s*:/gm
];

const failures = [];
let shaderCount = 0;
for (const file of sourceFiles(root.pathname)) {
  const source = readFileSync(file, 'utf8');
  const shaderPattern = /\/\*\s*wgsl\s*\*\/\s*`([\s\S]*?)`/g;
  for (const match of source.matchAll(shaderPattern)) {
    shaderCount++;
    const shaderSource = match[1];
    const shader = stripComments(shaderSource);
    for (const pattern of declarationPatterns) {
      pattern.lastIndex = 0;
      for (const declaration of shader.matchAll(pattern)) {
        const identifier = declaration[1];
        if (reservedIdentifiers.has(identifier)) {
          failures.push(`${relative(process.cwd(), file)}: reserved WGSL identifier '${identifier}'`);
        }
      }
    }
    try {
      new WgslReflect(shaderSource);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push(`${relative(process.cwd(), file)}: WGSL parse failed: ${message}`);
    }
  }
}

if (shaderCount === 0) failures.push('No /* wgsl */ shader template strings were found');
if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`Parsed and validated ${shaderCount} WGSL shader template strings.`);
