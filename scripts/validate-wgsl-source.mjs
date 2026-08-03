import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WgslReflect } from 'wgsl_reflect/wgsl_reflect.module.js';

const root = new URL('../src/', import.meta.url);

// WGSL 2026 Candidate Recommendation, section 16.2.
// A reserved word is forbidden anywhere in a WGSL module, not merely as a declaration name.
const reservedIdentifiers = new Set([
  'NULL', 'Self', 'abstract', 'active', 'alignas', 'alignof', 'as', 'asm',
  'asm_fragment', 'async', 'attribute', 'auto', 'await', 'become', 'cast',
  'catch', 'class', 'co_await', 'co_return', 'co_yield', 'coherent',
  'column_major', 'common', 'compile', 'compile_fragment', 'concept',
  'const_cast', 'consteval', 'constexpr', 'constinit', 'crate', 'debugger',
  'decltype', 'delete', 'demote', 'demote_to_helper', 'do', 'dynamic_cast',
  'enum', 'explicit', 'export', 'extends', 'extern', 'external', 'fallthrough',
  'filter', 'final', 'finally', 'friend', 'from', 'fxgroup', 'get', 'goto',
  'groupshared', 'highp', 'impl', 'implements', 'import', 'inline',
  'instanceof', 'interface', 'layout', 'lowp', 'macro', 'macro_rules', 'match',
  'mediump', 'meta', 'mod', 'module', 'move', 'mut', 'mutable', 'namespace',
  'new', 'nil', 'noexcept', 'noinline', 'nointerpolation', 'non_coherent',
  'noncoherent', 'noperspective', 'null', 'nullptr', 'of', 'operator',
  'package', 'packoffset', 'partition', 'pass', 'patch', 'pixelfragment',
  'precise', 'precision', 'premerge', 'priv', 'protected', 'pub', 'public',
  'readonly', 'ref', 'regardless', 'register', 'reinterpret_cast', 'require',
  'resource', 'restrict', 'self', 'set', 'shared', 'sizeof', 'smooth', 'snorm',
  'static', 'static_assert', 'static_cast', 'std', 'subroutine', 'super',
  'target', 'template', 'this', 'thread_local', 'throw', 'trait', 'try', 'type',
  'typedef', 'typeid', 'typename', 'typeof', 'union', 'unless', 'unorm',
  'unsafe', 'unsized', 'use', 'using', 'varying', 'virtual', 'volatile', 'wgsl',
  'where', 'with', 'writeonly', 'yield'
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

const failures = [];
let shaderCount = 0;
for (const file of sourceFiles(fileURLToPath(root))) {
  const source = readFileSync(file, 'utf8');
  const shaderPattern = /\/\*\s*wgsl\s*\*\/\s*`([\s\S]*?)`/g;
  for (const match of source.matchAll(shaderPattern)) {
    shaderCount++;
    const shaderSource = match[1];
    const shader = stripComments(shaderSource);
    const identifiers = shader.match(/[A-Za-z_][A-Za-z0-9_]*/g) ?? [];
    for (const identifier of identifiers) {
      if (reservedIdentifiers.has(identifier)) {
        failures.push(`${relative(process.cwd(), file)}: reserved WGSL token '${identifier}'`);
      }
      if (identifier === '_' || identifier.startsWith('__')) {
        failures.push(`${relative(process.cwd(), file)}: forbidden WGSL identifier '${identifier}'`);
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
  console.error([...new Set(failures)].join('\n'));
  process.exit(1);
}
console.log(`Parsed and validated ${shaderCount} WGSL shader template strings against the WGSL reserved-token set.`);
