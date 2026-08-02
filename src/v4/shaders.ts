import { computeShader as baseComputeShader } from './shadersBase';

export const computeShader = baseComputeShader
  .replace('  _pad0: u32,\n}', '  rowStart: u32,\n}')
  .replace('  let id = gid.xy;\n', '  let id = vec2u(gid.x, gid.y + p.rowStart);\n');

export { presentShader } from './shadersBase';
export { composeShader } from './composeShader';
