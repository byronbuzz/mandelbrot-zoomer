export const composeShader = /* wgsl */ `
const TILE_COLUMNS = 16u;
const TILE_ROWS = 16u;
const REASON_EXHAUSTED = 1u;
const REASON_MAGNITUDE = 2u;
const REASON_NON_FINITE = 3u;
const REASON_REBASE = 4u;

struct ComposeParams {
  width: u32,
  height: u32,
}
struct Telemetry {
  unresolved: atomic<u32>,
  exhausted: atomic<u32>,
  magnitudeGuard: atomic<u32>,
  nonFinite: atomic<u32>,
  rebaseFailures: atomic<u32>,
  tiles: array<atomic<u32>, 256>,
  maxExponentBits: atomic<u32>,
}
@group(0) @binding(0) var<uniform> p: ComposeParams;
@group(0) @binding(1) var baseTex: texture_2d<f32>;
@group(0) @binding(2) var repairTex: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> telemetry: Telemetry;

fn reasonFromAlpha(alpha: f32) -> u32 {
  if (alpha < .0625) { return REASON_EXHAUSTED; }
  if (alpha < .1875) { return REASON_MAGNITUDE; }
  if (alpha < .3125) { return REASON_NON_FINITE; }
  return REASON_REBASE;
}
fn recordUnresolved(id: vec2u, reason: u32) {
  atomicAdd(&telemetry.unresolved, 1u);
  if (reason == REASON_EXHAUSTED) { atomicAdd(&telemetry.exhausted, 1u); }
  if (reason == REASON_MAGNITUDE) { atomicAdd(&telemetry.magnitudeGuard, 1u); }
  if (reason == REASON_NON_FINITE) { atomicAdd(&telemetry.nonFinite, 1u); }
  if (reason == REASON_REBASE) { atomicAdd(&telemetry.rebaseFailures, 1u); }
  let tileX = min((id.x * TILE_COLUMNS) / max(p.width, 1u), TILE_COLUMNS - 1u);
  let tileY = min((id.y * TILE_ROWS) / max(p.height, 1u), TILE_ROWS - 1u);
  atomicAdd(&telemetry.tiles[tileY * TILE_COLUMNS + tileX], 1u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.xy;
  if (id.x >= p.width || id.y >= p.height) { return; }
  let base = textureLoad(baseTex, vec2i(id), 0);
  let repair = textureLoad(repairTex, vec2i(id), 0);
  let useRepair = base.a < .5 && repair.a >= .5;
  let result = select(base, repair, useRepair);
  textureStore(outTex, vec2i(id), result);
  if (result.a < .5) { recordUnresolved(id, reasonFromAlpha(result.a)); }
}`;
