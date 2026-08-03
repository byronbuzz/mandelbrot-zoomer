export const persistentDirectShader = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  targetIterations: u32,
  chunkIterations: u32,
  center: vec2f,
  pixelStep: vec2f,
  origin: vec2f,
  _pad0: vec2f,
}

struct PixelState {
  z: vec2f,
  iteration: u32,
  status: u32,
}

const STATUS_ACTIVE: u32 = 0u;
const STATUS_ESCAPED: u32 = 1u;
const STATUS_INTERIOR: u32 = 2u;
const STATUS_PROVISIONAL: u32 = 3u;

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> states: array<PixelState>;
@group(0) @binding(2) var<storage, read_write> activeCount: atomic<u32>;

fn coordinate(x: u32, y: u32) -> vec2f {
  return p.center + p.origin + vec2f(f32(x), f32(y)) * p.pixelStep;
}

fn analyticInterior(c: vec2f) -> bool {
  let shiftedX = c.x - 0.25;
  let ySquared = c.y * c.y;
  let q = shiftedX * shiftedX + ySquared;
  let cardioid = q * (q + shiftedX) <= 0.25 * ySquared;
  let bulbX = c.x + 1.0;
  let bulb = bulbX * bulbX + ySquared <= 0.0625;
  return cardioid || bulb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let index = gid.y * p.width + gid.x;
  var state = states[index];
  if (state.status == STATUS_ESCAPED || state.status == STATUS_INTERIOR) { return; }

  let c = coordinate(gid.x, gid.y);
  if (state.iteration == 0u && analyticInterior(c)) {
    state.status = STATUS_INTERIOR;
    states[index] = state;
    return;
  }

  var z = state.z;
  var iteration = state.iteration;
  var escaped = false;
  var step = 0u;
  loop {
    if (step >= p.chunkIterations || iteration >= p.targetIterations) { break; }
    let zr = z.x;
    let zi = z.y;
    z = vec2f(
      fma(zr, zr, -(zi * zi)) + c.x,
      fma(2.0 * zr, zi, c.y)
    );
    iteration += 1u;
    step += 1u;
    if (dot(z, z) > 256.0) {
      escaped = true;
      break;
    }
  }

  state.z = z;
  state.iteration = iteration;
  if (escaped) {
    state.status = STATUS_ESCAPED;
  } else if (iteration >= p.targetIterations) {
    state.status = STATUS_PROVISIONAL;
  } else {
    state.status = STATUS_ACTIVE;
    atomicAdd(&activeCount, 1u);
  }
  states[index] = state;
}
`;
