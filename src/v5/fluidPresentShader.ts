export const fluidPresentShader = /* wgsl */ `
struct PresentParams {
  historyScale: vec2f,
  historyOffset: vec2f,
  newWeight: f32,
  mode: u32,
  _pad0: vec2u,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> p: PresentParams;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var newImage: texture_2d<f32>;
@group(0) @binding(3) var historyImage: texture_2d<f32>;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 1.0),
    vec2f(2.0, 1.0),
    vec2f(0.0, -1.0)
  );
  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];
  return output;
}

fn checker(position: vec4f) -> vec3f {
  let x = u32(max(0.0, floor(position.x / 8.0)));
  let y = u32(max(0.0, floor(position.y / 8.0)));
  let level = .025 + f32((x + y) & 1u) * .015;
  return vec3f(level);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let historyUv = vec2f(.5) + p.historyOffset + (input.uv - vec2f(.5)) * p.historyScale;
  let historyInside = all(historyUv >= vec2f(0.0)) && all(historyUv <= vec2f(1.0));

  if (p.mode == 1u) {
    let clampedHistoryUv = clamp(historyUv, vec2f(0.0), vec2f(1.0));
    let history = textureSampleLevel(historyImage, imageSampler, clampedHistoryUv, 0.0);
    let historyColour = select(checker(input.position), history.rgb, history.a >= .5);
    if (historyInside) { return vec4f(historyColour, 1.0); }

    let overflow = max(abs(historyUv - vec2f(.5)) - vec2f(.5), vec2f(0.0));
    let edgeWeight = .38 * exp2(-32.0 * length(overflow));
    let background = vec3f(.008, .01, .014);
    return vec4f(mix(background, historyColour, edgeWeight), 1.0);
  }

  let current = textureSampleLevel(newImage, imageSampler, input.uv, 0.0);
  let currentColour = select(checker(input.position), current.rgb, current.a >= .5);
  if (!historyInside || p.newWeight >= .999) {
    return vec4f(currentColour, 1.0);
  }

  let history = textureSampleLevel(historyImage, imageSampler, historyUv, 0.0);
  let historyColour = select(checker(input.position), history.rgb, history.a >= .5);

  if (current.a < .5 && history.a >= .5) {
    return vec4f(historyColour, 1.0);
  }
  if (history.a < .5 && current.a >= .5) {
    return vec4f(currentColour, 1.0);
  }

  return vec4f(mix(historyColour, currentColour, p.newWeight), 1.0);
}`;
