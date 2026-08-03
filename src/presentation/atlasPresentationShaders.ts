export const atlasReprojectShader = /* wgsl */ `
struct Params { transform: vec4f, valid: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
struct Output { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var history: texture_2d<f32>;
@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> Output {
  var positions = array<vec2f, 3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  var uvs = array<vec2f, 3>(vec2f(0.0,1.0),vec2f(2.0,1.0),vec2f(0.0,-1.0));
  var out: Output; out.position=vec4f(positions[i],0.0,1.0); out.uv=uvs[i]; return out;
}
@fragment fn fragmentMain(input: Output) -> @location(0) vec4f {
  if (p.valid == 0u) { return vec4f(0.008,0.01,0.014,1.0); }
  let sourceUv = vec2f(0.5) + p.transform.zw + (input.uv-vec2f(0.5))*p.transform.xy;
  if (any(sourceUv < vec2f(0.0)) || any(sourceUv > vec2f(1.0))) {
    return vec4f(0.008,0.01,0.014,1.0);
  }
  return vec4f(textureSampleLevel(history,linearSampler,sourceUv,0.0).rgb,1.0);
}`;

export const atlasOverlayShader = /* wgsl */ `
struct Instance { rect: vec4f, atlasOrigin: vec2u, slot: u32, lease: u32 }
struct Output {
  @builtin(position) position: vec4f,
  @location(0) tileUv: vec2f,
  @location(1) @interpolate(flat) atlasOrigin: vec2u,
  @location(2) @interpolate(flat) slotLease: vec2u,
}
@group(0) @binding(0) var<storage,read> instances: array<Instance>;
@group(0) @binding(1) var colourAtlas: texture_2d<f32>;
@group(0) @binding(2) var qualityAtlas: texture_2d<f32>;
@group(0) @binding(3) var<storage,read> leases: array<u32>;
@vertex fn vertexMain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Output {
  var corners = array<vec2f,6>(
    vec2f(0.0,0.0),vec2f(1.0,0.0),vec2f(0.0,1.0),
    vec2f(0.0,1.0),vec2f(1.0,0.0),vec2f(1.0,1.0));
  let item=instances[instance]; let corner=corners[vertex]; let uv=mix(item.rect.xy,item.rect.zw,corner);
  var out: Output; out.position=vec4f(uv*vec2f(2.0,-2.0)+vec2f(-1.0,1.0),0.0,1.0);
  out.tileUv=corner; out.atlasOrigin=item.atlasOrigin;
  out.slotLease=vec2u(item.slot,item.lease); return out;
}
@fragment fn fragmentMain(input: Output) -> @location(0) vec4f {
  if (leases[input.slotLease.x] != input.slotLease.y) { discard; }
  let local=clamp(vec2i(floor(input.tileUv*128.0)),vec2i(0),vec2i(127));
  let coordinate=vec2i(input.atlasOrigin)+local;
  let quality=textureLoad(qualityAtlas,coordinate,0).x;
  if (quality <= 0.0) { discard; }
  return vec4f(textureLoad(colourAtlas,coordinate,0).rgb,1.0);
}`;

export const atlasPresentShader = /* wgsl */ `
struct Output { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var image: texture_2d<f32>;
@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> Output {
  var positions=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  var uvs=array<vec2f,3>(vec2f(0.0,1.0),vec2f(2.0,1.0),vec2f(0.0,-1.0));
  var out: Output; out.position=vec4f(positions[i],0.0,1.0); out.uv=uvs[i]; return out;
}
@fragment fn fragmentMain(input: Output) -> @location(0) vec4f {
  return vec4f(textureSampleLevel(image,linearSampler,input.uv,0.0).rgb,1.0);
}`;
