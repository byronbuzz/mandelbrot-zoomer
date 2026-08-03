const provenanceHelpers = /* wgsl */ `
const SEMANTIC_INVALID = 0u;
const SEMANTIC_ESCAPED = 1u;
const SEMANTIC_ANALYTIC = 2u;
const SEMANTIC_PROVISIONAL_CAP = 3u;
const SEMANTIC_FINAL_CAP = 4u;
const SEMANTIC_CONFLICT = 5u;
const SEMANTIC_ESCAPED_ESTIMATE = 6u;
const SEMANTIC_ANALYTIC_ESTIMATE = 7u;
const SEMANTIC_MASK = 7u;
const ORIGIN_SHIFT = 3u;
const ORIGIN_MASK = 3u;
const RANK_SHIFT = 5u;
const RANK_MASK = 127u;
const EVIDENCE_SHIFT = 12u;
const EVIDENCE_MASK = 1048575u;

fn semantic(packed: u32) -> u32 { return packed & SEMANTIC_MASK; }
fn origin(packed: u32) -> u32 { return (packed >> ORIGIN_SHIFT) & ORIGIN_MASK; }
fn rank(packed: u32) -> u32 { return (packed >> RANK_SHIFT) & RANK_MASK; }
fn evidence(packed: u32) -> u32 { return (packed >> EVIDENCE_SHIFT) & EVIDENCE_MASK; }
fn isDefinite(value: u32) -> bool {
  return value == SEMANTIC_ESCAPED || value == SEMANTIC_ANALYTIC;
}
fn isEstimate(value: u32) -> bool {
  return value == SEMANTIC_ESCAPED_ESTIMATE || value == SEMANTIC_ANALYTIC_ESTIMATE;
}
fn isDisplayDefinite(value: u32) -> bool { return isDefinite(value) || isEstimate(value); }
fn isCap(value: u32) -> bool {
  return value == SEMANTIC_PROVISIONAL_CAP || value == SEMANTIC_FINAL_CAP;
}
`;

export const atlasReprojectShader = /* wgsl */ `
struct Params {
  transform: vec4f,
  valid: u32,
  footprintRankDelta: i32,
  historyOrigin: u32,
  preserveFinalCap: u32,
}
struct VertexOutput { @builtin(position) position: vec4f, @location(0) uv: vec2f }
struct FragmentOutput {
  @location(0) colour: vec4f,
  @location(1) provenance: u32,
}
${provenanceHelpers}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var historyColour: texture_2d<f32>;
@group(0) @binding(3) var historyProvenance: texture_2d<u32>;

fn adjustedHistoryProvenance(packed: u32) -> u32 {
  let oldRank = i32(rank(packed));
  let adjustedRank = u32(clamp(oldRank + p.footprintRankDelta, 0, 127));
  let previousSemantic = semantic(packed);
  var adjustedSemantic = previousSemantic;
  if (p.preserveFinalCap == 0u) {
    if (previousSemantic == SEMANTIC_FINAL_CAP) {
      adjustedSemantic = SEMANTIC_PROVISIONAL_CAP;
    } else if (previousSemantic == SEMANTIC_ESCAPED) {
      adjustedSemantic = SEMANTIC_ESCAPED_ESTIMATE;
    } else if (previousSemantic == SEMANTIC_ANALYTIC) {
      adjustedSemantic = SEMANTIC_ANALYTIC_ESTIMATE;
    }
  }
  let retained = packed & ~(
    SEMANTIC_MASK | (ORIGIN_MASK << ORIGIN_SHIFT) | (RANK_MASK << RANK_SHIFT)
  );
  return retained
    | adjustedSemantic
    | ((p.historyOrigin & ORIGIN_MASK) << ORIGIN_SHIFT)
    | (adjustedRank << RANK_SHIFT);
}

@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  var uvs = array<vec2f, 3>(vec2f(0.0,1.0),vec2f(2.0,1.0),vec2f(0.0,-1.0));
  var out: VertexOutput; out.position=vec4f(positions[i],0.0,1.0); out.uv=uvs[i]; return out;
}

@fragment fn fragmentMain(input: VertexOutput) -> FragmentOutput {
  if (p.valid == 0u) { discard; }
  let sourceUv = vec2f(0.5) + p.transform.zw + (input.uv-vec2f(0.5))*p.transform.xy;
  if (any(sourceUv < vec2f(0.0)) || any(sourceUv > vec2f(1.0))) { discard; }
  let dimensions = vec2i(textureDimensions(historyProvenance));
  let coordinate = clamp(vec2i(floor(sourceUv * vec2f(dimensions))), vec2i(0), dimensions - vec2i(1));
  let provenance = adjustedHistoryProvenance(textureLoad(historyProvenance, coordinate, 0).x);
  if (semantic(provenance) == SEMANTIC_INVALID) { discard; }
  var out: FragmentOutput;
  out.colour = vec4f(textureSampleLevel(historyColour,linearSampler,sourceUv,0.0).rgb,1.0);
  out.provenance = provenance;
  return out;
}`;

export const atlasOverlayShader = /* wgsl */ `
struct Instance {
  rect: vec4f,
  atlasOrigin: vec2u,
  slot: u32,
  lease: u32,
  iterationEvidence: u32,
  capMode: u32,
  footprintRank: u32,
  targetIterations: u32,
}
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) tileUv: vec2f,
  @location(1) @interpolate(flat) atlasOrigin: vec2u,
  @location(2) @interpolate(flat) slotLease: vec2u,
  @location(3) @interpolate(flat) evidence: vec4u,
}
struct FragmentOutput {
  @location(0) colour: vec4f,
  @location(1) provenance: u32,
  @builtin(frag_depth) depth: f32,
}
${provenanceHelpers}
const ORIGIN_CURRENT = 3u;
@group(0) @binding(0) var<storage,read> instances: array<Instance>;
@group(0) @binding(1) var colourAtlas: texture_2d<f32>;
@group(0) @binding(2) var qualityAtlas: texture_2d<f32>;
@group(0) @binding(3) var evidenceAtlas: texture_2d<u32>;
@group(0) @binding(4) var<storage,read> leases: array<u32>;

fn packProvenance(status: u32, iterationEvidence: u32, footprintRank: u32) -> u32 {
  return status
    | (ORIGIN_CURRENT << ORIGIN_SHIFT)
    | (min(footprintRank, RANK_MASK) << RANK_SHIFT)
    | (min(iterationEvidence, EVIDENCE_MASK) << EVIDENCE_SHIFT);
}

fn currentCandidateDepth(status: u32, footprintRank: u32) -> f32 {
  let semanticTier = select(
    select(1u, 2u, status == SEMANTIC_FINAL_CAP),
    3u,
    isDisplayDefinite(status)
  );
  return f32((semanticTier << 7u) | min(footprintRank, RANK_MASK)) / 512.0;
}

@vertex fn vertexMain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
  var corners = array<vec2f,6>(
    vec2f(0.0,0.0),vec2f(1.0,0.0),vec2f(0.0,1.0),
    vec2f(0.0,1.0),vec2f(1.0,0.0),vec2f(1.0,1.0));
  let item=instances[instance]; let corner=corners[vertex]; let uv=mix(item.rect.xy,item.rect.zw,corner);
  var out: VertexOutput; out.position=vec4f(uv*vec2f(2.0,-2.0)+vec2f(-1.0,1.0),0.0,1.0);
  out.tileUv=corner; out.atlasOrigin=item.atlasOrigin;
  out.slotLease=vec2u(item.slot,item.lease);
  out.evidence=vec4u(item.iterationEvidence,item.capMode,item.footprintRank,item.targetIterations);
  return out;
}

@fragment fn fragmentMain(input: VertexOutput) -> FragmentOutput {
  if (leases[input.slotLease.x] != input.slotLease.y) { discard; }
  let local=clamp(vec2i(floor(input.tileUv*128.0)),vec2i(0),vec2i(127));
  let coordinate=vec2i(input.atlasOrigin)+local;
  let quality=textureLoad(qualityAtlas,coordinate,0);
  if (quality.x <= 0.0) { discard; }
  let pixelEvidence=textureLoad(evidenceAtlas,coordinate,0).x;
  var status = SEMANTIC_PROVISIONAL_CAP;
  let spatiallyExact=input.evidence.z >= 64u;
  if (quality.z < 0.375) {
    status = select(SEMANTIC_ESCAPED_ESTIMATE,SEMANTIC_ESCAPED,spatiallyExact);
  } else if (quality.z < 0.625) {
    status = select(SEMANTIC_ANALYTIC_ESTIMATE,SEMANTIC_ANALYTIC,spatiallyExact);
  } else if (input.evidence.y >= 2u && pixelEvidence >= input.evidence.w) {
    status = SEMANTIC_FINAL_CAP;
  }
  var out: FragmentOutput;
  out.colour=vec4f(textureLoad(colourAtlas,coordinate,0).rgb,1.0);
  out.provenance=packProvenance(status,pixelEvidence,input.evidence.z);
  out.depth=currentCandidateDepth(status,input.evidence.z);
  return out;
}`;

export const atlasMergeShader = /* wgsl */ `
struct Output { @builtin(position) position: vec4f, @location(0) uv: vec2f }
struct FragmentOutput { @location(0) colour: vec4f, @location(1) provenance: u32 }
struct MergeCounters {
  values: array<atomic<u32>, 16>,
}
${provenanceHelpers}
override TEST_INSTRUMENTATION: bool = false;
@group(0) @binding(0) var baseColour: texture_2d<f32>;
@group(0) @binding(1) var baseProvenance: texture_2d<u32>;
@group(0) @binding(2) var candidateColour: texture_2d<f32>;
@group(0) @binding(3) var candidateProvenance: texture_2d<u32>;
@group(0) @binding(4) var<storage,read_write> counters: MergeCounters;

fn compatiblePreference(base: u32, candidate: u32) -> i32 {
  let baseSemantic = semantic(base);
  let candidateSemantic = semantic(candidate);
  if (baseSemantic == SEMANTIC_INVALID) { return select(0, 1, candidateSemantic != SEMANTIC_INVALID); }
  if (candidateSemantic == SEMANTIC_INVALID) { return 0; }
  if (baseSemantic == SEMANTIC_CONFLICT) { return 0; }
  if (candidateSemantic == SEMANTIC_CONFLICT) { return 1; }
  let baseDefinite = isDefinite(baseSemantic);
  let candidateDefinite = isDefinite(candidateSemantic);
  if (baseDefinite && candidateDefinite) {
    if (baseSemantic != candidateSemantic) { return -1; }
    if (baseSemantic == SEMANTIC_ESCAPED && evidence(base) != evidence(candidate)) { return -2; }
  } else {
    if (baseDefinite) { return 0; }
    if (candidateDefinite) { return 1; }
  }
  let baseEstimate = isEstimate(baseSemantic);
  let candidateEstimate = isEstimate(candidateSemantic);
  if (baseEstimate && isCap(candidateSemantic)) { return 0; }
  if (candidateEstimate && isCap(baseSemantic)) { return 1; }
  if (baseSemantic == SEMANTIC_FINAL_CAP && candidateSemantic == SEMANTIC_PROVISIONAL_CAP) { return 0; }
  if (candidateSemantic == SEMANTIC_FINAL_CAP && baseSemantic == SEMANTIC_PROVISIONAL_CAP) { return 1; }

  if (isCap(baseSemantic) && isCap(candidateSemantic)) {
    if (evidence(candidate) != evidence(base)) {
      return select(0, 1, evidence(candidate) > evidence(base));
    }
  }
  if (rank(candidate) != rank(base)) { return select(0, 1, rank(candidate) > rank(base)); }
  if (origin(candidate) != origin(base)) { return select(0, 1, origin(candidate) > origin(base)); }
  if (evidence(candidate) != evidence(base)) {
    return select(0, 1, evidence(candidate) > evidence(base));
  }
  return 0;
}

fn conflictProvenance(base: u32, candidate: u32, reason: u32) -> u32 {
  let representative=select(base,candidate,
    origin(candidate) > origin(base)
    || (origin(candidate) == origin(base) && rank(candidate) > rank(base))
  );
  return (representative & ((ORIGIN_MASK << ORIGIN_SHIFT) | (RANK_MASK << RANK_SHIFT)))
    | SEMANTIC_CONFLICT
    | (reason << EVIDENCE_SHIFT);
}

@vertex fn vertexMain(@builtin(vertex_index) i: u32) -> Output {
  var positions=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  var uvs=array<vec2f,3>(vec2f(0.0,1.0),vec2f(2.0,1.0),vec2f(0.0,-1.0));
  var out: Output; out.position=vec4f(positions[i],0.0,1.0); out.uv=uvs[i]; return out;
}

@fragment fn fragmentMain(@builtin(position) position: vec4f) -> FragmentOutput {
  let coordinate=vec2i(position.xy);
  let base=textureLoad(baseProvenance,coordinate,0).x;
  let candidate=textureLoad(candidateProvenance,coordinate,0).x;
  let preference=compatiblePreference(base,candidate);
  let chooseCandidate=preference == 1;
  if (TEST_INSTRUMENTATION && semantic(candidate) != SEMANTIC_INVALID) {
    if (!chooseCandidate && preference >= 0) { atomicAdd(&counters.values[10],1u); }
    if (preference < 0) { atomicAdd(&counters.values[11],1u); }
    if (chooseCandidate && isDisplayDefinite(semantic(base)) && isCap(semantic(candidate))) {
      atomicAdd(&counters.values[8],1u);
      if ((semantic(base) == SEMANTIC_ESCAPED || semantic(base) == SEMANTIC_ESCAPED_ESTIMATE)
        && semantic(candidate) == SEMANTIC_PROVISIONAL_CAP) {
        atomicAdd(&counters.values[9],1u);
      }
    }
  }
  var out: FragmentOutput;
  let representativeCandidate=origin(candidate) > origin(base)
    || (origin(candidate) == origin(base) && rank(candidate) > rank(base));
  out.colour=select(
    textureLoad(baseColour,coordinate,0),
    textureLoad(candidateColour,coordinate,0),
    chooseCandidate || (preference < 0 && representativeCandidate)
  );
  out.provenance=select(
    select(base,candidate,chooseCandidate),
    conflictProvenance(base,candidate,select(1u,2u,preference == -2)),
    preference < 0
  );
  return out;
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

export const atlasContinuityReductionShader = /* wgsl */ `
struct Counters { values: array<atomic<u32>, 16> }
@group(0) @binding(0) var provenance: texture_2d<u32>;
@group(0) @binding(1) var<storage,read_write> counters: Counters;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dimensions=textureDimensions(provenance);
  if (gid.x >= dimensions.x || gid.y >= dimensions.y) { return; }
  let packed=textureLoad(provenance,vec2i(gid.xy),0).x;
  let status=packed & 7u;
  let source=(packed >> 3u) & 3u;
  atomicAdd(&counters.values[0],1u);
  if (status == 0u) { atomicAdd(&counters.values[1],1u); return; }
  if (source == 1u) { atomicAdd(&counters.values[2],1u); }
  if (source == 2u) { atomicAdd(&counters.values[3],1u); }
  if (source == 3u) { atomicAdd(&counters.values[4],1u); }
  if (status == 3u) { atomicAdd(&counters.values[5],1u); }
  if (status == 4u) { atomicAdd(&counters.values[6],1u); }
  if (status == 1u || status == 2u || status == 6u || status == 7u) {
    atomicAdd(&counters.values[7],1u);
  }
  if (status == 5u) { atomicAdd(&counters.values[12],1u); }
}
`;
