import type { CameraSnapshot } from '../camera/types';
import type { AtlasSlot, AcceptedTileAtlas } from './acceptedTileAtlas';
import {
  atlasContinuityReductionShader,
  atlasMergeShader,
  atlasOverlayShader,
  atlasPresentShader,
  atlasReprojectShader
} from './atlasPresentationShaders';
import {
  fixedDifferenceOverScale,
  packTransform,
  scaleRatio,
  transformIsFinite,
  type PackedTransform
} from './presentationMath';

const REPROJECT_UNIFORM_BYTES = 32;
const REPROJECT_UNIFORM_STRIDE = 256;
const MAX_HISTORY_SOURCES = 12;
const MAX_INSTANCES = 512;
const INSTANCE_WORDS = 12;
const SOURCE_TEXEL_ERROR_LIMIT = 0.01;
const FOOTPRINT_RANK_CENTRE = 64;
const FOOTPRINT_RANK_STEPS_PER_OCTAVE = 8;
const SNAPSHOT_INTERVAL_OCTAVES = 3;
const RETAINED_MEMORY_BUDGET_BYTES = 96 * 1024 * 1024;
const PRESENTATION_MEMORY_BUDGET_BYTES = 384 * 1024 * 1024;
const MAX_SNAPSHOTS = 8;
const CONTINUITY_COUNTER_WORDS = 16;
const CONTINUITY_COUNTER_BYTES = CONTINUITY_COUNTER_WORDS * Uint32Array.BYTES_PER_ELEMENT;
const CONTINUITY_READBACK_SLOTS = 8;
const CONTINUITY_FRAME_HISTORY = 512;
const CONTINUITY_TIMEOUT_MS = 10_000;

type ExactView = Readonly<{
  camera: CameraSnapshot;
  aspect: number;
  width: number;
  height: number;
  targetIterations: number;
  palettePhase: number;
  contentRevision: number;
}>;

type FrameSurface = Readonly<{ colour: GPUTexture; provenance: GPUTexture }>;

type ResourceSet = {
  surfaces: readonly [FrameSurface, FrameSurface];
  candidate: FrameSurface;
  candidateDepth: GPUTexture;
  width: number;
  height: number;
  epoch: number;
};

type RollingHead = Readonly<{ surface: FrameSurface; owner: ResourceSet; view: ExactView }>;
type Snapshot = Readonly<{ surface: FrameSurface; view: ExactView; generation: number }>;
type HistorySource = Readonly<{
  surface: FrameSurface;
  view: ExactView;
  owner: ResourceSet | null;
  origin: 1 | 2;
}>;
type ReadbackSlot = { buffer: GPUBuffer; busy: boolean };
type PendingCheckpoint = {
  snapshot: Snapshot;
  frameId: number;
  validation: Promise<GPUError | null>;
};

export type ContinuityFrame = Readonly<{
  frameId: number;
  viewRevision: number;
  requestId: number;
  completedBatchRevision: number;
  totalPixels: number;
  invalidPixels: number;
  historyPixels: number;
  retainedPixels: number;
  currentPixels: number;
  provisionalCapPixels: number;
  finalCapPixels: number;
  terminalPixels: number;
  qualityRegressionPixels: number;
  escapedToProvisionalBlackPixels: number;
  candidateRejectedLowerQualityPixels: number;
  semanticConflictEvents: number;
  conflictPixels: number;
  checkpointEligible: boolean;
  droppedReadbacks: number;
}>;

export type AtlasInstance = Readonly<{
  transform: PackedTransform;
  slot: AtlasSlot;
  iterationFrontier: number;
  capMode: 0 | 1 | 2;
  targetIterations: number;
}>;

export type AtlasPresenterDiagnostics = Readonly<{
  frames: number;
  historyFrames: number;
  fallbackFrames: number;
  anchorPromotions: number;
  instanceCount: number;
  resourceEpoch: number;
  worstReprojectionErrorTexels: number;
  lastFrameCpuMs: number;
  validationErrors: number;
  rollingFrames: number;
  snapshotCount: number;
  reducedFrame: number;
  totalPixels: number;
  invalidPixels: number;
  historyPixels: number;
  retainedPixels: number;
  currentPixels: number;
  provisionalCapPixels: number;
  finalCapPixels: number;
  terminalPixels: number;
  qualityRegressionPixels: number;
  escapedToProvisionalBlackPixels: number;
  candidateRejectedLowerQualityPixels: number;
  semanticConflictEvents: number;
  conflictPixels: number;
  droppedReadbacks: number;
}>;

function sameResolvedView(left: ExactView | null, right: ExactView): boolean {
  if (!left) return false;
  return left.aspect === right.aspect
    && left.width === right.width
    && left.height === right.height
    && left.targetIterations === right.targetIterations
    && left.camera.centerX.raw === right.camera.centerX.raw
    && left.camera.centerX.bits === right.camera.centerX.bits
    && left.camera.centerY.raw === right.camera.centerY.raw
    && left.camera.centerY.bits === right.camera.centerY.bits
    && left.camera.scale.mantissa === right.camera.scale.mantissa
    && left.camera.scale.exponent === right.camera.scale.exponent;
}

function sameCheckpointIdentity(left: ExactView | null, right: ExactView): boolean {
  return sameResolvedView(left, right)
    && left?.palettePhase === right.palettePhase
    && left?.contentRevision === right.contentRevision;
}

function f32SourceCoordinate(scale: number, offset: number, uv: number): number {
  return Math.fround(Math.fround(0.5 + offset) + Math.fround(Math.fround(uv - 0.5) * scale));
}

function historyTransform(source: ExactView, target: ExactView): PackedTransform {
  const scaleY = scaleRatio(target.camera.scale, source.camera.scale);
  return {
    scaleX: scaleY * target.aspect / source.aspect,
    scaleY,
    offsetX: fixedDifferenceOverScale(target.camera.centerX, source.camera.centerX, source.camera.scale) / source.aspect,
    offsetY: fixedDifferenceOverScale(target.camera.centerY, source.camera.centerY, source.camera.scale)
  };
}

function admitHistory(transform: PackedTransform, source: ExactView, target: ExactView): {
  accepted: boolean;
  packed: PackedTransform;
  error: number;
} {
  const packed = packTransform(transform);
  const samples = [
    [0.5, 0.5],
    [0.5 / target.width, 0.5 / target.height],
    [(target.width - 0.5) / target.width, 0.5 / target.height],
    [0.5 / target.width, (target.height - 0.5) / target.height],
    [(target.width - 0.5) / target.width, (target.height - 0.5) / target.height]
  ] as const;
  let error = 0;
  for (const [u, v] of samples) {
    const expectedX = 0.5 + transform.offsetX + (u - 0.5) * transform.scaleX;
    const expectedY = 0.5 + transform.offsetY + (v - 0.5) * transform.scaleY;
    error = Math.max(
      error,
      Math.abs(expectedX - f32SourceCoordinate(packed.scaleX, packed.offsetX, u)) * source.width,
      Math.abs(expectedY - f32SourceCoordinate(packed.scaleY, packed.offsetY, v)) * source.height
    );
  }
  return { accepted: transformIsFinite(packed) && error <= SOURCE_TEXEL_ERROR_LIMIT, packed, error };
}

function footprintRank(rectWidth: number, rectHeight: number, width: number, height: number): number {
  const projectedPixels = Math.max(Number.MIN_VALUE, rectWidth * width / 128, rectHeight * height / 128);
  return Math.max(0, Math.min(127, Math.round(
    FOOTPRINT_RANK_CENTRE - Math.log2(projectedPixels) * FOOTPRINT_RANK_STEPS_PER_OCTAVE
  )));
}

function footprintRankDelta(transform: PackedTransform, source: ExactView, target: ExactView): number {
  const xRatio = source.width * Math.abs(transform.scaleX) / target.width;
  const yRatio = source.height * Math.abs(transform.scaleY) / target.height;
  const conservativeRatio = Math.max(Number.MIN_VALUE, Math.min(xRatio, yRatio));
  return Math.max(-127, Math.min(127, Math.round(
    Math.log2(conservativeRatio) * FOOTPRINT_RANK_STEPS_PER_OCTAVE
  )));
}

function viewScaleOctaves(view: ExactView): number {
  return view.camera.scale.exponent + Math.log2(view.camera.scale.mantissa);
}

function transformCoverage(transform: PackedTransform): number {
  if (!transformIsFinite(transform) || transform.scaleX <= 0 || transform.scaleY <= 0) return 0;
  const left = 0.5 + transform.offsetX - 0.5 * transform.scaleX;
  const right = 0.5 + transform.offsetX + 0.5 * transform.scaleX;
  const top = 0.5 + transform.offsetY - 0.5 * transform.scaleY;
  const bottom = 0.5 + transform.offsetY + 0.5 * transform.scaleY;
  const coveredX = Math.max(0, Math.min(1, right) - Math.max(0, left)) / transform.scaleX;
  const coveredY = Math.max(0, Math.min(1, bottom) - Math.max(0, top)) / transform.scaleY;
  return Math.max(0, Math.min(1, coveredX * coveredY));
}

function surfaceBytes(view: ExactView): number {
  return Math.max(1, view.width * view.height * 8);
}

export class AtlasHistoryPresenter {
  private readonly reprojectPipeline: GPURenderPipeline;
  private readonly overlayPipeline: GPURenderPipeline;
  private readonly mergePipeline: GPURenderPipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly reprojectUniform: GPUBuffer;
  private readonly instanceBuffer: GPUBuffer;
  private readonly continuityCounterBuffer: GPUBuffer;
  private readonly continuityReadbacks: ReadbackSlot[];
  private resources: ResourceSet | null = null;
  private rollingHead: RollingHead | null = null;
  private stableCheckpoint: Snapshot | null = null;
  private snapshots: Snapshot[] = [];
  private pendingCheckpoint: PendingCheckpoint | null = null;
  private readonly retiringSets = new Set<ResourceSet>();
  private resourceEpoch = 0;
  private snapshotGeneration = 0;
  private frames = 0;
  private historyFrames = 0;
  private fallbackFrames = 0;
  private anchorPromotions = 0;
  private instanceCount = 0;
  private worstReprojectionErrorTexels = 0;
  private lastFrameCpuMs = 0;
  private validationErrors = 0;
  private destroyed = false;
  private continuityReadbackCursor = 0;
  private continuityValues = new Uint32Array(CONTINUITY_COUNTER_WORDS);
  private continuityReducedFrame = 0;
  private droppedReadbacks = 0;
  private readonly continuityFrames: ContinuityFrame[] = [];
  private continuityWaiters: Array<{
    afterFrame: number;
    resolve: (frame: ContinuityFrame) => void;
    reject: (error: Error) => void;
    timeout: ReturnType<typeof setTimeout>;
  }> = [];

  private constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: GPUTextureFormat,
    reprojectPipeline: GPURenderPipeline,
    overlayPipeline: GPURenderPipeline,
    mergePipeline: GPURenderPipeline,
    presentPipeline: GPURenderPipeline,
    private readonly continuityPipeline: GPUComputePipeline,
    private readonly continuityTest: boolean
  ) {
    this.reprojectPipeline = reprojectPipeline;
    this.overlayPipeline = overlayPipeline;
    this.mergePipeline = mergePipeline;
    this.presentPipeline = presentPipeline;
    this.sampler = device.createSampler({
      minFilter: 'linear', magFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'
    });
    this.reprojectUniform = device.createBuffer({
      label: 'atlas-history-reprojection-uniform-ring',
      size: REPROJECT_UNIFORM_STRIDE * MAX_HISTORY_SOURCES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.instanceBuffer = device.createBuffer({
      label: 'atlas-presentation-instances',
      size: MAX_INSTANCES * INSTANCE_WORDS * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.continuityCounterBuffer = device.createBuffer({
      label: 'presentation-continuity-counters',
      size: CONTINUITY_COUNTER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    this.continuityReadbacks = Array.from({ length: CONTINUITY_READBACK_SLOTS }, (_, index) => ({
      buffer: device.createBuffer({
        label: `presentation-continuity-readback-${index}`,
        size: CONTINUITY_COUNTER_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      }),
      busy: false
    }));
  }

  static async create(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat): Promise<AtlasHistoryPresenter> {
    const reprojectModule = device.createShaderModule({ code: atlasReprojectShader });
    const overlayModule = device.createShaderModule({ code: atlasOverlayShader });
    const mergeModule = device.createShaderModule({ code: atlasMergeShader });
    const presentModule = device.createShaderModule({ code: atlasPresentShader });
    const continuityModule = device.createShaderModule({ code: atlasContinuityReductionShader });
    const continuityTest = new URLSearchParams(location.search).get('continuityTest') === '1';
    await Promise.all([
      this.assertShaderValid(reprojectModule, 'history candidate reprojection'),
      this.assertShaderValid(overlayModule, 'semantic atlas candidate'),
      this.assertShaderValid(mergeModule, 'explicit provenance merge'),
      this.assertShaderValid(presentModule, 'canvas presentation'),
      this.assertShaderValid(continuityModule, 'continuity reduction')
    ]);
    const composeTargets: GPUColorTargetState[] = [{ format: 'rgba8unorm' }, { format: 'r32uint' }];
    const [reprojectPipeline, overlayPipeline, mergePipeline, presentPipeline, continuityPipeline] = await Promise.all([
      device.createRenderPipelineAsync({
        layout: 'auto', vertex: { module: reprojectModule, entryPoint: 'vertexMain' },
        fragment: { module: reprojectModule, entryPoint: 'fragmentMain', targets: composeTargets },
        primitive: { topology: 'triangle-list' }
      }),
      device.createRenderPipelineAsync({
        layout: 'auto', vertex: { module: overlayModule, entryPoint: 'vertexMain' },
        fragment: { module: overlayModule, entryPoint: 'fragmentMain', targets: composeTargets },
        depthStencil: {
          format: 'depth32float',
          depthWriteEnabled: true,
          depthCompare: 'greater-equal'
        },
        primitive: { topology: 'triangle-list' }
      }),
      device.createRenderPipelineAsync({
        layout: 'auto', vertex: { module: mergeModule, entryPoint: 'vertexMain' },
        fragment: {
          module: mergeModule,
          entryPoint: 'fragmentMain',
          constants: { TEST_INSTRUMENTATION: continuityTest ? 1 : 0 },
          targets: composeTargets
        },
        primitive: { topology: 'triangle-list' }
      }),
      device.createRenderPipelineAsync({
        layout: 'auto', vertex: { module: presentModule, entryPoint: 'vertexMain' },
        fragment: { module: presentModule, entryPoint: 'fragmentMain', targets: [{ format }] },
        primitive: { topology: 'triangle-list' }
      }),
      device.createComputePipelineAsync({
        layout: 'auto', compute: { module: continuityModule, entryPoint: 'main' }
      })
    ]);
    return new AtlasHistoryPresenter(
      device, context, format,
      reprojectPipeline, overlayPipeline, mergePipeline, presentPipeline,
      continuityPipeline, continuityTest
    );
  }

  get diagnostics(): AtlasPresenterDiagnostics {
    return {
      frames: this.frames,
      historyFrames: this.historyFrames,
      fallbackFrames: this.fallbackFrames,
      anchorPromotions: this.anchorPromotions,
      instanceCount: this.instanceCount,
      resourceEpoch: this.resourceEpoch,
      worstReprojectionErrorTexels: this.worstReprojectionErrorTexels,
      lastFrameCpuMs: this.lastFrameCpuMs,
      validationErrors: this.validationErrors,
      rollingFrames: Math.max(0, this.frames - this.fallbackFrames),
      snapshotCount: this.snapshots.length + (this.stableCheckpoint ? 1 : 0),
      reducedFrame: this.continuityReduceß®·¶‰žËkºwµçeÑ•´¹¥¹ÍÑ…¹”¹Í±½Ð¹à°(€€€€€€€¥Ñ•´¹¥¹ÍÑ…¹”¹Í±½Ð¹ä°(€€€€€€€¥Ñ•´¹¥¹ÍÑ…¹”¹Í±½Ð¹¥¹‘•à°(€€€€€€€¥Ñ•´¹¥¹ÍÑ…¹”¹Í±½Ð¹±•…Í”°(€€€€€€€5…Ñ ¹µ…à À°5…Ñ ¹™±½½È¡¥Ñ•´¹¥¹ÍÑ…¹”¹¥Ñ•É…Ñ¥½¹É½¹Ñ¥•È¤¤°(€€€€€€€¥Ñ•´¹¥¹ÍÑ…¹”¹…Á5½‘”°(€€€€€€€¥Ñ•´¹É…¹¬°(€€€€€€€5…Ñ ¹µ…à À°5…Ñ ¹™±½½È¡¥Ñ•´¹¥¹ÍÑ…¹”¹Ñ…É•Ñ%Ñ•É…Ñ¥½¹Ì¤¤(€€€€€t°‰…Í”€¬€Ð¤ì(€€€ô¤ì(€€€É•ÑÕÉ¸ì‘…Ñ„°½Õ¹Ðè…•ÁÑ•¹±•¹Ñ ôì(€ô((€ÁÉ¥Ù…Ñ”¥¹¥Ñ¥…±!•…¡É•Í½ÕÉ•ÌèI•Í½ÕÉ•M•Ð°É½±±¥¹œèI½±±¥¹!•…ð¹Õ±°¤èÉ…µ•MÕÉ™…”ì(€€€¥˜€¡É½±±¥¹œü¹½Ý¹•È€ôôôÉ•Í½ÕÉ•Ì€˜˜É½±±¥¹œ¹ÍÕÉ™…”€ôôôÉ•Í½ÕÉ•Ì¹ÍÕÉ™…•ÍlÁt¤É•ÑÕÉ¸É•Í½ÕÉ•Ì¹ÍÕÉ™…•ÍlÅtì(€€€É•ÑÕÉ¸É•Í½ÕÉ•Ì¹ÍÕÉ™…•ÍlÁtì(€ô((€ÁÉ¥Ù…Ñ”½Ñ¡•ÉMÕÉ™…”¡É•Í½ÕÉ•ÌèI•Í½ÕÉ•M•Ð°ÕÉÉ•¹ÐèÉ…µ•MÕÉ™…”¤èÉ…µ•MÕÉ™…”ì(€€€É•ÑÕÉ¸ÕÉÉ•¹Ð€ôôôÉ•Í½ÕÉ•Ì¹ÍÕÉ™…•ÍlÁt€üÉ•Í½ÕÉ•Ì¹ÍÕÉ™…•ÍlÅt€èÉ•Í½ÕÉ•Ì¹ÍÕÉ™…•ÍlÁtì(€ô((€ÁÉ¥Ù…Ñ”±•…ÉMÕÉ™…”¡•¹½‘•ÈèAU½µµ…¹‘¹½‘•È°ÍÕÉ™…”èÉ…µ•MÕÉ™…”¤èÙ½¥ì(€€€½¹ÍÐÁ…ÍÌ€ô•¹½‘•È¹‰•¥¹I•¹‘•ÉA…ÍÌ¡ì½±½ÉÑÑ…¡µ•¹ÑÌèÑ¡¥Ì¹ÍÕÉ™…•ÑÑ…¡µ•¹ÑÌ¡ÍÕÉ™…”¤ô¤ì(€€€Á…ÍÌ¹•¹ ¤ì(€ô((€ÁÉ¥Ù…Ñ”É•¹‘•É!¥ÍÑ½Éå…¹‘¥‘…Ñ” (€€€•¹½‘•ÈèAU½µµ…¹‘¹½‘•È°(€€€‘•ÍÑ¥¹…Ñ¥½¸èÉ…µ•MÕÉ™…”°(€€€Í½ÕÉ”è!¥ÍÑ½ÉåM½ÕÉ”€˜ìÑÉ…¹Í™½É´èA…­•‘QÉ…¹Í™½É´ô°(€€€Ñ…É•Ðèá…ÑY¥•Ü°(€€€Õ¹¥™½Éµ%¹‘•àè¹Õµ‰•È(€€¤èÙ½¥ì(€€€½¹ÍÐ½™™Í•Ð€ôÕ¹¥™½Éµ%¹‘•à€¨IAI=)Q}U9%=I5}MQI%ì(€€€Ñ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹ÝÉ¥Ñ•	Õ™™•È¡Ñ¡¥Ì¹É•ÁÉ½©•ÑU¹¥™½É´°½™™Í•Ð°¹•Ü±½…ÐÌÉÉÉ…ä¡l(€€€€€Í½ÕÉ”¹ÑÉ…¹Í™½É´¹Í…±•`°Í½ÕÉ”¹ÑÉ…¹Í™½É´¹Í…±•d°(€€€€€Í½ÕÉ”¹ÑÉ…¹Í™½É´¹½™™Í•Ñ`°Í½ÕÉ”¹ÑÉ…¹Í™½É´¹½™™Í•Ñd(€€€t¤¤ì(€€€Ñ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹ÝÉ¥Ñ•	Õ™™•È¡Ñ¡¥Ì¹É•ÁÉ½©•ÑU¹¥™½É´°½™™Í•Ð€¬€ÄØ°¹•ÜU¥¹ÐÌÉÉÉ…ä¡l(€€€€€€Ä°(€€€€€™½½ÑÁÉ¥¹ÑI…¹­•±Ñ„¡Í½ÕÉ”¹ÑÉ…¹Í™½É´°Í½ÕÉ”¹Ù¥•Ü°Ñ…É•Ð¤€øøø€À°(€€€€€Í½ÕÉ”¹½É¥¥¸°(€€€€€Í…µ•I•Í½±Ù•‘Y¥•Ü¡Í½ÕÉ”¹Ù¥•Ü°Ñ…É•Ð¤€ü€Ä€è€À(€€€t¤¤ì(€€€½¹ÍÐÉ½ÕÀ€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹É•ÁÉ½©•ÑA¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¡¥Ì¹É•ÁÉ½©•ÑU¹¥™½É´°½™™Í•Ð°Í¥é”èIAI=)Q}U9%=I5}	eQLôô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èÑ¡¥Ì¹Í…µÁ±•Èô°(€€€€€€€ì‰¥¹‘¥¹œè€È°É•Í½ÕÉ”èÍ½ÕÉ”¹ÍÕÉ™…”¹½±½ÕÈ¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ì°É•Í½ÕÉ”èÍ½ÕÉ”¹ÍÕÉ™…”¹ÁÉ½Ù•¹…¹”¹É•…Ñ•Y¥•Ü ¤ô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐÁ…ÍÌ€ô•¹½‘•È¹‰•¥¹I•¹‘•ÉA…ÍÌ¡ì½±½ÉÑÑ…¡µ•¹ÑÌèÑ¡¥Ì¹ÍÕÉ™…•ÑÑ…¡µ•¹ÑÌ¡‘•ÍÑ¥¹…Ñ¥½¸¤ô¤ì(€€€Á…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹É•ÁÉ½©•ÑA¥Á•±¥¹”¤ì(€€€Á…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°É½ÕÀ¤ì(€€€Á…ÍÌ¹‘É…Ü Ì¤ì(€€€Á…ÍÌ¹•¹ ¤ì(€ô((€ÁÉ¥Ù…Ñ”É•¹‘•ÉÑ±…Í…¹‘¥‘…Ñ” (€€€•¹½‘•ÈèAU½µµ…¹‘¹½‘•È°(€€€É•Í½ÕÉ•ÌèI•Í½ÕÉ•M•Ð°(€€€É½ÕÀèAU	¥¹‘É½ÕÀ°(€€€¥¹ÍÑ…¹•½Õ¹Ðè¹Õµ‰•È(€€¤èÙ½¥ì(€€€½¹ÍÐÁ…ÍÌ€ô•¹½‘•È¹‰•¥¹I•¹‘•ÉA…ÍÌ¡ì(€€€€€½±½ÉÑÑ…¡µ•¹ÑÌèÑ¡¥Ì¹ÍÕÉ™…•ÑÑ…¡µ•¹ÑÌ¡É•Í½ÕÉ•Ì¹…¹‘¥‘…Ñ”¤°(€€€€€‘•ÁÑ¡MÑ•¹¥±ÑÑ…¡µ•¹Ðèì(€€€€€€€Ù¥•ÜèÉ•Í½ÕÉ•Ì¹…¹‘¥‘…Ñ••ÁÑ ¹É•…Ñ•Y¥•Ü ¤°(€€€€€€€‘•ÁÑ¡±•…ÉY…±Õ”è€À°(€€€€€€€‘•ÁÑ¡1½…‘=Àè€±•…Èœ°(€€€€€€€‘•ÁÑ¡MÑ½É•=Àè€‘¥Í…Éœ(€€€€€ô(€€€ô¤ì(€€€Á…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹½Ù•É±…åA¥Á•±¥¹”¤ì(€€€Á…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°É½ÕÀ¤ì(€€€Á…ÍÌ¹‘É…Ü Ø°¥¹ÍÑ…¹•½Õ¹Ð¤ì(€€€Á…ÍÌ¹•¹ ¤ì(€ô((€ÁÉ¥Ù…Ñ”µ•É•MÕÉ™…•Ì (€€€•¹½‘•ÈèAU½µµ…¹‘¹½‘•È°(€€€‰…Í”èÉ…µ•MÕÉ™…”°(€€€…¹‘¥‘…Ñ”èÉ…µ•MÕÉ™…”°(€€€½ÕÑÁÕÐèÉ…µ•MÕÉ™…”(€€¤èÙ½¥ì(€€€½¹ÍÐÉ½ÕÀ€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹µ•É•A¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”è‰…Í”¹½±½ÕÈ¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”è‰…Í”¹ÁÉ½Ù•¹…¹”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€È°É•Í½ÕÉ”è…¹‘¥‘…Ñ”¹½±½ÕÈ¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ì°É•Í½ÕÉ”è…¹‘¥‘…Ñ”¹ÁÉ½Ù•¹…¹”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ð°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¡¥Ì¹½¹Ñ¥¹Õ¥Ñå½Õ¹Ñ•É	Õ™™•Èôô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐÁ…ÍÌ€ô•¹½‘•È¹‰•¥¹I•¹‘•ÉA…ÍÌ¡ì½±½ÉÑÑ…¡µ•¹ÑÌèÑ¡¥Ì¹ÍÕÉ™…•ÑÑ…¡µ•¹ÑÌ¡½ÕÑÁÕÐ¤ô¤ì(€€€Á…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹µ•É•A¥Á•±¥¹”¤ì(€€€Á…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°É½ÕÀ¤ì(€€€Á…ÍÌ¹‘É…Ü Ì¤ì(€€€Á…ÍÌ¹•¹ ¤ì(€ô((€ÁÉ¥Ù…Ñ”É•¹‘•É…¹Ù…Ì¡•¹½‘•ÈèAU½µµ…¹‘¹½‘•È°Í½ÕÉ”èÉ…µ•MÕÉ™…”¤èÙ½¥ì(€€€½¹ÍÐÉ½ÕÀ€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹ÁÉ•Í•¹ÑA¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èÑ¡¥Ì¹Í…µÁ±•Èô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èÍ½ÕÉ”¹½±½ÕÈ¹É•…Ñ•Y¥•Ü ¤ô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐÁ…ÍÌ€ô•¹½‘•È¹‰•¥¹I•¹‘•ÉA…ÍÌ¡ì½±½ÉÑÑ…¡µ•¹ÑÌèmì(€€€€€Ù¥•ÜèÑ¡¥Ì¹½¹Ñ•áÐ¹•ÑÕÉÉ•¹ÑQ•áÑÕÉ” ¤¹É•…Ñ•Y¥•Ü ¤°(€€€€€±•…ÉY…±Õ”èìÈè€À¸ÀÀà°œè€À¸ÀÄ°ˆè€À¸ÀÄÐ°„è€Äô°(€€€€€±½…‘=Àè€±•…Èœ°ÍÑ½É•=Àè€ÍÑ½É”œ(€€€õtô¤ì(€€€Á…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹ÁÉ•Í•¹ÑA¥Á•±¥¹”¤ì(€€€Á…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°É½ÕÀ¤ì(€€€Á…ÍÌ¹‘É…Ü Ì¤ì(€€€Á…ÍÌ¹•¹ ¤ì(€ô((€ÁÉ¥Ù…Ñ”ÍÕÉ™…•ÑÑ…¡µ•¹ÑÌ¡ÍÕÉ™…”èÉ…µ•MÕÉ™…”¤èAUI•¹‘•ÉA…ÍÍ½±½ÉÑÑ…¡µ•¹Ñmtì(€€€É•ÑÕÉ¸l(€€€€€ì(€€€€€€€Ù¥•ÜèÍÕÉ™…”¹½±½ÕÈ¹É•…Ñ•Y¥•Ü ¤°(€€€€€€€±•…ÉY…±Õ”èìÈè€À¸ÀÀà°œè€À¸ÀÄ°ˆè€À¸ÀÄÐ°„è€Äô°(€€€€€€€±½…‘=Àè€±•…Èœ°ÍÑ½É•=Àè€ÍÑ½É”œ(€€€€€ô°(€€€€€ì(€€€€€€€Ù¥•ÜèÍÕÉ™…”¹ÁÉ½Ù•¹…¹”¹É•…Ñ•Y¥•Ü ¤°(€€€€€€€±•…ÉY…±Õ”èìÈè€À°œè€À°ˆè€À°„è€Àô°(€€€€€€€±½…‘=Àè€±•…Èœ°ÍÑ½É•=Àè€ÍÑ½É”œ(€€€€€ô(€€€tì(€ô((€ÁÉ¥Ù…Ñ”•¹½‘•½¹Ñ¥¹Õ¥ÑåI•‘ÕÑ¥½¸ (€€€•¹½‘•ÈèAU½µµ…¹‘¹½‘•È°(€€€ÁÉ½Ù•¹…¹”èAUQ•áÑÕÉ”°(€€€Ý¥‘Ñ è¹Õµ‰•È°(€€€¡•¥¡Ðè¹Õµ‰•È°(€€€™É…µ•%è¹Õµ‰•È°(€€€Ù¥•ÝI•Ù¥Í¥½¸è¹Õµ‰•È°(€€€É•ÅÕ•ÍÑ%è¹Õµ‰•È°(€€€½µÁ±•Ñ•‘	…Ñ¡I•Ù¥Í¥½¸è¹Õµ‰•È(€€¤èì(€€€Í±½ÐèI•…‘‰…­M±½Ðì(€€€™É…µ•%è¹Õµ‰•Èì(€€€Ù¥•ÝI•Ù¥Í¥½¸è¹Õµ‰•Èì(€€€É•ÅÕ•ÍÑ%è¹Õµ‰•Èì(€€€½µÁ±•Ñ•‘	…Ñ¡I•Ù¥Í¥½¸è¹Õµ‰•Èì(€ôð¹Õ±°ì(€€€±•ÐÍ•±•Ñ•èI•…‘‰…­M±½Ðð¹Õ±°€ô¹Õ±°ì(€€€™½È€¡±•Ð½™™Í•Ð€ô€Àì½™™Í•Ð€ðÑ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•…‘‰…­Ì¹±•¹Ñ ì½™™Í•Ð¬¬¤ì(€€€€€½¹ÍÐ¥¹‘•à€ô€¡Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•…‘‰…­ÕÉÍ½È€¬½™™Í•Ð¤€”Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•…‘‰…­Ì¹±•¹Ñ ì(€€€€€½¹ÍÐ…¹‘¥‘…Ñ”€ôÑ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•…‘‰…­Ím¥¹‘•átì(€€€€€¥˜€¡…¹‘¥‘…Ñ”€˜˜€……¹‘¥‘…Ñ”¹‰ÕÍä¤ì(€€€€€€€Í•±•Ñ•€ô…¹‘¥‘…Ñ”ì(€€€€€€€Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•…‘‰…­ÕÉÍ½È€ô€¡¥¹‘•à€¬€Ä¤€”Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•…‘‰…­Ì¹±•¹Ñ ì(€€€€€€€‰É•…¬ì(€€€€€ô(€€€ô(€€€¥˜€ …Í•±•Ñ•¤ìÑ¡¥Ì¹‘É½ÁÁ•‘I•…‘‰…­Ì¬¬ìÉ•ÑÕÉ¸¹Õ±°ìô(€€€Í•±•Ñ•¹‰ÕÍä€ôÑÉÕ”ì(€€€½¹ÍÐÉ½ÕÀ€ôÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•	¥¹‘É½ÕÀ¡ì(€€€€€±…å½ÕÐèÑ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåA¥Á•±¥¹”¹•Ñ	¥¹‘É½ÕÁ1…å½ÕÐ À¤°(€€€€€•¹ÑÉ¥•Ìèl(€€€€€€€ì‰¥¹‘¥¹œè€À°É•Í½ÕÉ”èÁÉ½Ù•¹…¹”¹É•…Ñ•Y¥•Ü ¤ô°(€€€€€€€ì‰¥¹‘¥¹œè€Ä°É•Í½ÕÉ”èì‰Õ™™•ÈèÑ¡¥Ì¹½¹Ñ¥¹Õ¥Ñå½Õ¹Ñ•É	Õ™™•Èôô(€€€€€t(€€€ô¤ì(€€€½¹ÍÐÁ…ÍÌ€ô•¹½‘•È¹‰•¥¹½µÁÕÑ•A…ÍÌ¡ì±…‰•°è€ÁÉ•Í•¹Ñ…Ñ¥½¸µ½¹Ñ¥¹Õ¥ÑäµÉ•‘ÕÑ¥½¸œô¤ì(€€€Á…ÍÌ¹Í•ÑA¥Á•±¥¹”¡Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåA¥Á•±¥¹”¤ì(€€€Á…ÍÌ¹Í•Ñ	¥¹‘É½ÕÀ À°É½ÕÀ¤ì(€€€Á…ÍÌ¹‘¥ÍÁ…Ñ¡]½É­É½ÕÁÌ¡5…Ñ ¹•¥°¡Ý¥‘Ñ €¼€à¤°5…Ñ ¹•¥°¡¡•¥¡Ð€¼€à¤¤ì(€€€Á…ÍÌ¹•¹ ¤ì(€€€•¹½‘•È¹½Áå	Õ™™•ÉQ½	Õ™™•È¡Ñ¡¥Ì¹½¹Ñ¥¹Õ¥Ñå½Õ¹Ñ•É	Õ™™•È°€À°Í•±•Ñ•¹‰Õ™™•È°€À°=9Q%9U%Qe}=U9QI}	eQL¤ì(€€€É•ÑÕÉ¸ìÍ±½ÐèÍ•±•Ñ•°™É…µ•%°Ù¥•ÝI•Ù¥Í¥½¸°É•ÅÕ•ÍÑ%°½µÁ±•Ñ•‘	…Ñ¡I•Ù¥Í¥½¸ôì(€ô((€ÁÉ¥Ù…Ñ”™¥¹¥Í¡½¹Ñ¥¹Õ¥ÑåI•…‘‰…¬¡É•…‘‰…¬èì(€€€Í±½ÐèI•…‘‰…­M±½Ðì(€€€™É…µ•%è¹Õµ‰•Èì(€€€Ù¥•ÝI•Ù¥Í¥½¸è¹Õµ‰•Èì(€€€É•ÅÕ•ÍÑ%è¹Õµ‰•Èì(€€€½µÁ±•Ñ•‘	…Ñ¡I•Ù¥Í¥½¸è¹Õµ‰•Èì(€ô¤èÙ½¥ì(€€€Ù½¥É•…‘‰…¬¹Í±½Ð¹‰Õ™™•È¹µ…ÁÍå¹Œ¡AU5…Á5½‘”¹I¤¹Ñ¡•¸  ¤€ôøì(€€€€€½¹ÍÐÙ…±Õ•Ì€ô¹•ÜU¥¹ÐÌÉÉÉ…ä¡É•…‘‰…¬¹Í±½Ð¹‰Õ™™•È¹•Ñ5…ÁÁ•‘I…¹” ¤¤¹Í±¥” ¤ì(€€€€€É•…‘‰…¬¹Í±½Ð¹‰Õ™™•È¹Õ¹µ…À ¤ì(€€€€€É•…‘‰…¬¹Í±½Ð¹‰ÕÍä€ô™…±Í”ì(€€€€€¥˜€¡Ñ¡¥Ì¹‘•ÍÑÉ½å•¤É•ÑÕÉ¸ì(€€€€€½¹ÍÐ¡•­Á½¥¹Ñ±¥¥‰±”€ô€¡Ù…±Õ•ÍlÁt€üü€À¤€ø€À(€€€€€€€€˜˜€¡Ù…±Õ•ÍlÅt€üü€À¤€ôôô€À(€€€€€€€€˜˜€¡Ù…±Õ•ÍlÕt€üü€À¤€ôôô€À(€€€€€€€€˜˜€¡Ù…±Õ•Ílát€üü€À¤€ôôô€À(€€€€€€€€˜˜€¡Ù…±Õ•Ílåt€üü€À¤€ôôô€À(€€€€€€€€˜˜€¡Ù…±Õ•ÍlÄÉt€üü€À¤€ôôô€À(€€€€€€€€˜˜€¡Ù…±Õ•ÍlÑt€üü€À¤€ø€À(€€€€€€€€˜˜€¡Ù…±Õ•ÍlÙt€üü€À¤€¬€¡Ù…±Õ•ÍlÝt€üü€À¤€ôôô€¡Ù…±Õ•ÍlÁt€üü€À¤ì(€€€€€½¹ÍÐ™É…µ”è½¹Ñ¥¹Õ¥ÑåÉ…µ”€ôì(€€€€€€€™É…µ•%èÉ•…‘‰…¬¹™É…µ•%°(€€€€€€€Ù¥•ÝI•Ù¥Í¥½¸èÉ•…‘‰…¬¹Ù¥•ÝI•Ù¥Í¥½¸°(€€€€€€€É•ÅÕ•ÍÑ%èÉ•…‘‰…¬¹É•ÅÕ•ÍÑ%°(€€€€€€€½µÁ±•Ñ•‘	…Ñ¡I•Ù¥Í¥½¸èÉ•…‘‰…¬¹½µÁ±•Ñ•‘	…Ñ¡I•Ù¥Í¥½¸°(€€€€€€€Ñ½Ñ…±A¥á•±ÌèÙ…±Õ•ÍlÁt€üü€À°(€€€€€€€¥¹Ù…±¥‘A¥á•±ÌèÙ…±Õ•ÍlÅt€üü€À°(€€€€€€€¡¥ÍÑ½ÉåA¥á•±ÌèÙ…±Õ•ÍlÉt€üü€À°(€€€€€€€É•Ñ…¥¹•‘A¥á•±ÌèÙ…±Õ•ÍlÍt€üü€À°(€€€€€€€ÕÉÉ•¹ÑA¥á•±ÌèÙ…±Õ•ÍlÑt€üü€À°(€€€€€€€ÁÉ½Ù¥Í¥½¹…±…ÁA¥á•±ÌèÙ…±Õ•ÍlÕt€üü€À°(€€€€€€€™¥¹…±…ÁA¥á•±ÌèÙ…±Õ•ÍlÙt€üü€À°(€€€€€€€Ñ•Éµ¥¹…±A¥á•±ÌèÙ…±Õ•ÍlÝt€üü€À°(€€€€€€€ÅÕ…±¥ÑåI•É•ÍÍ¥½¹A¥á•±ÌèÙ…±Õ•Ílát€üü€À°(€€€€€€€•Í…Á•‘Q½AÉ½Ù¥Í¥½¹…±	±…­A¥á•±ÌèÙ…±Õ•Ílåt€üü€À°(€€€€€€€…¹‘¥‘…Ñ•I•©•Ñ•‘1½Ý•ÉEÕ…±¥ÑåA¥á•±ÌèÙ…±Õ•ÍlÄÁt€üü€À°(€€€€€€€Í•µ…¹Ñ¥½¹™±¥ÑÙ•¹ÑÌèÙ…±Õ•ÍlÄÅt€üü€À°(€€€€€€€½¹™±¥ÑA¥á•±ÌèÙ…±Õ•ÍlÄÉt€üü€À°(€€€€€€€¡•­Á½¥¹Ñ±¥¥‰±”°(€€€€€€€‘É½ÁÁ•‘I•…‘‰…­ÌèÑ¡¥Ì¹‘É½ÁÁ•‘I•…‘‰…­Ì(€€€€€ôì(€€€€€¥˜€¡™É…µ”¹™É…µ•%€øôÑ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•‘Õ•‘É…µ”¤ì(€€€€€€€Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåY…±Õ•Ì€ôÙ…±Õ•Ìì(€€€€€€€Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåI•‘Õ•‘É…µ”€ô™É…µ”¹™É…µ•%ì(€€€€€ô(€€€€€Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåÉ…µ•Ì¹ÁÕÍ ¡™É…µ”¤ì(€€€€€Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåÉ…µ•Ì¹Í½ÉÐ ¡±•™Ð°É¥¡Ð¤€ôø±•™Ð¹™É…µ•%€´É¥¡Ð¹™É…µ•%¤ì(€€€€€Ý¡¥±”€¡Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåÉ…µ•Ì¹±•¹Ñ €ø=9Q%9U%Qe}I5}!%MQ=Id¤Ñ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåÉ…µ•Ì¹Í¡¥™Ð ¤ì(€€€€€Ñ¡¥Ì¹É•Í½±Ù•½¹Ñ¥¹Õ¥Ñå]…¥Ñ•ÉÌ ¤ì(€€€€€¥˜€¡Ñ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ðü¹™É…µ•%€ôôô™É…µ”¹™É…µ•%¤ì(€€€€€€€½¹ÍÐÁ•¹‘¥¹œ€ôÑ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ðì(€€€€€€€Ù½¥Á•¹‘¥¹œ¹Ù…±¥‘…Ñ¥½¸¹Ñ¡•¸¡•ÉÉ½È€ôøì(€€€€€€€€€¥˜€¡Ñ¡¥Ì¹‘•ÍÑÉ½å•ñðÑ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ð€„ôôÁ•¹‘¥¹œ¤É•ÑÕÉ¸ì(€€€€€€€€€Ñ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ð€ô¹Õ±°ì(€€€€€€€€€¥˜€¡•ÉÉ½Èñð€…™É…µ”¹¡•­Á½¥¹Ñ±¥¥‰±”¤ì(€€€€€€€€€€€¥˜€¡•ÉÉ½È¤Ñ¡¥Ì¹Ù…±¥‘…Ñ¥½¹ÉÉ½ÉÌ¬¬ì(€€€€€€€€€€€Ñ¡¥Ì¹‘•ÍÑÉ½åMÕÉ™…”¡Á•¹‘¥¹œ¹Í¹…ÁÍ¡½Ð¹ÍÕÉ™…”¤ì(€€€€€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€ô(€€€€€€€€€Ñ¡¥Ì¹ÁÉ½µ½Ñ•¡•­Á½¥¹Ð¡Á•¹‘¥¹œ¹Í¹…ÁÍ¡½Ð¤ì(€€€€€€€ô°€ ¤€ôøì(€€€€€€€€€¥˜€ …Ñ¡¥Ì¹‘•ÍÑÉ½å•¤Ñ¡¥Ì¹Ù…±¥‘…Ñ¥½¹ÉÉ½ÉÌ¬¬ì(€€€€€€€€€¥˜€¡Ñ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ð€ôôôÁ•¹‘¥¹œ¤Ñ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ð€ô¹Õ±°ì(€€€€€€€€€Ñ¡¥Ì¹‘•ÍÑÉ½åMÕÉ™…”¡Á•¹‘¥¹œ¹Í¹…ÁÍ¡½Ð¹ÍÕÉ™…”¤ì(€€€€€€€ô¤ì(€€€€€ô(€€€ô°€ ¤€ôøì(€€€€€É•…‘‰…¬¹Í±½Ð¹‰ÕÍä€ô™…±Í”ì(€€€€€¥˜€ …Ñ¡¥Ì¹‘•ÍÑÉ½å•¤ì(€€€€€€€Ñ¡¥Ì¹‘É½ÁÁ•‘I•…‘‰…­Ì¬¬ì(€€€€€€€¥˜€¡Ñ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ðü¹™É…µ•%€ôôôÉ•…‘‰…¬¹™É…µ•%¤ì(€€€€€€€€€½¹ÍÐ™…¥±•€ôÑ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ðì(€€€€€€€€€Ñ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ð€ô¹Õ±°ì(€€€€€€€€€Ñ¡¥Ì¹‘•ÍÑÉ½åMÕÉ™…”¡™…¥±•¹Í¹…ÁÍ¡½Ð¹ÍÕÉ™…”¤ì(€€€€€€€ô(€€€€€ô(€€€ô¤ì(€ô((€ÁÉ¥Ù…Ñ”É•Í½±Ù•½¹Ñ¥¹Õ¥Ñå]…¥Ñ•ÉÌ ¤èÙ½¥ì(€€€½¹ÍÐÁ•¹‘¥¹œ€ôl¸¸¹Ñ¡¥Ì¹½¹Ñ¥¹Õ¥Ñå]…¥Ñ•ÉÍtì(€€€™½È€¡½¹ÍÐÝ…¥Ñ•È½˜Á•¹‘¥¹œ¤ì(€€€€€½¹ÍÐÉ•…‘ä€ôÑ¡¥Ì¹½¹Ñ¥¹Õ¥ÑåÉ…µ•Ì¹™¥¹¡™É…µ”€ôø™É…µ”¹™É…µ•%€øÝ…¥Ñ•È¹…™Ñ•ÉÉ…µ”¤ì(€€€€€¥˜€ …É•…‘ä¤½¹Ñ¥¹Õ”ì(€€€€€±•…ÉQ¥µ•½ÕÐ¡Ý…¥Ñ•È¹Ñ¥µ•½ÕÐ¤ì(€€€€€Ñ¡¥Ì¹½¹Ñ¥¹Õ¥Ñå]…¥Ñ•ÉÌ€ôÑ¡¥Ì¹½¹Ñ¥¹Õ¥Ñå]…¥Ñ•ÉÌ¹™¥±Ñ•È¡¥Ñ•´€ôø¥Ñ•´€„ôôÝ…¥Ñ•È¤ì(€€€€€Ý…¥Ñ•È¹É•Í½±Ù”¡É•…‘ä¤ì(€€€ô(€ô((€ÁÉ¥Ù…Ñ”ÁÉ½µ½Ñ•¡•­Á½¥¹Ð¡Í¹…ÁÍ¡½ÐèM¹…ÁÍ¡½Ð¤èÙ½¥ì(€€€½¹ÍÐÁÉ•Ù¥½ÕÌ€ôÑ¡¥Ì¹ÍÑ…‰±•¡•­Á½¥¹Ðì(€€€Ñ¡¥Ì¹ÍÑ…‰±•¡•­Á½¥¹Ð€ôÍ¹…ÁÍ¡½Ðì(€€€Ñ¡¥Ì¹…¹¡½ÉAÉ½µ½Ñ¥½¹Ì¬¬ì(€€€¥˜€¡ÁÉ•Ù¥½ÕÌ¤ì(€€€€€½¹ÍÐ±…Ñ•ÍÐ€ôÑ¡¥Ì¹Í¹…ÁÍ¡½ÑÍmÑ¡¥Ì¹Í¹…ÁÍ¡½ÑÌ¹±•¹Ñ €´€Åt€üü¹Õ±°ì(€€€€€½¹ÍÐ™…É¹½Õ €ô€…±…Ñ•ÍÐ(€€€€€€€ñð5…Ñ ¹…‰Ì¡Ù¥•ÝM…±•=Ñ…Ù•Ì¡ÁÉ•Ù¥½ÕÌ¹Ù¥•Ü¤€´Ù¥•ÝM…±•=Ñ…Ù•Ì¡±…Ñ•ÍÐ¹Ù¥•Ü¤¤€øôM9AM!=Q}%9QIY1}=QYLì(€€€€€¥˜€¡™…É¹½Õ ¤Ñ¡¥Ì¹Í¹…ÁÍ¡½ÑÌ¹ÁÕÍ ¡ÁÉ•Ù¥½ÕÌ¤ì(€€€€€•±Í”Ñ¡¥Ì¹‘•ÍÑÉ½åMÕÉ™…”¡ÁÉ•Ù¥½ÕÌ¹ÍÕÉ™…”¤ì(€€€ô(€€€Ñ¡¥Ì¹ÑÉ¥µI•Ñ…¥¹•‘!¥ÍÑ½Éä ¤ì(€ô((€ÁÉ¥Ù…Ñ”ÑÉ¥µI•Ñ…¥¹•‘!¥ÍÑ½Éä ¤èÙ½¥ì(€€€½¹ÍÐÍÑ…‰±•	åÑ•Ì€ôÑ¡¥Ì¹ÍÑ…‰±•¡•­Á½¥¹Ð€üÍÕÉ™…•	åÑ•Ì¡Ñ¡¥Ì¹ÍÑ…‰±•¡•­Á½¥¹Ð¹Ù¥•Ü¤€è€Àì(€€€±•ÐÉ•Ñ…¥¹•‘	åÑ•Ì€ôÍÑ…‰±•	åÑ•Ì€¬Ñ¡¥Ì¹Í¹…ÁÍ¡½ÑÌ¹É•‘Õ” ¡ÍÕ´°¥Ñ•´¤€ôøÍÕ´€¬ÍÕÉ™…•	åÑ•Ì¡¥Ñ•´¹Ù¥•Ü¤°€À¤ì(€€€Ý¡¥±”€¡Ñ¡¥Ì¹Í¹…ÁÍ¡½ÑÌ¹±•¹Ñ €ø5a}M9AM!=QLñðÉ•Ñ…¥¹•‘	åÑ•Ì€øIQ%9}55=Ie}	UQ}	eQL¤ì(€€€€€½¹ÍÐ½±‘•ÍÐ€ôÑ¡¥Ì¹Í¹…ÁÍ¡½ÑÌ¹Í¡¥™Ð ¤ì(€€€€€¥˜€ …½±‘•ÍÐ¤‰É•…¬ì(€€€€€É•Ñ…¥¹•‘	åÑ•Ì€´ôÍÕÉ™…•	åÑ•Ì¡½±‘•ÍÐ¹Ù¥•Ü¤ì(€€€€€Ñ¡¥Ì¹‘•ÍÑÉ½åMÕÉ™…”¡½±‘•ÍÐ¹ÍÕÉ™…”¤ì(€€€ô(€ô((€ÁÉ¥Ù…Ñ”ÁÉ•Í•¹Ñ…Ñ¥½¹	åÑ•Ì (€€€É•Í½ÕÉ•ÌèI•Í½ÕÉ•M•Ð°(€€€…Ñ±…Ìè•ÁÑ•‘Q¥±•Ñ±…Ì°(€€€…‘‘¥Ñ¥½¹…±I•Ñ…¥¹•‘	åÑ•Ìè¹Õµ‰•È(€€¤è¹Õµ‰•Èì(€€€½¹ÍÐ±¥Ù•MÕÉ™…•Í¹‘•ÁÑ €ôÉ•Í½ÕÉ•Ì¹Ý¥‘Ñ €¨É•Í½ÕÉ•Ì¹¡•¥¡Ð€¨€Èàì(€€€½¹ÍÐ…Ñ±…Í	åÑ•Ì€ô…Ñ±…Ì¹Ý¥‘Ñ €¨…Ñ±…Ì¹¡•¥¡Ð€¨€ÄÈì(€€€½¹ÍÐÍÑ…‰±•	åÑ•Ì€ôÑ¡¥Ì¹ÍÑ…‰±•¡•­Á½¥¹Ð€üÍÕÉ™…•	åÑ•Ì¡Ñ¡¥Ì¹ÍÑ…‰±•¡•­Á½¥¹Ð¹Ù¥•Ü¤€è€Àì(€€€½¹ÍÐÍ¹…ÁÍ¡½Ñ	åÑ•Ì€ôÑ¡¥Ì¹Í¹…ÁÍ¡½ÑÌ¹É•‘Õ” ¡ÍÕ´°¥Ñ•´¤€ôøÍÕ´€¬ÍÕÉ™…•	åÑ•Ì¡¥Ñ•´¹Ù¥•Ü¤°€À¤ì(€€€½¹ÍÐÁ•¹‘¥¹	åÑ•Ì€ôÑ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ð€üÍÕÉ™…•	åÑ•Ì¡Ñ¡¥Ì¹Á•¹‘¥¹¡•­Á½¥¹Ð¹Í¹…ÁÍ¡½Ð¹Ù¥•Ü¤€è€Àì(€€€É•ÑÕÉ¸±¥Ù•MÕÉ™…•Í¹‘•ÁÑ €¬…Ñ±…Í	åÑ•Ì€¬ÍÑ…‰±•	åÑ•Ì€¬Í¹…ÁÍ¡½Ñ	åÑ•Ì(€€€€€€¬Á•¹‘¥¹	åÑ•Ì€¬…‘‘¥Ñ¥½¹…±I•Ñ…¥¹•‘	åÑ•Ìì(€ô((€ÁÉ¥Ù…Ñ”•¹ÍÕÉ•M¥é”¡Ý¥‘Ñ è¹Õµ‰•È°¡•¥¡Ðè¹Õµ‰•È¤èÙ½¥ì(€€€¥˜€¡Ñ¡¥Ì¹É•Í½ÕÉ•Ìü¹Ý¥‘Ñ €ôôôÝ¥‘Ñ €˜˜Ñ¡¥Ì¹É•Í½ÕÉ•Ì¹¡•¥¡Ð€ôôô¡•¥¡Ð¤É•ÑÕÉ¸ì(€€€Ñ¡¥Ì¹É•Í½ÕÉ•Á½ ¬¬ì(€€€Ñ¡¥Ì¹É•Í½ÕÉ•Ì€ôÑ¡¥Ì¹É•…Ñ•M•Ð¡Ý¥‘Ñ °¡•¥¡Ð°Ñ¡¥Ì¹É•Í½ÕÉ•Á½ ¤ì(€ô((€ÁÉ¥Ù…Ñ”É•…Ñ•M•Ð¡Ý¥‘Ñ è¹Õµ‰•È°¡•¥¡Ðè¹Õµ‰•È°•Á½ è¹Õµ‰•È¤èI•Í½ÕÉ•M•Ðì(€€€½¹ÍÐÉ•…Ñ•MÕÉ™…”€ô€¡±…‰•°èÍÑÉ¥¹œ¤èÉ…µ•MÕÉ™…”€ôø€¡ì(€€€€€½±½ÕÈèÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•Q•áÑÕÉ”¡ì(€€€€€€€±…‰•°è€‘í±…‰•±ôµ½±½ÕÉ€°Í¥é”èmÝ¥‘Ñ °¡•¥¡Ñt°™½Éµ…Ðè€É‰„áÕ¹½É´œ°(€€€€€€€ÕÍ…”èAUQ•áÑÕÉ•UÍ…”¹I9I}QQ!59PðAUQ•áÑÕÉ•UÍ…”¹QaQUI}	%9%9ðAUQ•áÑÕÉ•UÍ…”¹=Ae}MI(€€€€€ô¤°(€€€€€ÁÉ½Ù•¹…¹”èÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•Q•áÑÕÉ”¡ì(€€€€€€€±…‰•°è€‘í±…‰•±ôµÁÉ½Ù•¹…¹•€°Í¥é”èmÝ¥‘Ñ °¡•¥¡Ñt°™½Éµ…Ðè€ÈÌÉÕ¥¹Ðœ°(€€€€€€€ÕÍ…”èAUQ•áÑÕÉ•UÍ…”¹I9I}QQ!59PðAUQ•áÑÕÉ•UÍ…”¹QaQUI}	%9%9ðAUQ•áÑÕÉ•UÍ…”¹=Ae}MI(€€€€€ô¤(€€€ô¤ì(€€€É•ÑÕÉ¸ì(€€€€€ÍÕÉ™…•ÌèmÉ•…Ñ•MÕÉ™…”¡É½±±¥¹œ´‘í•Á½¡ô´Á€¤°É•…Ñ•MÕÉ™…”¡É½±±¥¹œ´‘í•Á½¡ô´Å€¥t°(€€€€€…¹‘¥‘…Ñ”èÉ•…Ñ•MÕÉ™…”¡…¹‘¥‘…Ñ”´‘í•Á½¡õ€¤°(€€€€€…¹‘¥‘…Ñ••ÁÑ èÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•Q•áÑÕÉ”¡ì(€€€€€€€±…‰•°è…¹‘¥‘…Ñ”µ‘•ÁÑ ´‘í•Á½¡õ€°(€€€€€€€Í¥é”èmÝ¥‘Ñ °¡•¥¡Ñt°(€€€€€€€™½Éµ…Ðè€‘•ÁÑ ÌÉ™±½…Ðœ°(€€€€€€€ÕÍ…”èAUQ•áÑÕÉ•UÍ…”¹I9I}QQ!59P(€€€€€ô¤°(€€€€€Ý¥‘Ñ °¡•¥¡Ð°•Á½ (€€€ôì(€ô((€ÁÉ¥Ù…Ñ”É•…Ñ•I•Ñ…¥¹•‘MÕÉ™…”¡Ý¥‘Ñ è¹Õµ‰•È°¡•¥¡Ðè¹Õµ‰•È°•¹•É…Ñ¥½¸è¹Õµ‰•È¤èÉ…µ•MÕÉ™…”ì(€€€É•ÑÕÉ¸ì(€€€€€½±½ÕÈèÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•Q•áÑÕÉ”¡ì(€€€€€€€±…‰•°èÉ•Í½±Ù•µ¡•­Á½¥¹Ðµ½±½ÕÈ´‘í•¹•É…Ñ¥½¹õ€°Í¥é”èmÝ¥‘Ñ °¡•¥¡Ñt°™½Éµ…Ðè€É‰„áÕ¹½É´œ°(€€€€€€€ÕÍ…”èAUQ•áÑÕÉ•UÍ…”¹QaQUI}	%9%9ðAUQ•áÑÕÉ•UÍ…”¹=Ae}MP(€€€€€ô¤°(€€€€€ÁÉ½Ù•¹…¹”èÑ¡¥Ì¹‘•Ù¥”¹É•…Ñ•Q•áÑÕÉ”¡ì(€€€€€€€±…‰•°èÉ•Í½±Ù•µ¡•­Á½¥¹ÐµÁÉ½Ù•¹…¹”´‘í•¹•É…Ñ¥½¹õ€°Í¥é”èmÝ¥‘Ñ °¡•¥¡Ñt°™½Éµ…Ðè€ÈÌÉÕ¥¹Ðœ°(€€€€€€€ÕÍ…”èAUQ•áÑÕÉ•UÍ…”¹QaQUI}	%9%9ðAUQ•áÑÕÉ•UÍ…”¹=Ae}MP(€€€€€ô¤(€€€ôì(€ô((€ÁÉ¥Ù…Ñ”É•Ñ¥É•M•Ñ™Ñ•ÉMÕ‰µ¥ÑÑ•‘]½É¬¡É•Í½ÕÉ•ÌèI•Í½ÕÉ•M•Ð¤èÙ½¥ì(€€€¥˜€¡Ñ¡¥Ì¹É•Ñ¥É¥¹M•ÑÌ¹¡…Ì¡É•Í½ÕÉ•Ì¤¤É•ÑÕÉ¸ì(€€€Ñ¡¥Ì¹É•Ñ¥É¥¹M•ÑÌ¹…‘¡É•Í½ÕÉ•Ì¤ì(€€€Ù½¥Ñ¡¥Ì¹‘•Ù¥”¹ÅÕ•Õ”¹½¹MÕ‰µ¥ÑÑ•‘]½É­½¹” ¤¹Ñ¡•¸  ¤€ôøì(€€€€€¥˜€¡Ñ¡¥Ì¹É•Ñ¥É¥¹M•ÑÌ¹‘•±•Ñ”¡É•Í½ÕÉ•Ì¤¤Ñ¡¥Ì¹‘•ÍÑÉ½åM•Ð¡É•Í½ÕÉ•Ì¤ì(€€€ô°€ ¤€ôøì(€€€€€¥˜€¡Ñ¡¥Ì¹É•Ñ¥É¥¹M•ÑÌ¹‘•±•Ñ”¡É•Í½ÕÉ•Ì¤¤Ñ¡¥Ì¹‘•ÍÑÉ½åM•Ð¡É•Í½ÕÉ•Ì¤ì(€€€ô¤ì(€ô((€ÁÉ¥Ù…Ñ”‘•ÍÑÉ½åMÕÉ™…”¡ÍÕÉ™…”èÉ…µ•MÕÉ™…”¤èÙ½¥ì(€€€ÍÕÉ™…”¹½±½ÕÈ¹‘•ÍÑÉ½ä ¤ì(€€€ÍÕÉ™…”¹ÁÉ½Ù•¹…¹”¹‘•ÍÑÉ½ä ¤ì(€ô((€ÁÉ¥Ù…Ñ”‘•ÍÑÉ½åM•Ð¡É•Í½ÕÉ•ÌèI•Í½ÕÉ•M•Ð¤èÙ½¥ì(€€€™½È€¡½¹ÍÐÍÕÉ™…”½˜É•Í½ÕÉ•Ì¹ÍÕÉ™…•Ì¤Ñ¡¥Ì¹‘•ÍÑÉ½åMÕÉ™…”¡ÍÕÉ™…”¤ì(€€€Ñ¡¥Ì¹‘•ÍÑÉ½åMÕÉ™…”¡É•Í½ÕÉ•Ì¹…¹‘¥‘…Ñ”¤ì(€€€É•Í½ÕÉ•Ì¹…¹‘¥‘…Ñ••ÁÑ ¹‘•ÍÑÉ½ä ¤ì(€ô((€ÁÉ¥Ù…Ñ”ÍÑ…Ñ¥Œ…Íå¹Œ…ÍÍ•ÉÑM¡…‘•ÉY…±¥¡µ½‘Õ±”èAUM¡…‘•É5½‘Õ±”°±…‰•°èÍÑÉ¥¹œ¤èAÉ½µ¥Í”ñÙ½¥øì(€€€½¹ÍÐ¥¹™¼€ô…Ý…¥Ðµ½‘Õ±”¹•Ñ½µÁ¥±…Ñ¥½¹%¹™¼ ¤ì(€€€½¹ÍÐ•ÉÉ½ÉÌ€ô¥¹™¼¹µ•ÍÍ…•Ì¹™¥±Ñ•È ¡µ•ÍÍ…”èìÑåÁ”èÍÑÉ¥¹œô¤€ôøµ•ÍÍ…”¹ÑåÁ”€ôôô€•ÉÉ½Èœ¤ì(€€€¥˜€¡•ÉÉ½ÉÌ¹±•¹Ñ €ôôô€À¤É•ÑÕÉ¸ì(€€€½¹ÍÐ™¥ÉÍÐ€ô•ÉÉ½ÉÍlÁtì(€€€Ñ¡É½Ü¹•ÜÉÉ½È¡€‘í±…‰•±ô]M0±¥¹”€‘í™¥ÉÍÐ¹±¥¹•9Õµôè€‘í™¥ÉÍÐ¹µ•ÍÍ…•õ€¤ì(€ô)ô(