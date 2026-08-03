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
      reducedFrame: this.continuityReducedFrame,
      totalPixels: this.continuityValues[0] ?? 0,
      invalidPixels: this.continuityValues[1] ?? 0,
      historyPixels: this.continuityValues[2] ?? 0,
      retainedPixels: this.continuityValues[3] ?? 0,
      currentPixels: this.continuityValues[4] ?? 0,
      provisionalCapPixels: this.continuityValues[5] ?? 0,
      finalCapPixels: this.continuityValues[6] ?? 0,
      terminalPixels: this.continuityValues[7] ?? 0,
      qualityRegressionPixels: this.continuityValues[8] ?? 0,
      escapedToProvisionalBlackPixels: this.continuityValues[9] ?? 0,
      candidateRejectedLowerQualityPixels: this.continuityValues[10] ?? 0,
      semanticConflictEvents: this.continuityValues[11] ?? 0,
      conflictPixels: this.continuityValues[12] ?? 0,
      droppedReadbacks: this.droppedReadbacks
    };
  }

  get hasPendingPromotion(): boolean { return this.pendingCheckpoint !== null; }

  nextContinuityFrame(afterFrame: number): Promise<ContinuityFrame> {
    if (!this.continuityTest) return Promise.reject(new Error('Continuity frames require ?continuityTest=1'));
    const ready = this.continuityFrames.find(frame => frame.frameId > afterFrame);
    if (ready) return Promise.resolve(ready);
    return new Promise((resolve, reject) => {
      const waiter = {
        afterFrame,
        resolve,
        reject,
        timeout: setTimeout(() => {
          this.continuityWaiters = this.continuityWaiters.filter(item => item !== waiter);
          reject(new Error(`Timed out waiting for a continuity frame after ${afterFrame}`));
        }, CONTINUITY_TIMEOUT_MS)
      };
      this.continuityWaiters.push(waiter);
    });
  }

  present(
    camera: CameraSnapshot,
    aspect: number,
    width: number,
    height: number,
    atlas: AcceptedTileAtlas,
    instances: readonly AtlasInstance[],
    authoritative: boolean,
    targetIterations: number,
    palettePhase: number,
    requestId: number,
    completedBatchRevision: number
  ): boolean {
    const started = performance.now();
    if (this.destroyed || width <= 0 || height <= 0) return false;
    this.ensureSize(width, height);
    const resources = this.resources;
    if (!resources) return false;
    const view: ExactView = {
      camera,
      aspect,
      width,
      height,
      targetIterations,
      palettePhase,
      contentRevision: completedBatchRevision
    };
    const rollingBefore = this.rollingHead;
    // A resolved field is deliberately recomposed without history. This is the
    // only transition allowed to clear a conflict carried by rolling evidence;
    // GPU coverage/provenance reduction must validate it before retention.
    const histories = authoritative ? [] : this.historySources(view);
    this.worstReprojectionErrorTexels = histories.reduce((worst, source) => Math.max(worst, source.error), 0);
    if (histories.length > 0) this.historyFrames++; else this.fallbackFrames++;

    const packed = this.packInstances(instances, width, height);
    this.instanceCount = packed.count;
    if (packed.count > 0) this.device.queue.writeBuffer(this.instanceBuffer, 0, packed.data);

    const hasReadbackCapacity = this.continuityReadbacks.some(slot => !slot.busy);
    const checkpointFitsBudget = this.presentationBytes(resources, atlas, surfaceBytes(view))
      <= PRESENTATION_MEMORY_BUDGET_BYTES;
    const wantsCheckpoint = authoritative
      && !this.pendingCheckpoint
      && !sameCheckpointIdentity(this.stableCheckpoint?.view ?? null, view)
      && hasReadbackCapacity
      && checkpointFitsBudget;
    const wantsReadback = this.continuityTest || wantsCheckpoint;
    const validationScoped = this.continuityTest || wantsCheckpoint;
    if (validationScoped) this.device.pushErrorScope('validation');
    const encoder = this.device.createCommandEncoder({ label: 'explicit-provenance-presentation-frame' });
    const frameId = this.frames + 1;
    if (wantsReadback) encoder.clearBuffer(this.continuityCounterBuffer);

    let head = this.initialHead(resources, rollingBefore);
    this.clearSurface(encoder, head);
    let historyIndex = 0;
    for (const item of histories.slice(0, MAX_HISTORY_SOURCES)) {
      this.renderHistoryCandidate(encoder, resources.candidate, item, view, historyIndex++);
      const output = this.otherSurface(resources, head);
      this.mergeSurfaces(encoder, head, resources.candidate, output);
      head = output;
    }

    const overlayGroup = packed.count > 0 ? this.device.createBindGroup({
      layout: this.overlayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.instanceBuffer } },
        { binding: 1, resource: atlas.colour.createView() },
        { binding: 2, resource: atlas.quality.createView() },
        { binding: 3, resource: atlas.evidence.createView() },
        { binding: 4, resource: { buffer: atlas.leaseDirectory } }
      ]
    }) : null;
    if (overlayGroup && packed.count > 0) {
      this.renderAtlasCandidate(encoder, resources, overlayGroup, packed.count);
      const output = this.otherSurface(resources, head);
      this.mergeSurfaces(encoder, head, resources.candidate, output);
      head = output;
    }

    let checkpointCandidate: Snapshot | null = null;
    if (wantsCheckpoint) {
      checkpointCandidate = {
        surface: this.createRetainedSurface(width, height, this.snapshotGeneration + 1),
        view,
        generation: ++this.snapshotGeneration
      };
      encoder.copyTextureToTexture({ texture: head.colour }, { texture: checkpointCandidate.surface.colour }, { width, height });
      encoder.copyTextureToTexture({ texture: head.provenance }, { texture: checkpointCandidate.surface.provenance }, { width, height });
    }

    const readback = wantsReadback
      ? this.encodeContinuityReduction(
        encoder,
        head.provenance,
        width,
        height,
        frameId,
        camera.generation,
        requestId,
        completedBatchRevision
      )
      : null;
    this.renderCanvas(encoder, head);
    this.device.queue.submit([encoder.finish()]);
    const validation = validationScoped ? this.device.popErrorScope() : Promise.resolve(null);
    this.frames = frameId;
    this.rollingHead = { surface: head, owner: resources, view };
    if (checkpointCandidate && readback) {
      this.pendingCheckpoint = { snapshot: checkpointCandidate, frameId, validation };
    } else {
      if (checkpointCandidate) this.destroySurface(checkpointCandidate.surface);
      void validation.then(error => { if (error && !this.destroyed) this.validationErrors++; });
    }
    if (readback) this.finishContinuityReadback(readback);

    const previousOwner = rollingBefore?.owner ?? null;
    if (previousOwner && previousOwner !== resources) this.retireSetAfterSubmittedWork(previousOwner);
    this.lastFrameCpuMs = performance.now() - started;
    return true;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    const owners = new Set<ResourceSet>(this.retiringSets);
    if (this.resources) owners.add(this.resources);
    if (this.rollingHead) owners.add(this.rollingHead.owner);
    for (const owner of owners) this.destroySet(owner);
    if (this.stableCheckpoint) this.destroySurface(this.stableCheckpoint.surface);
    for (const snapshot of this.snapshots) this.destroySurface(snapshot.surface);
    if (this.pendingCheckpoint) this.destroySurface(this.pendingCheckpoint.snapshot.surface);
    this.resources = null;
    this.rollingHead = null;
    this.stableCheckpoint = null;
    this.snapshots = [];
    this.pendingCheckpoint = null;
    for (const waiter of this.continuityWaiters) {
      clearTimeout(waiter.timeout);
      waiter.reject(new Error('Presenter destroyed while waiting for continuity frame'));
    }
    this.continuityWaiters = [];
    this.reprojectUniform.destroy();
    this.instanceBuffer.destroy();
    this.continuityCounterBuffer.destroy();
    for (const slot of this.continuityReadbacks) slot.buffer.destroy();
  }

  private historySources(target: ExactView): Array<HistorySource & { transform: PackedTransform; error: number }> {
    // A composed display head is deliberately not a history source. Every
    // frame reprojects immutable resolved checkpoints directly into the target
    // view, making the result independent of the number of intermediate rAFs.
    const retained: HistorySource[] = [];
    if (this.stableCheckpoint && this.stableCheckpoint.view.palettePhase === target.palettePhase) retained.push({
      surface: this.stableCheckpoint.surface,
      view: this.stableCheckpoint.view,
      owner: null,
      origin: 2
    });
    for (const snapshot of this.snapshots) {
      if (snapshot.view.palettePhase !== target.palettePhase) continue;
      retained.push({ surface: snapshot.surface, view: snapshot.view, owner: null, origin: 2 });
    }
    const admit = (candidate: HistorySource) => {
      const admission = admitHistory(historyTransform(candidate.view, target), candidate.view, target);
      return admission.accepted ? {
        ...candidate,
        transform: admission.packed,
        error: admission.error,
        projectedRankDelta: footprintRankDelta(admission.packed, candidate.view, target),
        coverage: transformCoverage(admission.packed)
      } : null;
    };
    const ranked = retained.map(admit).filter((item): item is NonNullable<typeof item> => Boolean(item))
      .sort((left, right) => right.projectedRankDelta - left.projectedRankDelta
        || right.coverage - left.coverage);
    const selected: typeof ranked = [];
    let selectedCoverage = 0;
    for (const item of ranked) {
      if (selected.length > 0 && item.coverage <= selectedCoverage + 1e-6) continue;
      selected.push(item);
      selectedCoverage = Math.max(selectedCoverage, item.coverage);
      if (selected.length >= MAX_HISTORY_SOURCES) break;
    }
    return selected;
  }

  private packInstances(instances: readonly AtlasInstance[], width: number, height: number): {
    data: ArrayBuffer;
    count: number;
  } {
    const accepted = instances.flatMap(instance => {
      const { transform } = instance;
      if (!transformIsFinite(transform) || transform.scaleX <= 0 || transform.scaleY <= 0) return [];
      const left = 0.5 + (-0.5 - transform.offsetX) / transform.scaleX;
      const top = 0.5 + (-0.5 - transform.offsetY) / transform.scaleY;
      const right = 0.5 + (0.5 - transform.offsetX) / transform.scaleX;
      const bottom = 0.5 + (0.5 - transform.offsetY) / transform.scaleY;
      if (right <= 0 || bottom <= 0 || left >= 1 || top >= 1) return [];
      return [{
        instance,
        rect: [left, top, right, bottom] as const,
        rank: footprintRank(right - left, bottom - top, width, height)
      }];
    }).sort((left, right) => left.rank - right.rank).slice(0, MAX_INSTANCES);
    const data = new ArrayBuffer(accepted.length * INSTANCE_WORDS * Uint32Array.BYTES_PER_ELEMENT);
    const floats = new Float32Array(data);
    const words = new Uint32Array(data);
    accepted.forEach((item, index) => {
      const base = index * INSTANCE_WORDS;
      floats.set(item.rect, base);
      words.set([
        item.instance.slot.x,
        item.instance.slot.y,
        item.instance.slot.index,
        item.instance.slot.lease,
        Math.max(0, Math.floor(item.instance.iterationFrontier)),
        item.instance.capMode,
        item.rank,
        Math.max(0, Math.floor(item.instance.targetIterations))
      ], base + 4);
    });
    return { data, count: accepted.length };
  }

  private initialHead(resources: ResourceSet, rolling: RollingHead | null): FrameSurface {
    if (rolling?.owner === resources && rolling.surface === resources.surfaces[0]) return resources.surfaces[1];
    return resources.surfaces[0];
  }

  private otherSurface(resources: ResourceSet, current: FrameSurface): FrameSurface {
    return current === resources.surfaces[0] ? resources.surfaces[1] : resources.surfaces[0];
  }

  private clearSurface(encoder: GPUCommandEncoder, surface: FrameSurface): void {
    const pass = encoder.beginRenderPass({ colorAttachments: this.surfaceAttachments(surface) });
    pass.end();
  }

  private renderHistoryCandidate(
    encoder: GPUCommandEncoder,
    destination: FrameSurface,
    source: HistorySource & { transform: PackedTransform },
    target: ExactView,
    uniformIndex: number
  ): void {
    const offset = uniformIndex * REPROJECT_UNIFORM_STRIDE;
    this.device.queue.writeBuffer(this.reprojectUniform, offset, new Float32Array([
      source.transform.scaleX, source.transform.scaleY,
      source.transform.offsetX, source.transform.offsetY
    ]));
    this.device.queue.writeBuffer(this.reprojectUniform, offset + 16, new Uint32Array([
      1,
      footprintRankDelta(source.transform, source.view, target) >>> 0,
      source.origin,
      sameResolvedView(source.view, target) ? 1 : 0
    ]));
    const group = this.device.createBindGroup({
      layout: this.reprojectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.reprojectUniform, offset, size: REPROJECT_UNIFORM_BYTES } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: source.surface.colour.createView() },
        { binding: 3, resource: source.surface.provenance.createView() }
      ]
    });
    const pass = encoder.beginRenderPass({ colorAttachments: this.surfaceAttachments(destination) });
    pass.setPipeline(this.reprojectPipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }

  private renderAtlasCandidate(
    encoder: GPUCommandEncoder,
    resources: ResourceSet,
    group: GPUBindGroup,
    instanceCount: number
  ): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: this.surfaceAttachments(resources.candidate),
      depthStencilAttachment: {
        view: resources.candidateDepth.createView(),
        depthClearValue: 0,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard'
      }
    });
    pass.setPipeline(this.overlayPipeline);
    pass.setBindGroup(0, group);
    pass.draw(6, instanceCount);
    pass.end();
  }

  private mergeSurfaces(
    encoder: GPUCommandEncoder,
    base: FrameSurface,
    candidate: FrameSurface,
    output: FrameSurface
  ): void {
    const group = this.device.createBindGroup({
      layout: this.mergePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: base.colour.createView() },
        { binding: 1, resource: base.provenance.createView() },
        { binding: 2, resource: candidate.colour.createView() },
        { binding: 3, resource: candidate.provenance.createView() },
        { binding: 4, resource: { buffer: this.continuityCounterBuffer } }
      ]
    });
    const pass = encoder.beginRenderPass({ colorAttachments: this.surfaceAttachments(output) });
    pass.setPipeline(this.mergePipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }

  private renderCanvas(encoder: GPUCommandEncoder, source: FrameSurface): void {
    const group = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: source.colour.createView() }
      ]
    });
    const pass = encoder.beginRenderPass({ colorAttachments: [{
      view: this.context.getCurrentTexture().createView(),
      clearValue: { r: 0.008, g: 0.01, b: 0.014, a: 1 },
      loadOp: 'clear', storeOp: 'store'
    }] });
    pass.setPipeline(this.presentPipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }

  private surfaceAttachments(surface: FrameSurface): GPURenderPassColorAttachment[] {
    return [
      {
        view: surface.colour.createView(),
        clearValue: { r: 0.008, g: 0.01, b: 0.014, a: 1 },
        loadOp: 'clear', storeOp: 'store'
      },
      {
        view: surface.provenance.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear', storeOp: 'store'
      }
    ];
  }

  private encodeContinuityReduction(
    encoder: GPUCommandEncoder,
    provenance: GPUTexture,
    width: number,
    height: number,
    frameId: number,
    viewRevision: number,
    requestId: number,
    completedBatchRevision: number
  ): {
    slot: ReadbackSlot;
    frameId: number;
    viewRevision: number;
    requestId: number;
    completedBatchRevision: number;
  } | null {
    let selected: ReadbackSlot | null = null;
    for (let offset = 0; offset < this.continuityReadbacks.length; offset++) {
      const index = (this.continuityReadbackCursor + offset) % this.continuityReadbacks.length;
      const candidate = this.continuityReadbacks[index];
      if (candidate && !candidate.busy) {
        selected = candidate;
        this.continuityReadbackCursor = (index + 1) % this.continuityReadbacks.length;
        break;
      }
    }
    if (!selected) { this.droppedReadbacks++; return null; }
    selected.busy = true;
    const group = this.device.createBindGroup({
      layout: this.continuityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: provenance.createView() },
        { binding: 1, resource: { buffer: this.continuityCounterBuffer } }
      ]
    });
    const pass = encoder.beginComputePass({ label: 'presentation-continuity-reduction' });
    pass.setPipeline(this.continuityPipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    encoder.copyBufferToBuffer(this.continuityCounterBuffer, 0, selected.buffer, 0, CONTINUITY_COUNTER_BYTES);
    return { slot: selected, frameId, viewRevision, requestId, completedBatchRevision };
  }

  private finishContinuityReadback(readback: {
    slot: ReadbackSlot;
    frameId: number;
    viewRevision: number;
    requestId: number;
    completedBatchRevision: number;
  }): void {
    void readback.slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const values = new Uint32Array(readback.slot.buffer.getMappedRange()).slice();
      readback.slot.buffer.unmap();
      readback.slot.busy = false;
      if (this.destroyed) return;
      const checkpointEligible = (values[0] ?? 0) > 0
        && (values[1] ?? 0) === 0
        && (values[5] ?? 0) === 0
        && (values[8] ?? 0) === 0
        && (values[9] ?? 0) === 0
        && (values[12] ?? 0) === 0
        && (values[4] ?? 0) > 0
        && (values[6] ?? 0) + (values[7] ?? 0) === (values[0] ?? 0);
      const frame: ContinuityFrame = {
        frameId: readback.frameId,
        viewRevision: readback.viewRevision,
        requestId: readback.requestId,
        completedBatchRevision: readback.completedBatchRevision,
        totalPixels: values[0] ?? 0,
        invalidPixels: values[1] ?? 0,
        historyPixels: values[2] ?? 0,
        retainedPixels: values[3] ?? 0,
        currentPixels: values[4] ?? 0,
        provisionalCapPixels: values[5] ?? 0,
        finalCapPixels: values[6] ?? 0,
        terminalPixels: values[7] ?? 0,
        qualityRegressionPixels: values[8] ?? 0,
        escapedToProvisionalBlackPixels: values[9] ?? 0,
        candidateRejectedLowerQualityPixels: values[10] ?? 0,
        semanticConflictEvents: values[11] ?? 0,
        conflictPixels: values[12] ?? 0,
        checkpointEligible,
        droppedReadbacks: this.droppedReadbacks
      };
      if (frame.frameId >= this.continuityReducedFrame) {
        this.continuityValues = values;
        this.continuityReducedFrame = frame.frameId;
      }
      this.continuityFrames.push(frame);
      this.continuityFrames.sort((left, right) => left.frameId - right.frameId);
      while (this.continuityFrames.length > CONTINUITY_FRAME_HISTORY) this.continuityFrames.shift();
      this.resolveContinuityWaiters();
      if (this.pendingCheckpoint?.frameId === frame.frameId) {
        const pending = this.pendingCheckpoint;
        void pending.validation.then(error => {
          if (this.destroyed || this.pendingCheckpoint !== pending) return;
          this.pendingCheckpoint = null;
          if (error || !frame.checkpointEligible) {
            if (error) this.validationErrors++;
            this.destroySurface(pending.snapshot.surface);
            return;
          }
          this.promoteCheckpoint(pending.snapshot);
        }, () => {
          if (!this.destroyed) this.validationErrors++;
          if (this.pendingCheckpoint === pending) this.pendingCheckpoint = null;
          this.destroySurface(pending.snapshot.surface);
        });
      }
    }, () => {
      readback.slot.busy = false;
      if (!this.destroyed) {
        this.droppedReadbacks++;
        if (this.pendingCheckpoint?.frameId === readback.frameId) {
          const failed = this.pendingCheckpoint;
          this.pendingCheckpoint = null;
          this.destroySurface(failed.snapshot.surface);
        }
      }
    });
  }

  private resolveContinuityWaiters(): void {
    const pending = [...this.continuityWaiters];
    for (const waiter of pending) {
      const ready = this.continuityFrames.find(frame => frame.frameId > waiter.afterFrame);
      if (!ready) continue;
      clearTimeout(waiter.timeout);
      this.continuityWaiters = this.continuityWaiters.filter(item => item !== waiter);
      waiter.resolve(ready);
    }
  }

  private promoteCheckpoint(snapshot: Snapshot): void {
    const previous = this.stableCheckpoint;
    this.stableCheckpoint = snapshot;
    this.anchorPromotions++;
    if (previous) {
      const latest = this.snapshots[this.snapshots.length - 1] ?? null;
      const farEnough = !latest
        || Math.abs(viewScaleOctaves(previous.view) - viewScaleOctaves(latest.view)) >= SNAPSHOT_INTERVAL_OCTAVES;
      if (farEnough) this.snapshots.push(previous);
      else this.destroySurface(previous.surface);
    }
    this.trimRetainedHistory();
  }

  private trimRetainedHistory(): void {
    const stableBytes = this.stableCheckpoint ? surfaceBytes(this.stableCheckpoint.view) : 0;
    let retainedBytes = stableBytes + this.snapshots.reduce((sum, item) => sum + surfaceBytes(item.view), 0);
    while (this.snapshots.length > MAX_SNAPSHOTS || retainedBytes > RETAINED_MEMORY_BUDGET_BYTES) {
      const oldest = this.snapshots.shift();
      if (!oldest) break;
      retainedBytes -= surfaceBytes(oldest.view);
      this.destroySurface(oldest.surface);
    }
  }

  private presentationBytes(
    resources: ResourceSet,
    atlas: AcceptedTileAtlas,
    additionalRetainedBytes: number
  ): number {
    const liveSurfacesAndDepth = resources.width * resources.height * 28;
    const atlasBytes = atlas.width * atlas.height * 12;
    const stableBytes = this.stableCheckpoint ? surfaceBytes(this.stableCheckpoint.view) : 0;
    const snapshotBytes = this.snapshots.reduce((sum, item) => sum + surfaceBytes(item.view), 0);
    const pendingBytes = this.pendingCheckpoint ? surfaceBytes(this.pendingCheckpoint.snapshot.view) : 0;
    return liveSurfacesAndDepth + atlasBytes + stableBytes + snapshotBytes
      + pendingBytes + additionalRetainedBytes;
  }

  private ensureSize(width: number, height: number): void {
    if (this.resources?.width === width && this.resources.height === height) return;
    this.resourceEpoch++;
    this.resources = this.createSet(width, height, this.resourceEpoch);
  }

  private createSet(width: number, height: number, epoch: number): ResourceSet {
    const createSurface = (label: string): FrameSurface => ({
      colour: this.device.createTexture({
        label: `${label}-colour`, size: [width, height], format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
      }),
      provenance: this.device.createTexture({
        label: `${label}-provenance`, size: [width, height], format: 'r32uint',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
      })
    });
    return {
      surfaces: [createSurface(`rolling-${epoch}-0`), createSurface(`rolling-${epoch}-1`)],
      candidate: createSurface(`candidate-${epoch}`),
      candidateDepth: this.device.createTexture({
        label: `candidate-depth-${epoch}`,
        size: [width, height],
        format: 'depth32float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
      }),
      width, height, epoch
    };
  }

  private createRetainedSurface(width: number, height: number, generation: number): FrameSurface {
    return {
      colour: this.device.createTexture({
        label: `resolved-checkpoint-colour-${generation}`, size: [width, height], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      }),
      provenance: this.device.createTexture({
        label: `resolved-checkpoint-provenance-${generation}`, size: [width, height], format: 'r32uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      })
    };
  }

  private retireSetAfterSubmittedWork(resources: ResourceSet): void {
    if (this.retiringSets.has(resources)) return;
    this.retiringSets.add(resources);
    void this.device.queue.onSubmittedWorkDone().then(() => {
      if (this.retiringSets.delete(resources)) this.destroySet(resources);
    }, () => {
      if (this.retiringSets.delete(resources)) this.destroySet(resources);
    });
  }

  private destroySurface(surface: FrameSurface): void {
    surface.colour.destroy();
    surface.provenance.destroy();
  }

  private destroySet(resources: ResourceSet): void {
    for (const surface of resources.surfaces) this.destroySurface(surface);
    this.destroySurface(resources.candidate);
    resources.candidateDepth.destroy();
  }

  private static async assertShaderValid(module: GPUShaderModule, label: string): Promise<void> {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message: { type: string }) => message.type === 'error');
    if (errors.length === 0) return;
    const first = errors[0];
    throw new Error(`${label} WGSL line ${first.lineNum}: ${first.message}`);
  }
}
