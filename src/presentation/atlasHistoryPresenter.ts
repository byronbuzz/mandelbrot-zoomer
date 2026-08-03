import type { CameraSnapshot } from '../camera/types';
import type { AtlasSlot, AcceptedTileAtlas } from './acceptedTileAtlas';
import { atlasOverlayShader, atlasPresentShader, atlasReprojectShader } from './atlasPresentationShaders';
import {
  fixedDifferenceOverScale,
  packTransform,
  scaleRatio,
  transformIsFinite,
  type PackedTransform
} from './presentationMath';

const REPROJECT_UNIFORM_BYTES = 32;
const MAX_INSTANCES = 512;
const INSTANCE_FLOATS = 8;
const SOURCE_TEXEL_ERROR_LIMIT = 0.01;

type ExactView = Readonly<{
  camera: CameraSnapshot;
  aspect: number;
  width: number;
  height: number;
}>;

type ResourceSet = {
  anchor: GPUTexture;
  candidate: GPUTexture;
  width: number;
  height: number;
  epoch: number;
};

type Anchor = Readonly<{ texture: GPUTexture; owner: ResourceSet; view: ExactView }>;

export type AtlasInstance = Readonly<{ transform: PackedTransform; slot: AtlasSlot }>;

export type AtlasPresenterDiagnostics = Readonly<{
  frames: number;
  historyFrames: number;
  fallbackFrames: number;
  anchorPromotions: number;
  instanceCount: number;
  resourceEpoch: number;
  worstReprojectionErrorTexels: number;
  lastFrameCpuMs: number;
}>;

function f32SourceCoordinate(scale: number, offset: number, uv: number): number {
  return Math.fround(Math.fround(0.5 + offset) + Math.fround(Math.fround(uv - 0.5) * scale));
}

function historyTransform(source: ExactView, target: ExactView): PackedTransform {
  const scaleY = scaleRatio(target.camera.scale, source.camera.scale);
  return {
    scaleX: scaleY * target.aspect / source.aspect,
    scaleY,
    offsetX: fixedDifferenceOverScale(
      target.camera.centerX,
      source.camera.centerX,
      source.camera.scale
    ) / source.aspect,
    offsetY: fixedDifferenceOverScale(
      target.camera.centerY,
      source.camera.centerY,
      source.camera.scale
    )
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

export class AtlasHistoryPresenter {
  private readonly reprojectPipeline: GPURenderPipeline;
  private readonly overlayPipeline: GPURenderPipeline;
  private readonly presentPipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly reprojectUniform: GPUBuffer;
  private readonly instanceBuffer: GPUBuffer;
  private resources: ResourceSet | null = null;
  private anchor: Anchor | null = null;
  private resourceEpoch = 0;
  private frames = 0;
  private historyFrames = 0;
  private fallbackFrames = 0;
  private anchorPromotions = 0;
  private instanceCount = 0;
  private worstReprojectionErrorTexels = 0;
  private lastFrameCpuMs = 0;

  private constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly format: GPUTextureFormat,
    reprojectPipeline: GPURenderPipeline,
    overlayPipeline: GPURenderPipeline,
    presentPipeline: GPURenderPipeline
  ) {
    this.reprojectPipeline = reprojectPipeline;
    this.overlayPipeline = overlayPipeline;
    this.presentPipeline = presentPipeline;
    this.sampler = device.createSampler({
      minFilter: 'linear', magFilter: 'linear',
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge'
    });
    this.reprojectUniform = device.createBuffer({
      label: 'atlas-history-reprojection-uniform',
      size: REPROJECT_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.instanceBuffer = device.createBuffer({
      label: 'atlas-presentation-instances',
      size: MAX_INSTANCES * INSTANCE_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
  }

  static async create(
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat
  ): Promise<AtlasHistoryPresenter> {
    const reprojectModule = device.createShaderModule({ code: atlasReprojectShader });
    const overlayModule = device.createShaderModule({ code: atlasOverlayShader });
    const presentModule = device.createShaderModule({ code: atlasPresentShader });
    await Promise.all([
      this.assertShaderValid(reprojectModule, 'production history reprojection'),
      this.assertShaderValid(overlayModule, 'production atlas overlay'),
      this.assertShaderValid(presentModule, 'production canvas presentation')
    ]);
    const [reprojectPipeline, overlayPipeline, presentPipeline] = await Promise.all([
      device.createRenderPipelineAsync({
        layout: 'auto', vertex: { module: reprojectModule, entryPoint: 'vertexMain' },
        fragment: { module: reprojectModule, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' }
      }),
      device.createRenderPipelineAsync({
        layout: 'auto', vertex: { module: overlayModule, entryPoint: 'vertexMain' },
        fragment: { module: overlayModule, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' }
      }),
      device.createRenderPipelineAsync({
        layout: 'auto', vertex: { module: presentModule, entryPoint: 'vertexMain' },
        fragment: { module: presentModule, entryPoint: 'fragmentMain', targets: [{ format }] },
        primitive: { topology: 'triangle-list' }
      })
    ]);
    return new AtlasHistoryPresenter(
      device, context, format, reprojectPipeline, overlayPipeline, presentPipeline
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
      lastFrameCpuMs: this.lastFrameCpuMs
    };
  }

  present(
    camera: CameraSnapshot,
    aspect: number,
    width: number,
    height: number,
    atlas: AcceptedTileAtlas,
    instances: readonly AtlasInstance[],
    authoritative: boolean
  ): boolean {
    const started = performance.now();
    if (width <= 0 || height <= 0 || instances.length === 0) return false;
    this.ensureSize(width, height);
    const resources = this.resources;
    if (!resources) return false;
    const view: ExactView = { camera, aspect, width, height };
    const rawTransform = this.anchor
      ? historyTransform(this.anchor.view, view)
      : { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 };
    const admission = this.anchor
      ? admitHistory(rawTransform, this.anchor.view, view)
      : { accepted: false, packed: packTransform(rawTransform), error: 0 };
    this.worstReprojectionErrorTexels = admission.error;
    if (admission.accepted) this.historyFrames++; else this.fallbackFrames++;
    this.device.queue.writeBuffer(this.reprojectUniform, 0, new Float32Array([
      admission.packed.scaleX, admission.packed.scaleY,
      admission.packed.offsetX, admission.packed.offsetY
    ]));
    this.device.queue.writeBuffer(this.reprojectUniform, 16, new Uint32Array([
      admission.accepted ? 1 : 0, 0, 0, 0
    ]));

    const packedInstances = new Float32Array(Math.min(MAX_INSTANCES, instances.length) * INSTANCE_FLOATS);
    let count = 0;
    for (const { transform, slot } of instances) {
      if (count >= MAX_INSTANCES || !transformIsFinite(transform) || transform.scaleX <= 0 || transform.scaleY <= 0) continue;
      const left = 0.5 + (-0.5 - transform.offsetX) / transform.scaleX;
      const top = 0.5 + (-0.5 - transform.offsetY) / transform.scaleY;
      const right = 0.5 + (0.5 - transform.offsetX) / transform.scaleX;
      const bottom = 0.5 + (0.5 - transform.offsetY) / transform.scaleY;
      if (right <= 0 || bottom <= 0 || left >= 1 || top >= 1) continue;
      packedInstances.set([
        left, top, right, bottom,
        slot.x, slot.y, slot.index, slot.lease
      ], count * INSTANCE_FLOATS);
      count++;
    }
    if (count === 0) return false;
    this.instanceCount = count;
    this.device.queue.writeBuffer(this.instanceBuffer, 0, packedInstances, 0, count * INSTANCE_FLOATS);

    const historyTexture = this.anchor?.texture ?? resources.anchor;
    const reprojectGroup = this.device.createBindGroup({
      layout: this.reprojectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.reprojectUniform } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: historyTexture.createView() }
      ]
    });
    const overlayGroup = this.device.createBindGroup({
      layout: this.overlayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.instanceBuffer } },
        { binding: 1, resource: atlas.colour.createView() },
        { binding: 2, resource: atlas.quality.createView() },
        { binding: 3, resource: { buffer: atlas.leaseDirectory } }
      ]
    });
    const presentGroup = this.device.createBindGroup({
      layout: this.presentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: resources.candidate.createView() }
      ]
    });
    const encoder = this.device.createCommandEncoder({ label: 'atlas-history-frame' });
    const compose = encoder.beginRenderPass({ colorAttachments: [{
      view: resources.candidate.createView(),
      clearValue: { r: 0.008, g: 0.01, b: 0.014, a: 1 }, loadOp: 'clear', storeOp: 'store'
    }] });
    compose.setPipeline(this.reprojectPipeline); compose.setBindGroup(0, reprojectGroup); compose.draw(3);
    compose.setPipeline(this.overlayPipeline); compose.setBindGroup(0, overlayGroup); compose.draw(6, count);
    compose.end();
    const canvasPass = encoder.beginRenderPass({ colorAttachments: [{
      view: this.context.getCurrentTexture().createView(),
      clearValue: { r: 0.008, g: 0.01, b: 0.014, a: 1 }, loadOp: 'clear', storeOp: 'store'
    }] });
    canvasPass.setPipeline(this.presentPipeline); canvasPass.setBindGroup(0, presentGroup); canvasPass.draw(3);
    canvasPass.end();
    this.device.queue.submit([encoder.finish()]);
    this.frames++;

    if (authoritative) this.promote(resources, view);
    this.lastFrameCpuMs = performance.now() - started;
    return true;
  }

  destroy(): void {
    const owners = new Set<ResourceSet>();
    if (this.resources) owners.add(this.resources);
    if (this.anchor) owners.add(this.anchor.owner);
    for (const owner of owners) this.destroySet(owner);
    this.resources = null;
    this.anchor = null;
    this.reprojectUniform.destroy();
    this.instanceBuffer.destroy();
  }

  private promote(resources: ResourceSet, view: ExactView): void {
    const previousOwner = this.anchor?.owner ?? null;
    const promoted = resources.candidate;
    resources.candidate = resources.anchor;
    resources.anchor = promoted;
    this.anchor = { texture: resources.anchor, owner: resources, view };
    this.anchorPromotions++;
    if (previousOwner && previousOwner !== resources) {
      void this.device.queue.onSubmittedWorkDone().then(() => this.destroySet(previousOwner));
    }
  }

  private ensureSize(width: number, height: number): void {
    if (this.resources?.width === width && this.resources.height === height) return;
    const previous = this.resources;
    this.resourceEpoch++;
    this.resources = this.createSet(width, height, this.resourceEpoch);
    if (previous && this.anchor?.owner !== previous) this.destroySet(previous);
  }

  private createSet(width: number, height: number, epoch: number): ResourceSet {
    const create = (label: string) => this.device.createTexture({
      label, size: [width, height], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    return { anchor: create(`atlas-anchor-${epoch}`), candidate: create(`atlas-candidate-${epoch}`), width, height, epoch };
  }

  private destroySet(resources: ResourceSet): void {
    resources.anchor.destroy(); resources.candidate.destroy();
  }

  private static async assertShaderValid(module: GPUShaderModule, label: string): Promise<void> {
    const info = await module.getCompilationInfo();
    const error = info.messages.find(message => message.type === 'error');
    if (error) throw new Error(`${label} WGSL line ${error.lineNum}: ${error.message}`);
  }
}
