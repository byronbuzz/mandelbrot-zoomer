import './style.css';
import {
  fixedAddScaled,
  fixedFromNumber,
  fixedRescale,
  fixedSplitF32,
  requiredCoordinateBits,
  serializeFixed,
  type BigFixed
} from './bigFixed';
import {
  normalizeScale,
  scaleDeltaParts,
  scaleLog2,
  scaleLog10,
  scaleMultiply,
  type BinaryScale
} from './binaryScale';
import type { ReferenceRequest, ReferenceResponse } from './referenceProtocol';

const app = document.querySelector<HTMLElement>('#app');
if (!app) throw new Error('Missing app root');

const ITERATION_MIN = 50;
const ITERATION_MAX = 100_000;
const ITERATION_RATIO = ITERATION_MAX / ITERATION_MIN;
const INITIAL_ITERATION_SLIDER = Math.log(500 / ITERATION_MIN) / Math.log(ITERATION_RATIO);

app.innerHTML = `
<section class="shell">
  <canvas id="fractal"></canvas>
  <header><strong>Mandelbrot Zoomer</strong><span id="status">Initialising WebGPU…</span></header>
  <aside>
    <h2>Render</h2>
    <label>Zoom speed <output id="speedOut">1.00×/s</output><input id="speed" type="range" min="0.25" max="8" step="0.25" value="1"></label>
    <label>Iterations <output id="iterOut">500</output><input id="iterations" type="range" min="0" max="1" step="0.001" value="${INITIAL_ITERATION_SLIDER.toFixed(3)}"></label>
    <label>Palette phase <output id="palOut">0.00</output><input id="palette" type="range" min="0" max="1" step="0.005" value="0"></label>
    <label>Zoom depth <output id="zoomOut">1.00× · 10^0.00</output></label>
    <label>Precision <output id="precisionOut">f32</output></label>
    <label>Reference orbit <output id="orbitOut">inactive</output></label>
    <label>Coordinate bits <output id="bitsOut">160</output></label>
    <label>Scale exponent <output id="exponentOut">2</output></label>
    <label>Render state <output id="stateOut">full-quality</output></label>
    <label>GPU frame rate <output id="fpsOut">0.0 FPS</output></label>
    <label>Effective quality <output id="qualityOut">500 / 500 iter · 100%</output></label>
    <p><b>Numerical core V2:</b> arbitrary fixed-point camera coordinates, exponent-separated scale and an asynchronously generated high-precision centre reference orbit.</p>
  </aside>
</section>`;

function el<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`Missing element: ${selector}`);
  return node;
}

const canvas = el<HTMLCanvasElement>('#fractal');
const status = el<HTMLElement>('#status');
const speed = el<HTMLInputElement>('#speed');
const iterations = el<HTMLInputElement>('#iterations');
const palette = el<HTMLInputElement>('#palette');
const zoomOut = el<HTMLOutputElement>('#zoomOut');
const precisionOut = el<HTMLOutputElement>('#precisionOut');
const orbitOut = el<HTMLOutputElement>('#orbitOut');
const bitsOut = el<HTMLOutputElement>('#bitsOut');
const exponentOut = el<HTMLOutputElement>('#exponentOut');
const stateOut = el<HTMLOutputElement>('#stateOut');
const fpsOut = el<HTMLOutputElement>('#fpsOut');
const qualityOut = el<HTMLOutputElement>('#qualityOut');
const speedOut = el<HTMLOutputElement>('#speedOut');
const iterOut = el<HTMLOutputElement>('#iterOut');
const palOut = el<HTMLOutputElement>('#palOut');

const shader = `
struct Params {
  centerX: vec2f,
  centerY: vec2f,
  scaleMantissa: f32,
  aspect: f32,
  iterations: u32,
  phase: f32,
  width: u32,
  height: u32,
  mode: u32,
  orbitLength: u32,
  scaleExponent: i32,
  _pad0: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> referenceOrbit: array<vec4f>;

fn twoSum(a:f32,b:f32)->vec2f { let s=a+b; let bb=s-a; return vec2f(s,(a-(s-bb))+(b-bb)); }
fn dsAdd(a:vec2f,b:vec2f)->vec2f { let s=twoSum(a.x,b.x); let e=s.y+a.y+b.y; let h=twoSum(s.x,e); return vec2f(h.x,h.y); }
fn dsSub(a:vec2f,b:vec2f)->vec2f { return dsAdd(a,vec2f(-b.x,-b.y)); }
fn dsMul(a:vec2f,b:vec2f)->vec2f { let q=a.x*b.x; let e=fma(a.x,b.x,-q)+a.x*b.y+a.y*b.x+a.y*b.y; let h=twoSum(q,e); return vec2f(h.x,h.y); }
fn dsFrom(v:f32)->vec2f { return vec2f(v,0.0); }
fn complexSquare(x:vec2f,y:vec2f)->array<vec2f,2> { return array<vec2f,2>(dsSub(dsMul(x,x),dsMul(y,y)),dsMul(dsMul(x,y),vec2f(2.0,0.0))); }
fn paletteColour(t:f32)->vec3f { return .5+.5*cos(6.28318*(vec3f(t)+vec3f(0.0,.12,.24)+p.phase)); }
fn writeResult(id:vec2u,escaped:bool,iteration:u32,radius:f32){
  if(!escaped){textureStore(outTex,vec2i(id),vec4f(0,0,0,1));return;}
  let smoothValue=f32(iteration)+1.0-log2(log2(sqrt(max(radius,1.0001))));
  textureStore(outTex,vec2i(id),vec4f(paletteColour(fract(.018*smoothValue)),1));
}
fn pixelDelta(ox:f32,oy:f32)->vec2f {
  return vec2f(ldexp(ox*p.scaleMantissa,p.scaleExponent),ldexp(oy*p.scaleMantissa,p.scaleExponent));
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid:vec3u){
  let id=gid.xy;
  if(id.x>=p.width||id.y>=p.height){return;}
  let uv=(vec2f(id)+.5)/vec2f(f32(p.width),f32(p.height));
  let ox=(uv.x-.5)*p.aspect;
  let oy=uv.y-.5;
  let dc=pixelDelta(ox,oy);

  if(p.mode==0u){
    let c=vec2f(p.centerX.x,p.centerY.x)+dc;
    var z=vec2f(0.0); var iteration=0u; var radius=0.0;
    loop{radius=dot(z,z);if(iteration>=p.iterations||radius>256.0){break;}z=vec2f(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;iteration++;}
    writeResult(id,iteration<p.iterations,iteration,radius);return;
  }

  let cx=dsAdd(p.centerX,dsFrom(dc.x));
  let cy=dsAdd(p.centerY,dsFrom(dc.y));
  if(p.mode==1u){
    var zx=vec2f(0.0); var zy=vec2f(0.0); var iteration=0u; var radius=0.0;
    loop{
      radius=zx.x*zx.x+zy.x*zy.x;
      if(iteration>=p.iterations||radius>256.0){break;}
      let sq=complexSquare(zx,zy); zx=dsAdd(sq[0],cx); zy=dsAdd(sq[1],cy); iteration++;
    }
    writeResult(id,iteration<p.iterations,iteration,radius);return;
  }

  var dz=vec2f(0.0); var currentZ=vec2f(0.0); var iteration=0u; var radius=0.0; var direct=false;
  loop{
    let parts=referenceOrbit[iteration];
    let rz=vec2f(parts.x+parts.y,parts.z+parts.w);
    currentZ=rz+dz; radius=dot(currentZ,currentZ);
    if(iteration>=p.iterations||radius>256.0){break;}
    if(iteration+1u>=p.orbitLength){direct=true;break;}
    let dz2=vec2f(dz.x*dz.x-dz.y*dz.y,2.0*dz.x*dz.y);
    let cross=vec2f(2.0*(rz.x*dz.x-rz.y*dz.y),2.0*(rz.x*dz.y+rz.y*dz.x));
    dz=cross+dz2+dc;
    if(any(abs(dz)>vec2f(1e12))){direct=true;break;}
    iteration++;
  }
  if(direct&&iteration<p.iterations&&radius<=256.0){
    var zx=vec2f(currentZ.x,0.0); var zy=vec2f(currentZ.y,0.0);
    loop{
      radius=zx.x*zx.x+zy.x*zy.x;
      if(iteration>=p.iterations||radius>256.0){break;}
      let sq=complexSquare(zx,zy); zx=dsAdd(sq[0],cx); zy=dsAdd(sq[1],cy); iteration++;
    }
  }
  writeResult(id,iteration<p.iterations,iteration,radius);
}`;

if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a current Chromium browser with hardware acceleration enabled.');
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('No WebGPU adapter found');
const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');
if (!context) throw new Error('Unable to create WebGPU canvas context');
const gpuContext: GPUCanvasContext = context;
const canvasFormat: GPUTextureFormat = 'rgba8unorm';
gpuContext.configure({ device, format: canvasFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST, alphaMode: 'opaque' });
const module = device.createShaderModule({ code: shader });
const compilation = await module.getCompilationInfo();
const shaderErrors = compilation.messages.filter(message => message.type === 'error');
if (shaderErrors.length) {
  const first = shaderErrors[0];
  const detail = `WGSL line ${first.lineNum}: ${first.message}`;
  status.textContent = detail;
  throw new Error(detail);
}
const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const orbitBuffer = device.createBuffer({ size: (ITERATION_MAX + 1) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
status.textContent = `${adapter.info.vendor || 'GPU'} · WebGPU · numerical core V2`;
device.addEventListener('uncapturederror', event => { status.textContent = `WebGPU error: ${event.error.message}`; });
device.lost.then(reason => { status.textContent = `GPU device lost: ${reason.message || reason.reason}`; }).catch(() => undefined);

const worker = new Worker(new URL('./referenceWorker.ts', import.meta.url), { type: 'module' });
const TARGET_FPS = 60;
const SETTLE_MS = 180;
const REFINE_DELAY_MS = 90;
const MIN_RESOLUTION_SCALE = .35;
const MIN_ITERATION_SCALE = .70;
const DOUBLE_FLOAT_THRESHOLD_LOG10 = 4;
const PERTURBATION_THRESHOLD_LOG10 = 5;

type RenderStage = 'interactive' | 'refining' | 'full-quality';
type DisplayPrecision = 'f32' | 'double-float' | 'perturbation' | 'building reference';
type ReferenceCache = { key: string; iterations: number; length: number; escaped: boolean; bits: number; ms: number };

let coordinateBits = 160;
let centerX: BigFixed = fixedFromNumber(-.5, coordinateBits);
let centerY: BigFixed = fixedFromNumber(0, coordinateBits);
let viewportScale: BinaryScale = normalizeScale(.75, 2);
let direction = 0, px = .5, py = .5;
let panning = false, panPointer = -1, panX = 0, panY = 0;
let renderRequested = false, frameInFlight = false, pumpScheduled = false;
let lastInteraction = -Infinity, settleTimer = 0, refineTimer = 0;
let adaptiveResolutionScale = 1, adaptiveIterationScale = 1;
let fpsValue = 0, smoothedFrameMs = 1000 / TARGET_FPS, displayedIterations = 500, displayedResolution = 1;
let renderStage: RenderStage = 'full-quality', refineStep = 0;
let displayedPrecision: DisplayPrecision = 'f32';
let orbitStatus = 'inactive';
let referenceCache: ReferenceCache | null = null;
let pendingReferenceKey = '';
let referenceRequestId = 0;

function requestedIterations(): number {
  return Math.round(ITERATION_MIN * Math.pow(ITERATION_RATIO, Number(iterations.value)));
}
function log10Magnification(): number { return Math.log10(3) - scaleLog10(viewportScale); }
function ensureCoordinatePrecision(): void {
  const needed = requiredCoordinateBits(-scaleLog2(viewportScale) + Math.log2(3));
  if (needed <= coordinateBits) return;
  coordinateBits = needed;
  centerX = fixedRescale(centerX, coordinateBits);
  centerY = fixedRescale(centerY, coordinateBits);
  referenceCache = null;
}
function isInteractive(now = performance.now()): boolean { return direction !== 0 || panning || now - lastInteraction < SETTLE_MS; }
function resetController(): void { adaptiveResolutionScale = 1; adaptiveIterationScale = 1; smoothedFrameMs = 1000 / TARGET_FPS; }
function clearTimers(): void { if (settleTimer) clearTimeout(settleTimer); if (refineTimer) clearTimeout(refineTimer); }
function markInteraction(): void {
  lastInteraction = performance.now(); renderStage = 'interactive'; refineStep = 0; clearTimers();
  settleTimer = window.setTimeout(() => { renderStage = 'refining'; refineStep = 1; requestRender(); }, SETTLE_MS);
}
function roundedIterations(value: number): number {
  if (value < 1000) return Math.max(ITERATION_MIN, Math.round(value / 50) * 50);
  if (value < 10_000) return Math.round(value / 100) * 100;
  return Math.round(value / 1000) * 1000;
}
function effectiveQuality(stage: RenderStage): { iterations: number; resolution: number } {
  const requested = requestedIterations();
  if (stage === 'full-quality') return { iterations: requested, resolution: 1 };
  if (stage === 'refining') {
    const progress = refineStep === 1 ? .65 : .85;
    return { iterations: roundedIterations(requested * (.85 + .15 * progress)), resolution: .55 + .45 * progress };
  }
  return { iterations: roundedIterations(requested * adaptiveIterationScale), resolution: Math.max(MIN_RESOLUTION_SCALE, adaptiveResolutionScale) };
}
function updateController(frameMs: number): void {
  const target = 1000 / TARGET_FPS;
  if (frameMs > target * 1.08) {
    if (adaptiveResolutionScale > MIN_RESOLUTION_SCALE + .001) adaptiveResolutionScale = Math.max(MIN_RESOLUTION_SCALE, adaptiveResolutionScale * .88);
    else adaptiveIterationScale = Math.max(MIN_ITERATION_SCALE, adaptiveIterationScale * .94);
  } else if (frameMs < target * .78) {
    if (adaptiveIterationScale < .999) adaptiveIterationScale = Math.min(1, adaptiveIterationScale + .03);
    else adaptiveResolutionScale = Math.min(1, adaptiveResolutionScale * 1.06 + .01);
  }
}
function referenceKey(iterationLimit: number): string {
  return `${centerX.raw}:${centerX.bits}|${centerY.raw}:${centerY.bits}|${iterationLimit}`;
}
function requestReference(iterationLimit: number): boolean {
  const key = referenceKey(iterationLimit);
  if (referenceCache?.key === key && referenceCache.iterations >= iterationLimit) return true;
  if (pendingReferenceKey === key) return false;
  pendingReferenceKey = key;
  referenceRequestId++;
  const request: ReferenceRequest = {
    id: referenceRequestId,
    centerX: serializeFixed(centerX),
    centerY: serializeFixed(centerY),
    iterations: iterationLimit
  };
  orbitStatus = `building ${coordinateBits}-bit reference…`;
  worker.postMessage(request);
  return false;
}
worker.addEventListener('message', event => {
  const response = event.data as ReferenceResponse;
  if (response.id !== referenceRequestId) return;
  const key = pendingReferenceKey;
  pendingReferenceKey = '';
  device.queue.writeBuffer(orbitBuffer, 0, response.orbit);
  referenceCache = {
    key,
    iterations: response.length - 1,
    length: response.length,
    escaped: response.escaped,
    bits: response.bits,
    ms: response.generationMs
  };
  requestRender();
});

function updateReadouts(): void {
  const order = log10Magnification();
  const requested = requestedIterations();
  zoomOut.value = `10^${order.toFixed(2)}`;
  precisionOut.value = displayedPrecision;
  orbitOut.value = orbitStatus;
  bitsOut.value = `${coordinateBits}`;
  exponentOut.value = `${viewportScale.exponent}`;
  stateOut.value = renderStage;
  fpsOut.value = `${fpsValue.toFixed(1)} FPS`;
  qualityOut.value = `${displayedIterations} / ${requested} iter · ${Math.round(displayedResolution * 100)}%`;
  speedOut.value = `${Number(speed.value).toFixed(2)}×/s`;
  iterOut.value = requested.toLocaleString();
  palOut.value = Number(palette.value).toFixed(2);
}

async function renderFrame(): Promise<void> {
  frameInFlight = true; renderRequested = false;
  ensureCoordinatePrecision();
  const activeStage: RenderStage = isInteractive() ? 'interactive' : renderStage;
  const quality = effectiveQuality(activeStage);
  displayedIterations = quality.iterations; displayedResolution = quality.resolution;
  const w = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio * quality.resolution));
  const h = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio * quality.resolution));
  canvas.width = w; canvas.height = h;

  const order = log10Magnification();
  let mode = order < DOUBLE_FLOAT_THRESHOLD_LOG10 ? 0 : 1;
  orbitStatus = 'inactive';
  if (order >= PERTURBATION_THRESHOLD_LOG10) {
    if (requestReference(quality.iterations) && referenceCache) {
      mode = 2;
      displayedPrecision = 'perturbation';
      orbitStatus = `${referenceCache.bits}-bit · ${referenceCache.length - 1} iter · ${referenceCache.escaped ? 'hybrid' : 'complete'} · ${referenceCache.ms.toFixed(1)} ms`;
    } else {
      mode = 1;
      displayedPrecision = 'building reference';
    }
  } else displayedPrecision = mode === 0 ? 'f32' : 'double-float';

  const [cxHi, cxLo] = fixedSplitF32(centerX);
  const [cyHi, cyLo] = fixedSplitF32(centerY);
  const texture = device.createTexture({ size: [w, h], format: canvasFormat, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
  const data = new ArrayBuffer(64), f = new Float32Array(data), u = new Uint32Array(data), i = new Int32Array(data);
  f[0] = cxHi; f[1] = cxLo; f[2] = cyHi; f[3] = cyLo;
  f[4] = Math.fround(viewportScale.mantissa); f[5] = w / h; u[6] = quality.iterations; f[7] = Number(palette.value);
  u[8] = w; u[9] = h; u[10] = mode; u[11] = mode === 2 && referenceCache ? referenceCache.length : 1;
  i[12] = viewportScale.exponent;
  device.queue.writeBuffer(params, 0, data);
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: texture.createView() },
      { binding: 2, resource: { buffer: orbitBuffer } }
    ]
  });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); pass.end();
  encoder.copyTextureToTexture({ texture }, { texture: gpuContext.getCurrentTexture() }, { width: w, height: h });
  const started = performance.now();
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  const frameMs = Math.max(.1, performance.now() - started); texture.destroy();
  smoothedFrameMs = smoothedFrameMs * .78 + frameMs * .22; fpsValue = 1000 / smoothedFrameMs;
  if (activeStage === 'interactive') updateController(smoothedFrameMs);
  frameInFlight = false; updateReadouts();
  if (!isInteractive() && renderStage === 'refining') {
    if (refineStep < 2) refineTimer = window.setTimeout(() => { refineStep++; requestRender(); }, REFINE_DELAY_MS);
    else refineTimer = window.setTimeout(() => { renderStage = 'full-quality'; requestRender(); }, REFINE_DELAY_MS);
  }
  if (renderRequested) schedulePump();
}
function schedulePump(): void {
  if (pumpScheduled) return;
  pumpScheduled = true;
  requestAnimationFrame(() => { pumpScheduled = false; if (!frameInFlight && renderRequested) void renderFrame(); });
}
function requestRender(): void { renderRequested = true; schedulePump(); }
function pointer(event: { clientX: number; clientY: number }): void {
  const r = canvas.getBoundingClientRect(); px = (event.clientX - r.left) / r.width; py = (event.clientY - r.top) / r.height;
}
function moveByScaleMultipliers(xMultiplier: number, yMultiplier: number): void {
  const dx = scaleDeltaParts(viewportScale, xMultiplier);
  const dy = scaleDeltaParts(viewportScale, yMultiplier);
  centerX = fixedAddScaled(centerX, dx.mantissa, dx.exponent);
  centerY = fixedAddScaled(centerY, dy.mantissa, dy.exponent);
  referenceCache = null;
}
function zoomAboutPointer(factor: number, aspect: number): void {
  moveByScaleMultipliers((px - .5) * aspect * (1 - factor), (py - .5) * (1 - factor));
  viewportScale = scaleMultiply(viewportScale, factor);
  ensureCoordinatePrecision();
}

canvas.addEventListener('pointermove', event => {
  pointer(event);
  if (!panning || event.pointerId !== panPointer) return;
  const r = canvas.getBoundingClientRect();
  moveByScaleMultipliers(-(event.clientX - panX) / r.height, -(event.clientY - panY) / r.height);
  panX = event.clientX; panY = event.clientY; markInteraction(); requestRender();
});
canvas.addEventListener('pointerdown', event => {
  pointer(event); canvas.setPointerCapture(event.pointerId);
  if (event.button === 1) { panning = true; panPointer = event.pointerId; panX = event.clientX; panY = event.clientY; direction = 0; }
  else direction = event.button === 2 ? -1 : 1;
  markInteraction(); requestRender(); event.preventDefault();
});
function endPointer(event: PointerEvent): void {
  if (event.pointerId === panPointer) { panning = false; panPointer = -1; }
  direction = 0; lastInteraction = performance.now() - SETTLE_MS; clearTimers();
  renderStage = 'refining'; refineStep = 1; requestRender();
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('contextmenu', event => event.preventDefault());
canvas.addEventListener('auxclick', event => event.preventDefault());
canvas.addEventListener('wheel', event => {
  pointer(event); zoomAboutPointer(Math.exp(event.deltaY * .0012), canvas.clientWidth / canvas.clientHeight);
  markInteraction(); requestRender(); event.preventDefault();
}, { passive: false });
for (const input of [speed, iterations, palette]) input.addEventListener('input', () => {
  if (input === iterations) { resetController(); referenceCache = null; }
  if (input !== speed) { renderStage = 'full-quality'; refineStep = 0; }
  requestRender();
});
new ResizeObserver(() => { resetController(); requestRender(); }).observe(canvas);

let last = performance.now();
function tick(now: number): void {
  const dt = Math.min(.05, (now - last) / 1000); last = now;
  if (direction) {
    zoomAboutPointer(Math.exp(-direction * Number(speed.value) * dt), canvas.clientWidth / canvas.clientHeight);
    markInteraction(); requestRender();
  }
  requestAnimationFrame(tick);
}
updateReadouts(); requestRender(); requestAnimationFrame(tick);
