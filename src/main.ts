import './style.css';

const app = document.querySelector<HTMLElement>('#app');
if (!app) throw new Error('Missing app root');

app.innerHTML = `
<section class="shell">
  <canvas id="fractal"></canvas>
  <header><strong>Mandelbrot Zoomer</strong><span id="status">Initialising WebGPU…</span></header>
  <aside>
    <h2>Render</h2>
    <label>Zoom depth <output id="zoomOut">1.00× · 10^0.00</output></label>
    <label>Precision <output id="precisionOut">f32</output></label>
    <label>Render state <output id="stateOut">full-quality</output></label>
    <label>GPU frame rate <output id="fpsOut">0.0 FPS</output></label>
    <label>Effective quality <output id="qualityOut">500 iter · 100%</output></label>
    <label>Zoom speed <output id="speedOut">1.00×/s</output><input id="speed" type="range" min="0.25" max="8" step="0.25" value="1"></label>
    <label>Iterations <output id="iterOut">500</output><input id="iterations" type="range" min="50" max="10000" step="50" value="500"></label>
    <label>Internal resolution <output id="resOut">100%</output><input id="resolution" type="range" min="0.35" max="1" step="0.05" value="1"></label>
    <label>Palette phase <output id="palOut">0.00</output><input id="palette" type="range" min="0" max="1" step="0.005" value="0"></label>
    <p><b>XaoS-style navigation:</b> hold left mouse to zoom toward the pointer; hold right mouse to zoom out; middle-drag to pan. During movement, quality adapts toward 60 FPS; full quality is restored after movement stops.</p>
  </aside>
</section>`;

const canvas = document.querySelector<HTMLCanvasElement>('#fractal')!;
const status = document.querySelector<HTMLElement>('#status')!;
const speed = document.querySelector<HTMLInputElement>('#speed')!;
const iterations = document.querySelector<HTMLInputElement>('#iterations')!;
const resolution = document.querySelector<HTMLInputElement>('#resolution')!;
const palette = document.querySelector<HTMLInputElement>('#palette')!;
const zoomOut = document.querySelector<HTMLOutputElement>('#zoomOut')!;
const precisionOut = document.querySelector<HTMLOutputElement>('#precisionOut')!;
const stateOut = document.querySelector<HTMLOutputElement>('#stateOut')!;
const fpsOut = document.querySelector<HTMLOutputElement>('#fpsOut')!;
const qualityOut = document.querySelector<HTMLOutputElement>('#qualityOut')!;
const speedOut = document.querySelector<HTMLOutputElement>('#speedOut')!;
const iterOut = document.querySelector<HTMLOutputElement>('#iterOut')!;
const resOut = document.querySelector<HTMLOutputElement>('#resOut')!;
const palOut = document.querySelector<HTMLOutputElement>('#palOut')!;

const shader = `
struct Params {
  centerX: vec2f,
  centerY: vec2f,
  scale: vec2f,
  aspect: f32,
  iterations: u32,
  phase: f32,
  width: u32,
  height: u32,
  mode: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;

fn dsAdd(a: vec2f, b: vec2f) -> vec2f {
  let s = a.x + b.x;
  let v = s - a.x;
  let e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
  let hi = s + e;
  return vec2f(hi, e - (hi - s));
}
fn dsSub(a: vec2f, b: vec2f) -> vec2f { return dsAdd(a, vec2f(-b.x, -b.y)); }
fn dsMul(a: vec2f, b: vec2f) -> vec2f {
  let product = a.x * b.x;
  let error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x + a.y * b.y;
  let hi = product + error;
  return vec2f(hi, error - (hi - product));
}
fn dsScale(a: vec2f, b: f32) -> vec2f {
  let product = a.x * b;
  let error = fma(a.x, b, -product) + a.y * b;
  let hi = product + error;
  return vec2f(hi, error - (hi - product));
}
fn paletteColour(t: f32) -> vec3f {
  return .5 + .5 * cos(6.28318 * (vec3f(t) + vec3f(0.0, .12, .24) + p.phase));
}
fn writeResult(id: vec2u, escaped: bool, iteration: u32, radius: f32) {
  if (!escaped) {
    textureStore(outTex, vec2i(id), vec4f(0, 0, 0, 1));
    return;
  }
  let smoothValue = f32(iteration) + 1.0 - log2(log2(sqrt(radius)));
  textureStore(outTex, vec2i(id), vec4f(paletteColour(fract(.018 * smoothValue)), 1));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.xy;
  if (id.x >= p.width || id.y >= p.height) { return; }
  let uv = (vec2f(id) + .5) / vec2f(f32(p.width), f32(p.height));
  let ox = (uv.x - .5) * p.aspect;
  let oy = uv.y - .5;

  if (p.mode == 0u) {
    let c = vec2f(p.centerX.x, p.centerY.x) + vec2f(ox, oy) * p.scale.x;
    let q = (c.x - .25) * (c.x - .25) + c.y * c.y;
    if (q * (q + c.x - .25) <= .25 * c.y * c.y || (c.x + 1.) * (c.x + 1.) + c.y * c.y <= .0625) {
      textureStore(outTex, vec2i(id), vec4f(0, 0, 0, 1)); return;
    }
    var z = vec2f(0.);
    var iteration = 0u;
    var radius = 0.0;
    loop {
      radius = dot(z, z);
      if (iteration >= p.iterations || radius > 256.) { break; }
      z = vec2f(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
      iteration++;
    }
    writeResult(id, iteration < p.iterations, iteration, radius);
    return;
  }

  let cx = dsAdd(p.centerX, dsScale(p.scale, ox));
  let cy = dsAdd(p.centerY, dsScale(p.scale, oy));
  let cxApprox = cx.x + cx.y;
  let cyApprox = cy.x + cy.y;
  let q = (cxApprox - .25) * (cxApprox - .25) + cyApprox * cyApprox;
  if (q * (q + cxApprox - .25) <= .25 * cyApprox * cyApprox || (cxApprox + 1.) * (cxApprox + 1.) + cyApprox * cyApprox <= .0625) {
    textureStore(outTex, vec2i(id), vec4f(0, 0, 0, 1)); return;
  }
  var zx = vec2f(0.);
  var zy = vec2f(0.);
  var iteration = 0u;
  var radius = 0.0;
  loop {
    radius = zx.x * zx.x + zy.x * zy.x;
    if (iteration >= p.iterations || radius > 256.) { break; }
    let zx2 = dsMul(zx, zx);
    let zy2 = dsMul(zy, zy);
    let zxy = dsMul(zx, zy);
    zx = dsAdd(dsSub(zx2, zy2), cx);
    zy = dsAdd(dsScale(zxy, 2.0), cy);
    iteration++;
  }
  writeResult(id, iteration < p.iterations, iteration, radius);
}`;

if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a current Chromium browser with hardware acceleration enabled.');
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('No WebGPU adapter found');
const features: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
const device = await adapter.requestDevice({ requiredFeatures: features });
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
const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const info = adapter.info;
status.textContent = `${info.vendor || 'GPU'} · WebGPU${device.features.has('timestamp-query') ? ' · GPU timing' : ''}`;

device.addEventListener('uncapturederror', event => { status.textContent = `WebGPU error: ${event.error.message}`; });
device.lost.then(reason => { status.textContent = `GPU device lost: ${reason.message || reason.reason}`; }).catch(() => undefined);

const DOUBLE_FLOAT_THRESHOLD = 1e4;
const TARGET_FRAME_MS = 1000 / 60;
const SETTLE_MS = 180;
const MIN_ITERATION_SCALE = 0.08;
const MIN_RESOLUTION_SCALE = 0.35;

let centerX = -0.5, centerY = 0, scale = 3, direction = 0, px = .5, py = .5;
let panning = false, panPointer = -1, panX = 0, panY = 0;
let renderRequested = false, frameInFlight = false, pumpScheduled = false;
let lastInteraction = -Infinity, refinementTimer = 0;
let adaptiveIterationScale = 1, adaptiveResolutionScale = 1;
let fpsValue = 0, smoothedFrameMs = TARGET_FRAME_MS;
let displayedIterations = 500, displayedResolution = 1, displayedInteractive = false;

function split(value: number): [number, number] {
  const hi = Math.fround(value);
  return [hi, Math.fround(value - hi)];
}
function isInteractive(now = performance.now()): boolean {
  return direction !== 0 || panning || now - lastInteraction < SETTLE_MS;
}
function markInteraction(): void {
  lastInteraction = performance.now();
  if (refinementTimer) window.clearTimeout(refinementTimer);
  refinementTimer = window.setTimeout(() => requestRender(), SETTLE_MS + 10);
}
function roundedIterations(value: number): number {
  if (value <= 100) return Math.max(50, Math.round(value / 10) * 10);
  return Math.max(50, Math.round(value / 50) * 50);
}
function effectiveQuality(interactive: boolean): { iterations: number; resolution: number } {
  const requestedIterations = Number(iterations.value);
  const requestedResolution = Number(resolution.value);
  if (!interactive) return { iterations: requestedIterations, resolution: requestedResolution };
  return {
    iterations: roundedIterations(Math.max(50, requestedIterations * adaptiveIterationScale)),
    resolution: Math.max(0.35, requestedResolution * adaptiveResolutionScale)
  };
}
function updateController(frameMs: number): void {
  if (frameMs > TARGET_FRAME_MS * 1.08) {
    if (adaptiveIterationScale > MIN_ITERATION_SCALE + 0.001) adaptiveIterationScale = Math.max(MIN_ITERATION_SCALE, adaptiveIterationScale * 0.82);
    else adaptiveResolutionScale = Math.max(MIN_RESOLUTION_SCALE, adaptiveResolutionScale * 0.9);
  } else if (frameMs < TARGET_FRAME_MS * 0.78) {
    if (adaptiveResolutionScale < 0.995) adaptiveResolutionScale = Math.min(1, adaptiveResolutionScale * 1.06 + 0.01);
    else adaptiveIterationScale = Math.min(1, adaptiveIterationScale * 1.08 + 0.01);
  }
}
function updateReadouts(): void {
  const magnification = 3 / scale;
  const orders = Math.log10(magnification);
  zoomOut.value = `${magnification < 1000 ? magnification.toFixed(2) : magnification.toExponential(2)}× · 10^${orders.toFixed(2)}`;
  precisionOut.value = magnification >= DOUBLE_FLOAT_THRESHOLD ? 'double-float' : 'f32';
  stateOut.value = displayedInteractive ? 'interactive' : 'full-quality';
  fpsOut.value = `${fpsValue.toFixed(1)} FPS`;
  qualityOut.value = `${displayedIterations} iter · ${Math.round(displayedResolution * 100)}%`;
  speedOut.value = `${Number(speed.value).toFixed(2)}×/s`;
  iterOut.value = iterations.value;
  resOut.value = `${Math.round(Number(resolution.value) * 100)}%`;
  palOut.value = Number(palette.value).toFixed(2);
}

async function renderFrame(): Promise<void> {
  frameInFlight = true;
  renderRequested = false;
  const interactive = isInteractive();
  const quality = effectiveQuality(interactive);
  displayedInteractive = interactive;
  displayedIterations = quality.iterations;
  displayedResolution = quality.resolution;

  const w = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio * quality.resolution));
  const h = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio * quality.resolution));
  canvas.width = w;
  canvas.height = h;

  const magnification = 3 / scale;
  const deepMode = magnification >= DOUBLE_FLOAT_THRESHOLD;
  const [cxHi, cxLo] = split(centerX);
  const [cyHi, cyLo] = split(centerY);
  const [scaleHi, scaleLo] = split(scale);
  const texture = device.createTexture({ size: [w, h], format: canvasFormat, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
  const data = new ArrayBuffer(48), f = new Float32Array(data), u = new Uint32Array(data);
  f[0] = cxHi; f[1] = cxLo; f[2] = cyHi; f[3] = cyLo; f[4] = scaleHi; f[5] = scaleLo;
  f[6] = w / h; u[7] = quality.iterations; f[8] = Number(palette.value);
  u[9] = w; u[10] = h; u[11] = deepMode ? 1 : 0;
  device.queue.writeBuffer(params, 0, data);

  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: texture.createView() }]
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
  pass.end();
  encoder.copyTextureToTexture({ texture }, { texture: gpuContext.getCurrentTexture() }, { width: w, height: h });

  const started = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const frameMs = Math.max(0.1, performance.now() - started);
  texture.destroy();

  smoothedFrameMs = smoothedFrameMs * 0.78 + frameMs * 0.22;
  fpsValue = 1000 / smoothedFrameMs;
  if (interactive) updateController(smoothedFrameMs);
  frameInFlight = false;
  updateReadouts();

  if (renderRequested || isInteractive()) schedulePump();
}
function schedulePump(): void {
  if (pumpScheduled) return;
  pumpScheduled = true;
  requestAnimationFrame(() => {
    pumpScheduled = false;
    if (!frameInFlight && renderRequested) void renderFrame();
  });
}
function requestRender(): void {
  renderRequested = true;
  schedulePump();
}
function pointer(e: PointerEvent): void {
  const r = canvas.getBoundingClientRect();
  px = (e.clientX - r.left) / r.width;
  py = (e.clientY - r.top) / r.height;
}

canvas.addEventListener('pointermove', e => {
  pointer(e);
  if (!panning || e.pointerId !== panPointer) return;
  const r = canvas.getBoundingClientRect();
  const dx = e.clientX - panX, dy = e.clientY - panY;
  centerX -= dx / r.width * (r.width / r.height) * scale;
  centerY -= dy / r.height * scale;
  panX = e.clientX; panY = e.clientY;
  markInteraction();
  requestRender();
});
canvas.addEventListener('pointerdown', e => {
  pointer(e);
  canvas.setPointerCapture(e.pointerId);
  if (e.button === 1) {
    panning = true; panPointer = e.pointerId; panX = e.clientX; panY = e.clientY; direction = 0;
  } else direction = e.button === 2 ? -1 : 1;
  markInteraction();
  requestRender();
  e.preventDefault();
});
function endPointer(e: PointerEvent): void {
  if (e.pointerId === panPointer) { panning = false; panPointer = -1; }
  direction = 0;
  markInteraction();
  requestRender();
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('auxclick', e => e.preventDefault());
canvas.addEventListener('wheel', e => {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const a = r.width / r.height;
  const ax = centerX + (x - .5) * a * scale, ay = centerY + (y - .5) * scale;
  const k = Math.exp(e.deltaY * .0012);
  scale *= k;
  centerX = ax + (centerX - ax) * k;
  centerY = ay + (centerY - ay) * k;
  markInteraction();
  requestRender();
  e.preventDefault();
}, { passive: false });
for (const el of [speed, iterations, resolution, palette]) el.addEventListener('input', () => {
  if (el === iterations || el === resolution) {
    adaptiveIterationScale = Math.min(adaptiveIterationScale, 1);
    adaptiveResolutionScale = Math.min(adaptiveResolutionScale, 1);
  }
  requestRender();
});
new ResizeObserver(requestRender).observe(canvas);

let last = performance.now();
function tick(now: number): void {
  const dt = Math.min(.05, (now - last) / 1000);
  last = now;
  if (direction) {
    const a = canvas.clientWidth / canvas.clientHeight;
    const ax = centerX + (px - .5) * a * scale, ay = centerY + (py - .5) * scale;
    const k = Math.exp(-direction * Number(speed.value) * dt);
    scale *= k;
    centerX = ax + (centerX - ax) * k;
    centerY = ay + (centerY - ay) * k;
    markInteraction();
    requestRender();
  }
  requestAnimationFrame(tick);
}

requestRender();
requestAnimationFrame(tick);
