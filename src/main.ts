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
    <label>GPU frame rate <output id="fpsOut">0.0 FPS</output></label>
    <label>Zoom speed <output id="speedOut">2.50×/s</output><input id="speed" type="range" min="0.25" max="8" step="0.25" value="2.5"></label>
    <label>Iterations <output id="iterOut">500</output><input id="iterations" type="range" min="50" max="3000" step="50" value="500"></label>
    <label>Internal resolution <output id="resOut">100%</output><input id="resolution" type="range" min="0.35" max="1" step="0.05" value="1"></label>
    <label>Palette phase <output id="palOut">0.00</output><input id="palette" type="range" min="0" max="1" step="0.005" value="0"></label>
    <p><b>XaoS-style navigation:</b> hold left mouse to zoom toward the pointer; hold right mouse to zoom out; middle-drag to pan. Use the wheel for discrete zooming.</p>
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
const fpsOut = document.querySelector<HTMLOutputElement>('#fpsOut')!;
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

fn ds_add(a: vec2f, b: vec2f) -> vec2f {
  let s = a.x + b.x;
  let v = s - a.x;
  let e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
  let hi = s + e;
  return vec2f(hi, e - (hi - s));
}
fn ds_sub(a: vec2f, b: vec2f) -> vec2f { return ds_add(a, -b); }
fn ds_mul(a: vec2f, b: vec2f) -> vec2f {
  let product = a.x * b.x;
  let error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x + a.y * b.y;
  let hi = product + error;
  return vec2f(hi, error - (hi - product));
}
fn ds_scale(a: vec2f, b: f32) -> vec2f {
  let product = a.x * b;
  let error = fma(a.x, b, -product) + a.y * b;
  let hi = product + error;
  return vec2f(hi, error - (hi - product));
}
fn colour(t: f32) -> vec3f {
  return .5 + .5 * cos(6.28318 * (vec3f(t) + vec3f(0.0, .12, .24) + p.phase));
}
fn storeResult(id: vec2u, escaped: bool, i: u32, radius: f32) {
  if (!escaped) {
    textureStore(outTex, vec2i(id), vec4f(0, 0, 0, 1));
    return;
  }
  let smooth = f32(i) + 1.0 - log2(log2(sqrt(radius)));
  textureStore(outTex, vec2i(id), vec4f(colour(fract(.018 * smooth)), 1));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id3: vec3u) {
  let id = id3.xy;
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
    var i = 0u;
    var radius = 0.0;
    loop {
      radius = dot(z, z);
      if (i >= p.iterations || radius > 256.) { break; }
      z = vec2f(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
      i++;
    }
    storeResult(id, i < p.iterations, i, radius);
    return;
  }

  let cx = ds_add(p.centerX, ds_scale(p.scale, ox));
  let cy = ds_add(p.centerY, ds_scale(p.scale, oy));
  let cxh = cx.x + cx.y;
  let cyh = cy.x + cy.y;
  let q = (cxh - .25) * (cxh - .25) + cyh * cyh;
  if (q * (q + cxh - .25) <= .25 * cyh * cyh || (cxh + 1.) * (cxh + 1.) + cyh * cyh <= .0625) {
    textureStore(outTex, vec2i(id), vec4f(0, 0, 0, 1)); return;
  }
  var zx = vec2f(0.);
  var zy = vec2f(0.);
  var i = 0u;
  var radius = 0.0;
  loop {
    radius = zx.x * zx.x + zy.x * zy.x;
    if (i >= p.iterations || radius > 256.) { break; }
    let zx2 = ds_mul(zx, zx);
    let zy2 = ds_mul(zy, zy);
    let zxy = ds_mul(zx, zy);
    zx = ds_add(ds_sub(zx2, zy2), cx);
    zy = ds_add(ds_scale(zxy, 2.0), cy);
    i++;
  }
  storeResult(id, i < p.iterations, i, radius);
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
gpuContext.configure({
  device,
  format: canvasFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  alphaMode: 'opaque'
});
const module = device.createShaderModule({ code: shader });
const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const info = adapter.info;
status.textContent = `${info.vendor || 'GPU'} · WebGPU${device.features.has('timestamp-query') ? ' · GPU timing' : ''}`;

const DOUBLE_FLOAT_THRESHOLD = 1e4;
let centerX = -0.5, centerY = 0, scale = 3, direction = 0, px = .5, py = .5, queued = false;
let panning = false, panPointer = -1, panX = 0, panY = 0;
let submittedFrames = 0, completedFrames = 0, completionPending = false;
let fpsLast = performance.now(), fpsValue = 0;

function split(value: number): [number, number] {
  const hi = Math.fround(value);
  return [hi, Math.fround(value - hi)];
}
function armCompletionTracker() {
  if (completionPending || submittedFrames === 0) return;
  const batch = submittedFrames;
  submittedFrames = 0;
  completionPending = true;
  void device.queue.onSubmittedWorkDone().then(() => {
    completedFrames += batch;
    completionPending = false;
    armCompletionTracker();
  });
}
function updateReadouts(rs: number, deepMode: boolean) {
  const magnification = 3 / scale;
  const orders = Math.log10(magnification);
  zoomOut.value = `${magnification < 1000 ? magnification.toFixed(2) : magnification.toExponential(2)}× · 10^${orders.toFixed(2)}`;
  precisionOut.value = deepMode ? 'double-float' : 'f32';
  fpsOut.value = `${fpsValue.toFixed(1)} FPS`;
  speedOut.value = `${Number(speed.value).toFixed(2)}×/s`;
  iterOut.value = iterations.value;
  resOut.value = `${Math.round(rs * 100)}%`;
  palOut.value = Number(palette.value).toFixed(2);
}

function render() {
  queued = false;
  const rs = Number(resolution.value);
  const w = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio * rs));
  const h = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio * rs));
  canvas.width = w;
  canvas.height = h;
  const magnification = 3 / scale;
  const deepMode = magnification >= DOUBLE_FLOAT_THRESHOLD;
  const [cxHi, cxLo] = split(centerX);
  const [cyHi, cyLo] = split(centerY);
  const [scaleHi, scaleLo] = split(scale);
  const texture = device.createTexture({
    size: [w, h],
    format: canvasFormat,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
  });
  const data = new ArrayBuffer(48), f = new Float32Array(data), u = new Uint32Array(data);
  f[0] = cxHi; f[1] = cxLo; f[2] = cyHi; f[3] = cyLo; f[4] = scaleHi; f[5] = scaleLo;
  f[6] = w / h; u[7] = Number(iterations.value); f[8] = Number(palette.value);
  u[9] = w; u[10] = h; u[11] = deepMode ? 1 : 0;
  device.queue.writeBuffer(params, 0, data);
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: texture.createView() }]
  });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
  pass.end();
  encoder.copyTextureToTexture({ texture }, { texture: gpuContext.getCurrentTexture() }, { width: w, height: h });
  device.queue.submit([encoder.finish()]);
  texture.destroy();
  submittedFrames++;
  armCompletionTracker();
  updateReadouts(rs, deepMode);
}
function queue() { if (!queued) { queued = true; requestAnimationFrame(render); } }
function pointer(e: PointerEvent) {
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
  panX = e.clientX; panY = e.clientY; queue();
});
canvas.addEventListener('pointerdown', e => {
  pointer(e);
  canvas.setPointerCapture(e.pointerId);
  if (e.button === 1) {
    panning = true; panPointer = e.pointerId; panX = e.clientX; panY = e.clientY; direction = 0;
  } else direction = e.button === 2 ? -1 : 1;
  e.preventDefault();
});
function endPointer(e: PointerEvent) {
  if (e.pointerId === panPointer) { panning = false; panPointer = -1; }
  direction = 0;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('auxclick', e => e.preventDefault());
canvas.addEventListener('wheel', e => {
  const r = canvas.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  const a = r.width / r.height, ax = centerX + (x - .5) * a * scale, ay = centerY + (y - .5) * scale;
  const k = Math.exp(e.deltaY * .0012);
  scale *= k; centerX = ax + (centerX - ax) * k; centerY = ay + (centerY - ay) * k;
  queue(); e.preventDefault();
}, { passive: false });
for (const el of [speed, iterations, resolution, palette]) el.addEventListener('input', queue);
new ResizeObserver(queue).observe(canvas);

let last = performance.now();
function tick(now: number) {
  const dt = Math.min(.05, (now - last) / 1000);
  last = now;
  if (now - fpsLast >= 500) {
    fpsValue = completedFrames * 1000 / (now - fpsLast);
    completedFrames = 0;
    fpsLast = now;
    const deepMode = 3 / scale >= DOUBLE_FLOAT_THRESHOLD;
    updateReadouts(Number(resolution.value), deepMode);
  }
  if (direction) {
    const a = canvas.clientWidth / canvas.clientHeight;
    const ax = centerX + (px - .5) * a * scale, ay = centerY + (py - .5) * scale;
    const k = Math.exp(-direction * Number(speed.value) * dt);
    scale *= k; centerX = ax + (centerX - ax) * k; centerY = ay + (centerY - ay) * k;
    queue();
  }
  requestAnimationFrame(tick);
}
queue();
requestAnimationFrame(tick);
