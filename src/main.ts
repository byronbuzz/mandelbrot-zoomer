import './style.css';

const app = document.querySelector<HTMLElement>('#app');
if (!app) throw new Error('Missing app root');

app.innerHTML = `
<section class="shell">
  <canvas id="fractal"></canvas>
  <header><strong>Mandelbrot Zoomer</strong><span id="status">Initialising WebGPU…</span></header>
  <aside>
    <h2>Render</h2>
    <label>Iterations <output id="iterOut">500</output><input id="iterations" type="range" min="50" max="3000" step="50" value="500"></label>
    <label>Internal resolution <output id="resOut">100%</output><input id="resolution" type="range" min="0.35" max="1" step="0.05" value="1"></label>
    <label>Palette phase <output id="palOut">0.00</output><input id="palette" type="range" min="0" max="1" step="0.005" value="0"></label>
    <p><b>XaoS-style navigation:</b> hold left mouse to zoom toward the pointer; hold right mouse to zoom out. Use the wheel for discrete zooming.</p>
  </aside>
</section>`;

const canvas = document.querySelector<HTMLCanvasElement>('#fractal')!;
const status = document.querySelector<HTMLElement>('#status')!;
const iterations = document.querySelector<HTMLInputElement>('#iterations')!;
const resolution = document.querySelector<HTMLInputElement>('#resolution')!;
const palette = document.querySelector<HTMLInputElement>('#palette')!;
const iterOut = document.querySelector<HTMLOutputElement>('#iterOut')!;
const resOut = document.querySelector<HTMLOutputElement>('#resOut')!;
const palOut = document.querySelector<HTMLOutputElement>('#palOut')!;

const shader = `
struct Params { center: vec2f, scale: f32, aspect: f32, iterations: u32, phase: f32, width: u32, height: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;
fn palette(t:f32)->vec3f { return .5 + .5*cos(6.28318*(vec3f(t)+vec3f(0.0,.12,.24)+p.phase)); }
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=p.width||id.y>=p.height){return;}
 let uv=(vec2f(id.xy)+.5)/vec2f(f32(p.width),f32(p.height));
 let c=p.center+vec2f((uv.x-.5)*p.aspect,uv.y-.5)*p.scale;
 let q=(c.x-.25)*(c.x-.25)+c.y*c.y;
 if(q*(q+c.x-.25)<=.25*c.y*c.y||(c.x+1.)*(c.x+1.)+c.y*c.y<=.0625){textureStore(outTex,vec2i(id.xy),vec4f(0,0,0,1));return;}
 var z=vec2f(0.); var i=0u;
 loop{if(i>=p.iterations||dot(z,z)>256.){break;} z=vec2f(z.x*z.x-z.y*z.y,2.*z.x*z.y)+c;i++;}
 if(i>=p.iterations){textureStore(outTex,vec2i(id.xy),vec4f(0,0,0,1));return;}
 let s=f32(i)+1.-log2(log2(length(z)));
 textureStore(outTex,vec2i(id.xy),vec4f(palette(fract(.018*s)),1));
}`;

if (!navigator.gpu) throw new Error('WebGPU is unavailable. Use a current Chromium browser with hardware acceleration enabled.');
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('No WebGPU adapter found');
const features: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
const device = await adapter.requestDevice({ requiredFeatures: features });
const context = canvas.getContext('webgpu');
if (!context) throw new Error('Unable to create WebGPU canvas context');
const gpuContext: GPUCanvasContext = context;
const format = navigator.gpu.getPreferredCanvasFormat();
gpuContext.configure({ device, format, alphaMode: 'opaque' });
const module = device.createShaderModule({ code: shader });
const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const info = adapter.info;
status.textContent = `${info.vendor || 'GPU'} · WebGPU${device.features.has('timestamp-query') ? ' · GPU timing' : ''}`;

let centerX = -0.5, centerY = 0, scale = 3, direction = 0, px = .5, py = .5, queued = false;
function render(){
 queued=false;
 const rs=Number(resolution.value);
 const w=Math.max(1,Math.floor(canvas.clientWidth*devicePixelRatio*rs));
 const h=Math.max(1,Math.floor(canvas.clientHeight*devicePixelRatio*rs));
 canvas.width=w; canvas.height=h;
 const texture=device.createTexture({size:[w,h],format:'rgba8unorm',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});
 const data=new ArrayBuffer(32), f=new Float32Array(data), u=new Uint32Array(data);
 f[0]=centerX;f[1]=centerY;f[2]=scale;f[3]=w/h;u[4]=Number(iterations.value);f[5]=Number(palette.value);u[6]=w;u[7]=h;
 device.queue.writeBuffer(params,0,data);
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:texture.createView()}]});
 const encoder=device.createCommandEncoder(), pass=encoder.beginComputePass();
 pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(w/8),Math.ceil(h/8));pass.end();
 encoder.copyTextureToTexture({texture},{texture:gpuContext.getCurrentTexture()},{width:w,height:h});
 device.queue.submit([encoder.finish()]);texture.destroy();
 iterOut.value=iterations.value;resOut.value=`${Math.round(rs*100)}%`;palOut.value=Number(palette.value).toFixed(2);
}
function queue(){if(!queued){queued=true;requestAnimationFrame(render);}}
function pointer(e:PointerEvent){const r=canvas.getBoundingClientRect();px=(e.clientX-r.left)/r.width;py=(e.clientY-r.top)/r.height;}
canvas.addEventListener('pointermove',pointer);
canvas.addEventListener('pointerdown',e=>{pointer(e);direction=e.button===2?-1:1;canvas.setPointerCapture(e.pointerId);e.preventDefault();});
canvas.addEventListener('pointerup',e=>{direction=0;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height,a=r.width/r.height,ax=centerX+(x-.5)*a*scale,ay=centerY+(y-.5)*scale,k=Math.exp(e.deltaY*.0012);scale*=k;centerX=ax+(centerX-ax)*k;centerY=ay+(centerY-ay)*k;queue();e.preventDefault();},{passive:false});
for(const el of [iterations,resolution,palette])el.addEventListener('input',queue);
new ResizeObserver(queue).observe(canvas);
let last=performance.now();
function tick(now:number){const dt=Math.min(.05,(now-last)/1000);last=now;if(direction){const a=canvas.clientWidth/canvas.clientHeight,ax=centerX+(px-.5)*a*scale,ay=centerY+(py-.5)*scale,k=Math.exp(-direction*2.5*dt);scale*=k;centerX=ax+(centerX-ax)*k;centerY=ay+(centerY-ay)*k;queue();}requestAnimationFrame(tick);}
queue();requestAnimationFrame(tick);
