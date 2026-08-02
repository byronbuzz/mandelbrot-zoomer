export type UiElements = Readonly<{
  canvas: HTMLCanvasElement;
  status: HTMLElement;
  speed: HTMLInputElement;
  iterations: HTMLInputElement;
  palette: HTMLInputElement;
  zoomOut: HTMLOutputElement;
  precisionOut: HTMLOutputElement;
  orbitOut: HTMLOutputElement;
  healthOut: HTMLOutputElement;
  bitsOut: HTMLOutputElement;
  exponentOut: HTMLOutputElement;
  stateOut: HTMLOutputElement;
  fpsOut: HTMLOutputElement;
  qualityOut: HTMLOutputElement;
  speedOut: HTMLOutputElement;
  iterOut: HTMLOutputElement;
  palOut: HTMLOutputElement;
}>;

const ITERATION_MIN = 50;
const ITERATION_MAX = 100_000;
const ITERATION_RATIO = ITERATION_MAX / ITERATION_MIN;
const INITIAL_ITERATION_SLIDER = Math.log(500 / ITERATION_MIN) / Math.log(ITERATION_RATIO);

function required<T extends Element>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
}

export function createUi(app: HTMLElement): UiElements {
  app.innerHTML = `
<section class="shell">
  <canvas id="fractal"></canvas>
  <header><strong>Mandelbrot Zoomer</strong><span id="status">Initialising WebGPU…</span></header>
  <aside>
    <h2>Render</h2>
    <label>Zoom speed <output id="speedOut">1.00×/s</output><input id="speed" type="range" min="0.25" max="8" step="0.25" value="1"></label>
    <label>Iterations <output id="iterOut">500</output><input id="iterations" type="range" min="0" max="1" step="0.001" value="${INITIAL_ITERATION_SLIDER.toFixed(3)}"></label>
    <label>Palette phase <output id="palOut">0.00</output><input id="palette" type="range" min="0" max="1" step="0.005" value="0"></label>
    <label>Zoom depth <output id="zoomOut">10^0.00</output></label>
    <label>Precision <output id="precisionOut">f32</output></label>
    <label>Reference orbit <output id="orbitOut">inactive</output></label>
    <label>Perturbation health <output id="healthOut">inactive</output></label>
    <label>Coordinate bits <output id="bitsOut">160</output></label>
    <label>Scale exponent <output id="exponentOut">2</output></label>
    <label>Render state <output id="stateOut">full-quality</output></label>
    <label>Last completed render rate <output id="fpsOut">0.0 FPS</output></label>
    <label>Effective quality <output id="qualityOut">500 / 500 iter · 100%</output></label>
    <p><b>Numerical core V4.5:</b> four-limb GPU reference orbits, cancellation-safe reconstruction and survivor-preserving reference search.</p>
  </aside>
</section>`;
  return {
    canvas: required<HTMLCanvasElement>(app, '#fractal'),
    status: required<HTMLElement>(app, '#status'),
    speed: required<HTMLInputElement>(app, '#speed'),
    iterations: required<HTMLInputElement>(app, '#iterations'),
    palette: required<HTMLInputElement>(app, '#palette'),
    zoomOut: required<HTMLOutputElement>(app, '#zoomOut'),
    precisionOut: required<HTMLOutputElement>(app, '#precisionOut'),
    orbitOut: required<HTMLOutputElement>(app, '#orbitOut'),
    healthOut: required<HTMLOutputElement>(app, '#healthOut'),
    bitsOut: required<HTMLOutputElement>(app, '#bitsOut'),
    exponentOut: required<HTMLOutputElement>(app, '#exponentOut'),
    stateOut: required<HTMLOutputElement>(app, '#stateOut'),
    fpsOut: required<HTMLOutputElement>(app, '#fpsOut'),
    qualityOut: required<HTMLOutputElement>(app, '#qualityOut'),
    speedOut: required<HTMLOutputElement>(app, '#speedOut'),
    iterOut: required<HTMLOutputElement>(app, '#iterOut'),
    palOut: required<HTMLOutputElement>(app, '#palOut')
  };
}

export function iterationCount(sliderValue: string): number {
  return Math.round(ITERATION_MIN * Math.pow(ITERATION_RATIO, Number(sliderValue)));
}
