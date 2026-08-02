import { BUILD_LABEL } from './build';

export type UiTab = 'main' | 'colours' | 'stats' | 'help';

export type UiElements = Readonly<{
  shell: HTMLElement;
  canvas: HTMLCanvasElement;
  status: HTMLElement;
  panel: HTMLElement;
  restorePanel: HTMLButtonElement;
  closePanel: HTMLButtonElement;
  copyDiagnostics: HTMLButtonElement;
  copyFeedback: HTMLElement;
  resetControls: HTMLButtonElement;
  resetLocation: HTMLButtonElement;
  speed: HTMLInputElement;
  iterations: HTMLInputElement;
  palette: HTMLInputElement;
  zoomOut: HTMLOutputElement;
  precisionOut: HTMLOutputElement;
  orbitOut: HTMLOutputElement;
  healthOut: HTMLOutputElement;
  failureOut: HTMLOutputElement;
  maxExponentOut: HTMLOutputElement;
  bitsOut: HTMLOutputElement;
  exponentOut: HTMLOutputElement;
  stateOut: HTMLOutputElement;
  displayFpsOut: HTMLOutputElement;
  fpsOut: HTMLOutputElement;
  qualityOut: HTMLOutputElement;
  timingOut: HTMLOutputElement;
  batchesOut: HTMLOutputElement;
  anchorMoveOut: HTMLOutputElement;
  referenceTimeOut: HTMLOutputElement;
  adapterOut: HTMLOutputElement;
  buildOut: HTMLOutputElement;
  speedOut: HTMLOutputElement;
  iterOut: HTMLOutputElement;
  palOut: HTMLOutputElement;
  hudZoomOut: HTMLOutputElement;
  hudFpsOut: HTMLOutputElement;
  panelVisible: () => boolean;
  selectedTab: () => UiTab;
  setPanelVisible: (visible: boolean) => void;
  togglePanel: () => void;
  showTab: (tab: UiTab) => void;
  focusSelectedTab: () => void;
  setStatsWarning: (active: boolean) => void;
}>;

const ITERATION_MIN = 50;
const ITERATION_MAX = 100_000;
const ITERATION_RATIO = ITERATION_MAX / ITERATION_MIN;
export const INITIAL_ITERATION_SLIDER = Math.log(500 / ITERATION_MIN) / Math.log(ITERATION_RATIO);
export const DEFAULT_SPEED = '1';
export const DEFAULT_PALETTE_PHASE = '0';

const PANEL_VISIBLE_KEY = 'mandelbrot-zoomer.panel-visible';
const PANEL_TAB_KEY = 'mandelbrot-zoomer.panel-tab';
const UI_TABS: readonly UiTab[] = ['main', 'colours', 'stats', 'help'];

function required<T extends Element>(root: ParentNode, selector: string): T {
  const element = root.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
}

function storedValue(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storeValue(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable in private or embedded browsing contexts.
  }
}

function isUiTab(value: string | null): value is UiTab {
  return value !== null && UI_TABS.includes(value as UiTab);
}

export function createUi(app: HTMLElement): UiElements {
  app.innerHTML = `
<section class="shell" id="shell">
  <canvas id="fractal" tabindex="0" aria-label="Interactive Mandelbrot fractal"></canvas>
  <header><strong>Mandelbrot Zoomer</strong><span id="status">Initialising WebGPU…</span></header>

  <div class="minimal-hud" id="minimalHud" hidden>
    <output id="hudZoomOut">10^0.00</output>
    <output id="hudFpsOut">0 Hz · 0.0 FPS</output>
  </div>
  <button class="panel-restore" id="restorePanel" type="button" hidden aria-label="Show controls">Controls</button>

  <aside class="control-panel" id="controlPanel" aria-label="Mandelbrot controls">
    <div class="panel-heading">
      <h2>Controls</h2>
      <button class="icon-button" id="closePanel" type="button" aria-label="Hide controls">×</button>
    </div>
    <nav class="panel-tabs" role="tablist" aria-label="Control sections">
      <button type="button" role="tab" data-tab="main" aria-controls="panel-main">Main</button>
      <button type="button" role="tab" data-tab="colours" aria-controls="panel-colours">Colours</button>
      <button type="button" role="tab" data-tab="stats" aria-controls="panel-stats">Stats <span class="warning-badge" id="statsBadge" hidden>!</span></button>
      <button type="button" role="tab" data-tab="help" aria-controls="panel-help">Help</button>
    </nav>

    <section class="tab-panel" id="panel-main" data-panel="main" role="tabpanel">
      <label class="control-row">Zoom speed <output id="speedOut">1.00×/s</output><input id="speed" type="range" min="0.25" max="8" step="0.25" value="${DEFAULT_SPEED}"></label>
      <label class="control-row">Iterations <output id="iterOut">500</output><input id="iterations" type="range" min="0" max="1" step="0.001" value="${INITIAL_ITERATION_SLIDER.toFixed(3)}"></label>
      <div class="stat-row primary-stat"><span>Zoom depth</span><output id="zoomOut">10^0.00</output></div>
      <div class="button-row">
        <button id="resetControls" type="button">Reset controls</button>
        <button id="resetLocation" type="button">Reset location</button>
      </div>
    </section>

    <section class="tab-panel" id="panel-colours" data-panel="colours" role="tabpanel" hidden>
      <label class="control-row">Palette phase <output id="palOut">0.00</output><input id="palette" type="range" min="0" max="1" step="0.005" value="${DEFAULT_PALETTE_PHASE}"></label>
      <p class="panel-note">Palette presets, logarithmic palette length, colour cycling and depth-driven colour evolution will build on this tab.</p>
    </section>

    <section class="tab-panel stats-panel" id="panel-stats" data-panel="stats" role="tabpanel" hidden>
      <div class="stat-row"><span>Precision</span><output id="precisionOut">f32</output></div>
      <div class="stat-row multi-line"><span>Reference orbit</span><output id="orbitOut">inactive</output></div>
      <div class="stat-row multi-line"><span>Perturbation health</span><output id="healthOut">inactive</output></div>
      <div class="stat-row multi-line"><span>Failure causes</span><output id="failureOut">inactive</output></div>
      <div class="stat-row"><span>Maximum perturbation</span><output id="maxExponentOut">inactive</output></div>
      <div class="stat-row"><span>Coordinate bits</span><output id="bitsOut">160</output></div>
      <div class="stat-row"><span>Scale exponent</span><output id="exponentOut">2</output></div>
      <div class="stat-row"><span>Render state</span><output id="stateOut">full-quality</output></div>
      <div class="stat-row"><span>Display presentation rate</span><output id="displayFpsOut">0 Hz</output></div>
      <div class="stat-row"><span>Anchor-frame rate</span><output id="fpsOut">0.0 FPS</output></div>
      <div class="stat-row"><span>Effective quality</span><output id="qualityOut">500 / 500 iter · 100%</output></div>
      <div class="stat-row"><span>Compute + present</span><output id="timingOut">0.0 + 0.0 ms</output></div>
      <div class="stat-row"><span>GPU batches</span><output id="batchesOut">0</output></div>
      <div class="stat-row"><span>Reference-anchor move</span><output id="anchorMoveOut">0.0 px</output></div>
      <div class="stat-row"><span>Reference generation</span><output id="referenceTimeOut">inactive</output></div>
      <div class="stat-row multi-line"><span>GPU</span><output id="adapterOut">initialising…</output></div>
      <div class="stat-row"><span>Build</span><output id="buildOut">${BUILD_LABEL}</output></div>
      <div class="button-row diagnostics-actions">
        <button id="copyDiagnostics" type="button">Copy diagnostics</button>
        <span id="copyFeedback" role="status" aria-live="polite"></span>
      </div>
    </section>

    <section class="tab-panel help-panel" id="panel-help" data-panel="help" role="tabpanel" hidden>
      <dl class="help-list">
        <div><dt>Left mouse</dt><dd>Hold to zoom in around the pointer.</dd></div>
        <div><dt>Right mouse</dt><dd>Hold to zoom out around the pointer.</dd></div>
        <div><dt>Middle drag</dt><dd>Pan the viewport.</dd></div>
        <div><dt>Mouse wheel</dt><dd>Step zoom around the pointer.</dd></div>
        <div><dt>Tab</dt><dd>Hide or restore this panel when focus is on the canvas.</dd></div>
        <div><dt>?</dt><dd>Open this Help tab.</dd></div>
        <div><dt>Escape</dt><dd>Hide the panel and return focus to the fractal.</dd></div>
      </dl>
      <p class="panel-note">Touch zooming and panning will arrive with the V5 fluid-zoom work.</p>
    </section>
  </aside>
</section>`;

  const shell = required<HTMLElement>(app, '#shell');
  const canvas = required<HTMLCanvasElement>(app, '#fractal');
  const panel = required<HTMLElement>(app, '#controlPanel');
  const restorePanel = required<HTMLButtonElement>(app, '#restorePanel');
  const closePanel = required<HTMLButtonElement>(app, '#closePanel');
  const minimalHud = required<HTMLElement>(app, '#minimalHud');
  const statsBadge = required<HTMLElement>(app, '#statsBadge');
  const tabButtons = Object.fromEntries(UI_TABS.map(tab => [
    tab,
    required<HTMLButtonElement>(panel, `[data-tab="${tab}"]`)
  ])) as Record<UiTab, HTMLButtonElement>;
  const tabPanels = Object.fromEntries(UI_TABS.map(tab => [
    tab,
    required<HTMLElement>(panel, `[data-panel="${tab}"]`)
  ])) as Record<UiTab, HTMLElement>;

  let currentTab: UiTab = isUiTab(storedValue(PANEL_TAB_KEY)) ? storedValue(PANEL_TAB_KEY) as UiTab : 'main';
  let visible = storedValue(PANEL_VISIBLE_KEY) !== 'false';

  const showTab = (tab: UiTab): void => {
    currentTab = tab;
    for (const candidate of UI_TABS) {
      const selected = candidate === tab;
      tabButtons[candidate].setAttribute('aria-selected', String(selected));
      tabButtons[candidate].tabIndex = selected ? 0 : -1;
      tabPanels[candidate].hidden = !selected;
    }
    storeValue(PANEL_TAB_KEY, tab);
  };

  const setPanelVisible = (nextVisible: boolean): void => {
    visible = nextVisible;
    shell.classList.toggle('panel-hidden', !visible);
    panel.setAttribute('aria-hidden', String(!visible));
    panel.toggleAttribute('inert', !visible);
    restorePanel.hidden = visible;
    minimalHud.hidden = visible;
    storeValue(PANEL_VISIBLE_KEY, String(visible));
  };

  for (const tab of UI_TABS) {
    tabButtons[tab].addEventListener('click', () => showTab(tab));
  }
  closePanel.addEventListener('click', () => {
    setPanelVisible(false);
    canvas.focus({ preventScroll: true });
  });
  restorePanel.addEventListener('click', () => {
    setPanelVisible(true);
    tabButtons[currentTab].focus({ preventScroll: true });
  });

  showTab(currentTab);
  setPanelVisible(visible);

  return {
    shell,
    canvas,
    status: required<HTMLElement>(app, '#status'),
    panel,
    restorePanel,
    closePanel,
    copyDiagnostics: required<HTMLButtonElement>(app, '#copyDiagnostics'),
    copyFeedback: required<HTMLElement>(app, '#copyFeedback'),
    resetControls: required<HTMLButtonElement>(app, '#resetControls'),
    resetLocation: required<HTMLButtonElement>(app, '#resetLocation'),
    speed: required<HTMLInputElement>(app, '#speed'),
    iterations: required<HTMLInputElement>(app, '#iterations'),
    palette: required<HTMLInputElement>(app, '#palette'),
    zoomOut: required<HTMLOutputElement>(app, '#zoomOut'),
    precisionOut: required<HTMLOutputElement>(app, '#precisionOut'),
    orbitOut: required<HTMLOutputElement>(app, '#orbitOut'),
    healthOut: required<HTMLOutputElement>(app, '#healthOut'),
    failureOut: required<HTMLOutputElement>(app, '#failureOut'),
    maxExponentOut: required<HTMLOutputElement>(app, '#maxExponentOut'),
    bitsOut: required<HTMLOutputElement>(app, '#bitsOut'),
    exponentOut: required<HTMLOutputElement>(app, '#exponentOut'),
    stateOut: required<HTMLOutputElement>(app, '#stateOut'),
    displayFpsOut: required<HTMLOutputElement>(app, '#displayFpsOut'),
    fpsOut: required<HTMLOutputElement>(app, '#fpsOut'),
    qualityOut: required<HTMLOutputElement>(app, '#qualityOut'),
    timingOut: required<HTMLOutputElement>(app, '#timingOut'),
    batchesOut: required<HTMLOutputElement>(app, '#batchesOut'),
    anchorMoveOut: required<HTMLOutputElement>(app, '#anchorMoveOut'),
    referenceTimeOut: required<HTMLOutputElement>(app, '#referenceTimeOut'),
    adapterOut: required<HTMLOutputElement>(app, '#adapterOut'),
    buildOut: required<HTMLOutputElement>(app, '#buildOut'),
    speedOut: required<HTMLOutputElement>(app, '#speedOut'),
    iterOut: required<HTMLOutputElement>(app, '#iterOut'),
    palOut: required<HTMLOutputElement>(app, '#palOut'),
    hudZoomOut: required<HTMLOutputElement>(app, '#hudZoomOut'),
    hudFpsOut: required<HTMLOutputElement>(app, '#hudFpsOut'),
    panelVisible: () => visible,
    selectedTab: () => currentTab,
    setPanelVisible,
    togglePanel: () => setPanelVisible(!visible),
    showTab,
    focusSelectedTab: () => tabButtons[currentTab].focus({ preventScroll: true }),
    setStatsWarning: active => { statsBadge.hidden = !active; }
  };
}

export function resetControlValues(ui: Pick<UiElements, 'speed' | 'iterations' | 'palette'>): void {
  ui.speed.value = DEFAULT_SPEED;
  ui.iterations.value = INITIAL_ITERATION_SLIDER.toFixed(3);
  ui.palette.value = DEFAULT_PALETTE_PHASE;
}

export function iterationCount(sliderValue: string): number {
  return Math.round(ITERATION_MIN * Math.pow(ITERATION_RATIO, Number(sliderValue)));
}
