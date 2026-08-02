import { APP_NAME, BUILD_LABEL } from './build';

export type AppUi = Readonly<{
  canvas: HTMLCanvasElement;
  panel: HTMLElement;
  panelToggle: HTMLButtonElement;
  status: HTMLElement;
  speed: HTMLInputElement;
  iterations: HTMLInputElement;
  palette: HTMLInputElement;
  resetLocation: HTMLButtonElement;
  copyDiagnostics: HTMLButtonElement;
  zoomOut: HTMLOutputElement;
  stateOut: HTMLOutputElement;
  precisionOut: HTMLOutputElement;
  fieldOut: HTMLOutputElement;
  jobsOut: HTMLOutputElement;
  timingOut: HTMLOutputElement;
  displayOut: HTMLOutputElement;
  renderSizeOut: HTMLOutputElement;
  navigationOut: HTMLOutputElement;
  gpuOut: HTMLOutputElement;
  buildOut: HTMLOutputElement;
  copyFeedback: HTMLElement;
}>;

function outputRow(label: string, id: string): string {
  return `<div class="stat-row"><span>${label}</span><output id="${id}">—</output></div>`;
}

export function createUi(root: HTMLElement): AppUi {
  root.innerHTML = `
    <div class="app-shell">
      <canvas id="fractal-canvas" tabindex="0" aria-label="Interactive Mandelbrot fractal"></canvas>
      <header class="top-bar">
        <div class="brand"><strong>${APP_NAME}</strong><span>${BUILD_LABEL}</span></div>
        <button id="panel-toggle" class="icon-button" type="button" aria-label="Toggle controls">Controls</button>
      </header>
      <aside id="control-panel" class="control-panel" aria-label="Fractal controls">
        <div class="panel-header">
          <div><strong>Controls</strong><small>Persistent field renderer</small></div>
          <button id="panel-close" class="close-button" type="button" aria-label="Close controls">×</button>
        </div>
        <section class="control-section">
          <label>Zoom speed <input id="speed" type="range" min="0.25" max="5" step="0.05" value="1.5"></label>
          <label>Iteration target <input id="iterations" type="range" min="100" max="12000" step="100" value="500"></label>
          <label>Palette phase <input id="palette" type="range" min="0" max="1" step="0.01" value="0.08"></label>
          <div class="button-row">
            <button id="reset-location" type="button">Reset location</button>
            <button id="copy-diagnostics" type="button">Copy diagnostics</button>
          </div>
          <p id="copy-feedback" class="feedback" aria-live="polite"></p>
        </section>
        <section class="stats-section">
          ${outputRow('Zoom depth', 'zoom-out')}
          ${outputRow('Interaction state', 'state-out')}
          ${outputRow('Precision', 'precision-out')}
          ${outputRow('Field', 'field-out')}
          ${outputRow('Jobs', 'jobs-out')}
          ${outputRow('Compute batch', 'timing-out')}
          ${outputRow('Presentation', 'display-out')}
          ${outputRow('Render size', 'render-size-out')}
          ${outputRow('Navigation budget', 'navigation-out')}
          ${outputRow('GPU', 'gpu-out')}
          ${outputRow('Build', 'build-out')}
        </section>
        <p class="help-copy">Hold or tap the canvas to zoom in. Right-click to zoom out. Middle-drag to pan. The displayed field is continuously reprojected while new samples replace it.</p>
      </aside>
      <div id="status" class="status-line" aria-live="polite"></div>
    </div>
  `;

  const required = <T extends Element>(selector: string): T => {
    const element = root.querySelector<T>(selector);
    if (!element) throw new Error(`Missing UI element: ${selector}`);
    return element;
  };

  const panel = required<HTMLElement>('#control-panel');
  const panelToggle = required<HTMLButtonElement>('#panel-toggle');
  const setPanelVisible = (visible: boolean): void => {
    panel.classList.toggle('is-hidden', !visible);
    panelToggle.setAttribute('aria-expanded', String(visible));
  };
  panelToggle.addEventListener('click', () => setPanelVisible(panel.classList.contains('is-hidden')));
  required<HTMLButtonElement>('#panel-close').addEventListener('click', () => setPanelVisible(false));

  return {
    canvas: required<HTMLCanvasElement>('#fractal-canvas'),
    panel,
    panelToggle,
    status: required<HTMLElement>('#status'),
    speed: required<HTMLInputElement>('#speed'),
    iterations: required<HTMLInputElement>('#iterations'),
    palette: required<HTMLInputElement>('#palette'),
    resetLocation: required<HTMLButtonElement>('#reset-location'),
    copyDiagnostics: required<HTMLButtonElement>('#copy-diagnostics'),
    zoomOut: required<HTMLOutputElement>('#zoom-out'),
    stateOut: required<HTMLOutputElement>('#state-out'),
    precisionOut: required<HTMLOutputElement>('#precision-out'),
    fieldOut: required<HTMLOutputElement>('#field-out'),
    jobsOut: required<HTMLOutputElement>('#jobs-out'),
    timingOut: required<HTMLOutputElement>('#timing-out'),
    displayOut: required<HTMLOutputElement>('#display-out'),
    renderSizeOut: required<HTMLOutputElement>('#render-size-out'),
    navigationOut: required<HTMLOutputElement>('#navigation-out'),
    gpuOut: required<HTMLOutputElement>('#gpu-out'),
    buildOut: required<HTMLOutputElement>('#build-out'),
    copyFeedback: required<HTMLElement>('#copy-feedback')
  };
}
