import { APP_NAME, BUILD_LABEL } from './app/build';

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function setOutput(id: string, value: string): void {
  const output = document.querySelector<HTMLOutputElement>(`#${id}`);
  if (output) output.value = value;
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.append(textarea);
  textarea.select();
  document.execCommand('copy');
  textarea.remove();
}

try {
  await import('./app/main');
} catch (error) {
  const message = errorMessage(error);
  console.error(`${APP_NAME} startup failed`, error);

  const report = [
    `${APP_NAME} ${BUILD_LABEL}`,
    `Captured: ${new Date().toISOString()}`,
    `URL: ${location.href}`,
    `Startup error: ${message}`,
    `User agent: ${navigator.userAgent}`
  ].join('\n');

  const status = document.querySelector<HTMLElement>('#status');
  if (status) status.textContent = `Renderer startup failed: ${message}`;
  setOutput('state-out', 'startup failed');
  setOutput('precision-out', 'shader/device unavailable');
  setOutput('field-out', message);
  setOutput('jobs-out', 'not started');
  setOutput('timing-out', 'not started');
  setOutput('display-out', 'UI only');
  setOutput('render-size-out', 'not allocated');
  setOutput('navigation-out', 'not started');
  setOutput('gpu-out', 'initialisation failed');
  setOutput('build-out', BUILD_LABEL);

  const button = document.querySelector<HTMLButtonElement>('#copy-diagnostics');
  const feedback = document.querySelector<HTMLElement>('#copy-feedback');
  button?.addEventListener('click', () => {
    void copyText(report)
      .then(() => { if (feedback) feedback.textContent = 'Startup diagnostics copied'; })
      .catch(copyError => {
        console.error(copyError);
        if (feedback) feedback.textContent = 'Copy failed';
      });
  });
}
