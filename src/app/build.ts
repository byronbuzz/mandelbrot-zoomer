declare const __BUILD_SHA__: string;

export const APP_NAME = 'WebGPU Fractal Zoomer';
export const APP_VERSION = '1.3.0';
export const BUILD_SHA = typeof __BUILD_SHA__ === 'string' ? __BUILD_SHA__ : 'dev';
export const BUILD_LABEL = `${APP_VERSION} · ${BUILD_SHA}`;
