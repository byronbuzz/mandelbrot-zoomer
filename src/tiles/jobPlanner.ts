import type { FieldJob, NavigationProfile, RenderRequest } from './types';

const TILE_SIZE = 512;

function tileJobs(
  request: RenderRequest,
  width: number,
  height: number,
  blockSize: 1 | 2 | 4,
  iterations: number,
  acceptIterationCap: boolean,
  phase: FieldJob['phase'],
  priorityBase: number
): FieldJob[] {
  const jobs: FieldJob[] = [];
  const focusX = request.focusX * width;
  const focusY = request.focusY * height;
  const diagonal = Math.max(1, Math.hypot(width, height));
  for (let y = 0; y < height; y += TILE_SIZE) {
    const tileHeight = Math.min(TILE_SIZE, height - y);
    for (let x = 0; x < width; x += TILE_SIZE) {
      const tileWidth = Math.min(TILE_SIZE, width - x);
      const centerX = x + tileWidth * 0.5;
      const centerY = y + tileHeight * 0.5;
      const distance = Math.hypot(centerX - focusX, centerY - focusY) / diagonal;
      jobs.push({
        requestId: request.requestId,
        x,
        y,
        width: tileWidth,
        height: tileHeight,
        blockSize,
        iterations,
        acceptIterationCap,
        phase,
        priority: priorityBase + distance
      });
    }
  }
  return jobs;
}

export function planFieldJobs(
  request: RenderRequest,
  width: number,
  height: number,
  navigation: NavigationProfile
): FieldJob[] {
  const target = Math.max(1, Math.floor(request.targetIterations));
  if (request.interaction === 'moving') {
    return [{
      requestId: request.requestId,
      x: 0,
      y: 0,
      width,
      height,
      blockSize: navigation.blockSize,
      iterations: navigation.iterations,
      acceptIterationCap: false,
      phase: 'coarse',
      priority: 0
    }];
  }

  const jobs: FieldJob[] = [];
  jobs.push({
    requestId: request.requestId,
    x: 0,
    y: 0,
    width,
    height,
    blockSize: 8,
    iterations: Math.min(target, request.interaction === 'settling' ? 128 : 96),
    acceptIterationCap: false,
    phase: 'coarse',
    priority: 0
  });

  jobs.push(...tileJobs(
    request,
    width,
    height,
    4,
    Math.min(target, request.interaction === 'settling' ? 256 : 160),
    false,
    'medium',
    10
  ));

  jobs.push(...tileJobs(
    request,
    width,
    height,
    2,
    Math.min(target, request.interaction === 'settling' ? 384 : 256),
    false,
    'fine',
    20
  ));

  if (request.interaction === 'settled') {
    jobs.push(...tileJobs(request, width, height, 1, target, true, 'fine', 30));
  }

  return jobs.sort((left, right) => left.priority - right.priority);
}
