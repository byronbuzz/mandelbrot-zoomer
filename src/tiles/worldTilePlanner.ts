import {
  fixed,
  fixedAddScaled,
  type BigFixed
} from '../bigFixed';
import {
  scaleDeltaParts,
  scaleLog2
} from '../binaryScale';
import type { CameraSnapshot } from '../camera/types';
import {
  PERSISTENT_TILE_BITS,
  PERSISTENT_TILE_SIZE,
  type PersistentTileDescriptor,
  type PersistentTileKey
} from './persistentTileTypes';

const TILE_MARGIN = 1;

function floorDivPowerOfTwo(value: BigFixed, exponent: number): bigint {
  const shift = value.bits + exponent;
  return shift >= 0
    ? value.raw >> BigInt(shift)
    : value.raw << BigInt(-shift);
}

function dyadicFixed(integer: bigint, exponent: number, bits: number): BigFixed {
  const shift = bits + exponent;
  return shift >= 0
    ? fixed(integer << BigInt(shift), bits)
    : fixed(integer >> BigInt(-shift), bits);
}

function tileKey(sampleExponent: number, tileX: bigint, tileY: bigint): PersistentTileKey {
  return `${sampleExponent}:${tileX.toString()}:${tileY.toString()}`;
}

export function sampleExponentForViewport(
  camera: CameraSnapshot,
  renderHeight: number,
  levelOffset = 0
): number {
  const pixelExponent = scaleLog2(camera.scale) - Math.log2(Math.max(1, renderHeight));
  return Math.floor(pixelExponent) + levelOffset;
}

export function tileSpanExponent(sampleExponent: number): number {
  return sampleExponent + PERSISTENT_TILE_BITS;
}

export function descriptorForIndex(
  camera: CameraSnapshot,
  sampleExponent: number,
  tileX: bigint,
  tileY: bigint,
  focusWorldX: BigFixed,
  focusWorldY: BigFixed
): PersistentTileDescriptor {
  const halfTile = BigInt(PERSISTENT_TILE_SIZE / 2);
  const centerSampleX = tileX * BigInt(PERSISTENT_TILE_SIZE) + halfTile;
  const centerSampleY = tileY * BigInt(PERSISTENT_TILE_SIZE) + halfTile;
  const centerX = dyadicFixed(centerSampleX, sampleExponent, camera.centerX.bits);
  const centerY = dyadicFixed(centerSampleY, sampleExponent, camera.centerY.bits);
  const spanExponent = tileSpanExponent(sampleExponent);
  const span = Math.pow(2, spanExponent);
  const dx = Number(centerX.raw - focusWorldX.raw) * Math.pow(2, -centerX.bits) / span;
  const dy = Number(centerY.raw - focusWorldY.raw) * Math.pow(2, -centerY.bits) / span;
  return {
    key: tileKey(sampleExponent, tileX, tileY),
    sampleExponent,
    tileX,
    tileY,
    centerX,
    centerY,
    distanceFromFocus: Number.isFinite(dx) && Number.isFinite(dy)
      ? Math.hypot(dx, dy)
      : Number.POSITIVE_INFINITY
  };
}

export function visibleTileDescriptors(
  camera: CameraSnapshot,
  aspect: number,
  renderHeight: number,
  focusX: number,
  focusY: number,
  levelOffset = 0
): PersistentTileDescriptor[] {
  const sampleExponent = sampleExponentForViewport(camera, renderHeight, levelOffset);
  const spanExponent = tileSpanExponent(sampleExponent);
  const halfHeight = scaleDeltaParts(camera.scale, 0.5);
  const halfWidth = scaleDeltaParts(camera.scale, 0.5 * aspect);
  const minX = fixedAddScaled(camera.centerX, -halfWidth.mantissa, halfWidth.exponent);
  const maxX = fixedAddScaled(camera.centerX, halfWidth.mantissa, halfWidth.exponent);
  const minY = fixedAddScaled(camera.centerY, -halfHeight.mantissa, halfHeight.exponent);
  const maxY = fixedAddScaled(camera.centerY, halfHeight.mantissa, halfHeight.exponent);
  const focusDx = scaleDeltaParts(camera.scale, (focusX - 0.5) * aspect);
  const focusDy = scaleDeltaParts(camera.scale, focusY - 0.5);
  const focusWorldX = fixedAddScaled(camera.centerX, focusDx.mantissa, focusDx.exponent);
  const focusWorldY = fixedAddScaled(camera.centerY, focusDy.mantissa, focusDy.exponent);

  const firstX = floorDivPowerOfTwo(minX, spanExponent) - BigInt(TILE_MARGIN);
  const lastX = floorDivPowerOfTwo(maxX, spanExponent) + BigInt(TILE_MARGIN);
  const firstY = floorDivPowerOfTwo(minY, spanExponent) - BigInt(TILE_MARGIN);
  const lastY = floorDivPowerOfTwo(maxY, spanExponent) + BigInt(TILE_MARGIN);
  const descriptors: PersistentTileDescriptor[] = [];

  for (let tileY = firstY; tileY <= lastY; tileY++) {
    for (let tileX = firstX; tileX <= lastX; tileX++) {
      descriptors.push(descriptorForIndex(
        camera,
        sampleExponent,
        tileX,
        tileY,
        focusWorldX,
        focusWorldY
      ));
    }
  }

  return descriptors.sort((left, right) => left.distanceFromFocus - right.distanceFromFocus);
}
