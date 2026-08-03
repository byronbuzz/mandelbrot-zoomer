export type PresentationView = Readonly<{
  centerX: number;
  centerY: number;
  height: number;
  aspect: number;
}>;

export type ReprojectionTransform = Readonly<{
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}>;

export type ReprojectionAdmission = Readonly<{
  accepted: boolean;
  packed: ReprojectionTransform;
  worstSourceTexelError: number;
}>;

const SOURCE_TEXEL_ERROR_LIMIT = 0.01;

export function reprojectionTransform(
  source: PresentationView,
  target: PresentationView
): ReprojectionTransform {
  return {
    scaleX: (target.height * target.aspect) / (source.height * source.aspect),
    scaleY: target.height / source.height,
    offsetX: (target.centerX - source.centerX) / (source.height * source.aspect),
    offsetY: (target.centerY - source.centerY) / source.height
  };
}

export function sourceUv(
  transform: ReprojectionTransform,
  u: number,
  v: number
): readonly [number, number] {
  return [
    0.5 + transform.offsetX + (u - 0.5) * transform.scaleX,
    0.5 + transform.offsetY + (v - 0.5) * transform.scaleY
  ];
}

export function admitReprojection(
  transform: ReprojectionTransform,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number
): ReprojectionAdmission {
  const packed = {
    scaleX: Math.fround(transform.scaleX),
    scaleY: Math.fround(transform.scaleY),
    offsetX: Math.fround(transform.offsetX),
    offsetY: Math.fround(transform.offsetY)
  };
  const samples = [
    [0.5, 0.5],
    [0.5 / targetWidth, 0.5 / targetHeight],
    [(targetWidth - 0.5) / targetWidth, 0.5 / targetHeight],
    [0.5 / targetWidth, (targetHeight - 0.5) / targetHeight],
    [(targetWidth - 0.5) / targetWidth, (targetHeight - 0.5) / targetHeight]
  ] as const;
  let worstSourceTexelError = 0;
  for (const [u, v] of samples) {
    const expected = sourceUv(transform, u, v);
    const actual = sourceUv(packed, u, v);
    worstSourceTexelError = Math.max(
      worstSourceTexelError,
      Math.abs(expected[0] - actual[0]) * sourceWidth,
      Math.abs(expected[1] - actual[1]) * sourceHeight
    );
  }
  const finite = Object.values(packed).every(Number.isFinite);
  return {
    accepted: finite && worstSourceTexelError <= SOURCE_TEXEL_ERROR_LIMIT,
    packed,
    worstSourceTexelError
  };
}

export function sameView(left: PresentationView, right: PresentationView): boolean {
  return left.centerX === right.centerX
    && left.centerY === right.centerY
    && left.height === right.height
    && left.aspect === right.aspect;
}
