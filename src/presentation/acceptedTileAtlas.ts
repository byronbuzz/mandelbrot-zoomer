import { PERSISTENT_TILE_SIZE } from '../tiles/persistentTileTypes';

const ATLAS_COLUMNS = 32;
const ATLAS_ROWS = 16;

export type AtlasSlot = Readonly<{ index: number; x: number; y: number; lease: number }>;

export class AcceptedTileAtlas {
  readonly colour: GPUTexture;
  readonly quality: GPUTexture;
  readonly leaseDirectory: GPUBuffer;
  readonly width = ATLAS_COLUMNS * PERSISTENT_TILE_SIZE;
  readonly height = ATLAS_ROWS * PERSISTENT_TILE_SIZE;
  private readonly free: number[] = [];
  private readonly leases = new Uint32Array(ATLAS_COLUMNS * ATLAS_ROWS);

  constructor(private readonly device: GPUDevice) {
    const usage = GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING;
    this.colour = device.createTexture({
      label: 'accepted-tile-colour-atlas',
      size: [this.width, this.height],
      format: 'rgba8unorm',
      usage
    });
    this.quality = device.createTexture({
      label: 'accepted-tile-quality-atlas',
      size: [this.width, this.height],
      format: 'rgba8unorm',
      usage
    });
    this.leaseDirectory = device.createBuffer({
      label: 'accepted-tile-atlas-leases',
      size: this.leases.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    for (let index = ATLAS_COLUMNS * ATLAS_ROWS - 1; index >= 0; index--) this.free.push(index);
  }

  get availableSlots(): number {
    return this.free.length;
  }

  allocate(): AtlasSlot {
    const index = this.free.pop();
    if (index === undefined) throw new Error('Accepted tile atlas exhausted');
    this.leases[index] = (this.leases[index] + 1) >>> 0 || 1;
    this.device.queue.writeBuffer(this.leaseDirectory, index * Uint32Array.BYTES_PER_ELEMENT, this.leases, index, 1);
    return {
      index,
      x: (index % ATLAS_COLUMNS) * PERSISTENT_TILE_SIZE,
      y: Math.floor(index / ATLAS_COLUMNS) * PERSISTENT_TILE_SIZE,
      lease: this.leases[index]
    };
  }

  release(slot: AtlasSlot): void {
    this.free.push(slot.index);
  }

  encodeCopy(encoder: GPUCommandEncoder, slot: AtlasSlot, colour: GPUTexture, quality: GPUTexture): void {
    const destination = { texture: this.colour, origin: { x: slot.x, y: slot.y } };
    encoder.copyTextureToTexture({ texture: colour }, destination, {
      width: PERSISTENT_TILE_SIZE,
      height: PERSISTENT_TILE_SIZE
    });
    encoder.copyTextureToTexture(
      { texture: quality },
      { texture: this.quality, origin: { x: slot.x, y: slot.y } },
      { width: PERSISTENT_TILE_SIZE, height: PERSISTENT_TILE_SIZE }
    );
  }

  destroy(): void {
    this.colour.destroy();
    this.quality.destroy();
    this.leaseDirectory.destroy();
  }
}
