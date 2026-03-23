import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { MarchingCubes } from "three/examples/jsm/objects/MarchingCubes.js";
import type { DicomSlice, SegmentationClass, SegmentationMask } from "@/types/segmentation";

function parseHSL(hsl: string): [number, number, number] {
  const match = hsl.match(/hsl\((\d+),\s*(\d+)%?,\s*(\d+)%?\)/);
  if (!match) return [0, 0, 0];
  return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  s /= 100;
  l /= 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

interface Volume3DViewerProps {
  slices: DicomSlice[];
  masks: SegmentationMask[];
  classes: SegmentationClass[];
  overlayOpacity: number;
  imageOpacity?: number;
  showErrors?: boolean;
  smoothness?: number;
  sliceSpacingScale?: number;
}

type SurfaceVolume = {
  classId: number;
  color: [number, number, number];
  resolution: number;
  field: Float32Array;
  scale: [number, number, number];
};

function ClassSurface({
  field,
  resolution,
  color,
  scale,
  opacity,
  smoothness,
}: {
  field: Float32Array;
  resolution: number;
  color: [number, number, number];
  scale: [number, number, number];
  opacity: number;
  smoothness: number;
}) {
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(color[0] / 255, color[1] / 255, color[2] / 255),
        transparent: true,
        opacity,
        metalness: 0.05,
        roughness: 0.35,
      }),
    [color, opacity],
  );

  const surface = useMemo(() => new MarchingCubes(resolution, material, false, false, 50000), [material, resolution]);

  useEffect(() => {
    surface.reset();
    surface.isolation = 50;

    for (let i = 0; i < field.length; i += 1) {
      surface.field[i] = field[i];
    }

    const passes = Math.max(0, Math.round(smoothness));
    for (let i = 0; i < passes; i += 1) {
      surface.blur(0.35);
    }

    surface.scale.set(scale[0], scale[1], scale[2]);
    surface.update();
  }, [field, scale, smoothness, surface]);

  useEffect(() => {
    return () => {
      surface.geometry.dispose();
      material.dispose();
    };
  }, [material, surface]);

  return <primitive object={surface} />;
}

export function Volume3DViewer({
  slices,
  masks,
  classes,
  overlayOpacity,
  showErrors = true,
  smoothness = 0,
  sliceSpacingScale = 1,
}: Volume3DViewerProps) {
  const surfaceVolumes = useMemo<SurfaceVolume[]>(() => {
    if (!slices.length || !masks.length) return [];

    const width = slices[0].width;
    const height = slices[0].height;
    const depth = slices.length;
    const longestAxis = Math.max(width, height, depth);
    const resolution = Math.max(28, Math.min(56, Math.round(longestAxis / 6)));
    const margin = Math.max(2, Math.round(resolution * 0.06));
    const interior = Math.max(2, resolution - margin * 2);
    const maskBySlice = new Map(masks.map((mask) => [mask.sliceIndex, mask]));

    return classes
      .filter((cls) => cls.visible)
      .map((cls) => {
        const [r, g, b] = hslToRgb(...parseHSL(cls.color));
        const field = new Float32Array(resolution * resolution * resolution);
        let hasVoxels = false;

        for (let z = 0; z < resolution; z += 1) {
          const zNorm = (z - margin) / Math.max(1, interior - 1);
          if (zNorm < 0 || zNorm > 1) continue;
          const srcZ = Math.round(zNorm * Math.max(0, depth - 1));
          const mask = maskBySlice.get(srcZ);
          if (!mask) continue;
          if (!showErrors && mask.errors && mask.errors[cls.id]) continue;

          for (let y = 0; y < resolution; y += 1) {
            const yNorm = (y - margin) / Math.max(1, interior - 1);
            if (yNorm < 0 || yNorm > 1) continue;
            const srcY = Math.round(yNorm * Math.max(0, height - 1));
            for (let x = 0; x < resolution; x += 1) {
              const xNorm = (x - margin) / Math.max(1, interior - 1);
              if (xNorm < 0 || xNorm > 1) continue;
              const srcX = Math.round(xNorm * Math.max(0, width - 1));
              const srcIndex = srcY * width + srcX;
              if (mask.data[srcIndex] !== cls.id) continue;
              const index = z * resolution * resolution + y * resolution + x;
              field[index] = 100;
              hasVoxels = true;
            }
          }
        }

        if (!hasVoxels) return null;

        return {
          classId: cls.id,
          color: [r, g, b] as [number, number, number],
          resolution,
          field,
          scale: [
            1.6,
            (height / Math.max(1, width)) * 1.6,
            (depth / Math.max(1, width)) * 2.2 * sliceSpacingScale,
          ],
        };
      })
      .filter((item): item is SurfaceVolume => item !== null);
  }, [classes, masks, showErrors, sliceSpacingScale, slices]);

  return (
    <div className="w-full h-full bg-black/95 rounded-lg overflow-hidden relative cursor-move">
      <Canvas camera={{ position: [2.4, -3.4, 2.4], fov: 38 }}>
        <ambientLight intensity={1.4} />
        <directionalLight position={[6, 8, 5]} intensity={2.2} />
        <directionalLight position={[-5, -4, -6]} intensity={0.8} color="#7ea5ff" />
        <gridHelper args={[4, 8, "#223047", "#182231"]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -1.4]} />
        {surfaceVolumes.map((volume) => (
          <ClassSurface
            key={volume.classId}
            field={volume.field}
            resolution={volume.resolution}
            color={volume.color}
            scale={volume.scale}
            opacity={Math.max(0.2, overlayOpacity)}
            smoothness={smoothness}
          />
        ))}
        <OrbitControls enableZoom enablePan enableRotate />
      </Canvas>
      <div className="absolute inset-x-0 bottom-4 pointer-events-none flex justify-center text-xs font-mono text-zinc-400">
        Drag to rotate. Scroll to zoom.
      </div>
      {!surfaceVolumes.length && (
        <div className="absolute inset-0 flex items-center justify-center text-xs font-mono text-zinc-500">
          No visible 3D surface for the current masks.
        </div>
      )}
    </div>
  );
}
