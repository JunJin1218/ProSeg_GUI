import { useEffect, useMemo, useRef, useState } from "react";
import type {
  BoundingBoxPrompt,
  DicomSlice,
  SegmentationClass,
  SegmentationMask,
} from "@/types/segmentation";

interface SliceViewerProps {
  slice: DicomSlice;
  mask?: SegmentationMask;
  classes: SegmentationClass[];
  overlayOpacity: number;
  imageOpacity?: number;
  showErrors?: boolean;
  promptBoxes?: BoundingBoxPrompt[];
  selectedPromptClassId?: number;
  onPromptChange?: (prompt: BoundingBoxPrompt) => void;
}

type DraftBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

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

export function SliceViewer({
  slice,
  mask,
  classes,
  overlayOpacity,
  imageOpacity = 1.0,
  showErrors = true,
  promptBoxes = [],
  selectedPromptClassId,
  onPromptChange,
}: SliceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const [draftBox, setDraftBox] = useState<DraftBox | null>(null);

  const classColorMap = useMemo(() => {
    const map: Record<number, { rgb: [number, number, number]; visible: boolean; color: string }> = {};
    classes.forEach((cls) => {
      const [h, s, l] = parseHSL(cls.color);
      map[cls.id] = { rgb: hslToRgb(h, s, l), visible: cls.visible, color: cls.color };
    });
    return map;
  }, [classes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height, pixelData, windowCenter, windowWidth, rescaleSlope, rescaleIntercept } = slice;
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    const wMin = windowCenter - windowWidth / 2;
    const wMax = windowCenter + windowWidth / 2;
    const baseAlpha = Math.max(0, Math.min(255, Math.round(imageOpacity * 255)));

    for (let i = 0; i < pixelData.length; i++) {
      const hu = pixelData[i] * rescaleSlope + rescaleIntercept;
      let gray = ((hu - wMin) / (wMax - wMin)) * 255;
      gray = Math.max(0, Math.min(255, gray));

      const idx = i * 4;
      data[idx] = gray;
      data[idx + 1] = gray;
      data[idx + 2] = gray;
      data[idx + 3] = baseAlpha;
    }

    if (mask && mask.data.length === width * height) {
      for (let i = 0; i < mask.data.length; i++) {
        const classId = mask.data[i];
        if (classId === 0) continue;
        if (!showErrors && mask.errors && mask.errors[classId]) continue;

        const classInfo = classColorMap[classId];
        if (!classInfo || !classInfo.visible) continue;

        const idx = i * 4;
        const [r, g, b] = classInfo.rgb;
        const alpha = overlayOpacity;
        data[idx] = data[idx] * (1 - alpha) + r * alpha;
        data[idx + 1] = data[idx + 1] * (1 - alpha) + g * alpha;
        data[idx + 2] = data[idx + 2] * (1 - alpha) + b * alpha;
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [slice, mask, classColorMap, overlayOpacity, imageOpacity, showErrors]);

  const toImageCoords = (clientX: number, clientY: number) => {
    const surface = surfaceRef.current;
    if (!surface) return null;
    const rect = surface.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;

    const x = Math.round(((clientX - rect.left) / rect.width) * (slice.width - 1));
    const y = Math.round(((clientY - rect.top) / rect.height) * (slice.height - 1));

    return {
      x: Math.max(0, Math.min(slice.width - 1, x)),
      y: Math.max(0, Math.min(slice.height - 1, y)),
    };
  };

  const finishDraft = () => {
    if (!draftBox || !selectedPromptClassId || !onPromptChange) {
      setDraftBox(null);
      return;
    }

    const x1 = Math.min(draftBox.x1, draftBox.x2);
    const y1 = Math.min(draftBox.y1, draftBox.y2);
    const x2 = Math.max(draftBox.x1, draftBox.x2);
    const y2 = Math.max(draftBox.y1, draftBox.y2);

    if (x2 - x1 < 3 || y2 - y1 < 3) {
      setDraftBox(null);
      return;
    }

    onPromptChange({
      classId: selectedPromptClassId,
      sliceIndex: slice.index,
      x1,
      y1,
      x2,
      y2,
    });
    setDraftBox(null);
  };

  const selectedClass = selectedPromptClassId ? classColorMap[selectedPromptClassId] : undefined;

  return (
    <div className="flex items-center justify-center w-full h-full bg-viewer-bg rounded-lg overflow-hidden">
      <div
        ref={surfaceRef}
        className="relative max-w-full max-h-full"
        onMouseDown={(event) => {
          if (!selectedPromptClassId || !onPromptChange) return;
          const coords = toImageCoords(event.clientX, event.clientY);
          if (!coords) return;
          draggingRef.current = true;
          setDraftBox({ x1: coords.x, y1: coords.y, x2: coords.x, y2: coords.y });
        }}
        onMouseMove={(event) => {
          if (!draggingRef.current) return;
          const coords = toImageCoords(event.clientX, event.clientY);
          if (!coords) return;
          setDraftBox((prev) => (prev ? { ...prev, x2: coords.x, y2: coords.y } : prev));
        }}
        onMouseUp={() => {
          draggingRef.current = false;
          finishDraft();
        }}
        onMouseLeave={() => {
          if (!draggingRef.current) return;
          draggingRef.current = false;
          finishDraft();
        }}
      >
        <canvas
          ref={canvasRef}
          className="viewer-canvas max-w-full max-h-full object-contain"
          style={{ imageRendering: "pixelated" }}
        />
        <svg
          viewBox={`0 0 ${slice.width} ${slice.height}`}
          className="absolute inset-0 w-full h-full pointer-events-none"
        >
          {promptBoxes.map((prompt) => {
            const cls = classColorMap[prompt.classId];
            const x = Math.min(prompt.x1, prompt.x2);
            const y = Math.min(prompt.y1, prompt.y2);
            const width = Math.abs(prompt.x2 - prompt.x1);
            const height = Math.abs(prompt.y2 - prompt.y1);
            const isSelected = prompt.classId === selectedPromptClassId;
            return (
              <rect
                key={`${prompt.classId}-${prompt.sliceIndex}`}
                x={x}
                y={y}
                width={width}
                height={height}
                fill={cls?.color ?? "transparent"}
                fillOpacity={0.08}
                stroke={cls?.color ?? "#ffffff"}
                strokeWidth={isSelected ? 2.5 : 1.5}
                strokeDasharray={isSelected ? "6 4" : "0"}
              />
            );
          })}
          {draftBox && selectedClass && (
            <rect
              x={Math.min(draftBox.x1, draftBox.x2)}
              y={Math.min(draftBox.y1, draftBox.y2)}
              width={Math.abs(draftBox.x2 - draftBox.x1)}
              height={Math.abs(draftBox.y2 - draftBox.y1)}
              fill={selectedClass.color}
              fillOpacity={0.12}
              stroke={selectedClass.color}
              strokeWidth={2.5}
              strokeDasharray="6 4"
            />
          )}
        </svg>
      </div>
    </div>
  );
}
