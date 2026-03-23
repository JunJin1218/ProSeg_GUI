export interface DicomSlice {
  index: number;
  fileName: string;
  pixelData: number[];
  width: number;
  height: number;
  windowCenter: number;
  windowWidth: number;
  rescaleSlope: number;
  rescaleIntercept: number;
  instanceNumber: number;
}

export interface SegmentationClass {
  id: number;
  name: string;
  color: string;
  visible: boolean;
}

export interface SegmentationMask {
  sliceIndex: number;
  width: number;
  height: number;
  // Per-pixel class labels (0 = background)
  data: number[];
  confidences?: Record<number, number>;
  errors?: Record<number, boolean>;
}

export interface BoundingBoxPrompt {
  classId: number;
  sliceIndex: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface PredictionResult {
  masks: SegmentationMask[];
  model?: string;
  predictorType?: string;
}

export const SEGMENTATION_CLASSES: SegmentationClass[] = [
  { id: 1, name: "Prostate1", color: "hsl(0, 80%, 65%)", visible: true },
  { id: 2, name: "Prostate2", color: "hsl(30, 90%, 60%)", visible: true },
  { id: 3, name: "Pubic Bone", color: "hsl(200, 70%, 65%)", visible: true },
  { id: 4, name: "Bladder", color: "hsl(140, 60%, 55%)", visible: true },
  { id: 5, name: "Rectum", color: "hsl(300, 50%, 70%)", visible: true },
  { id: 6, name: "SV", color: "hsl(50, 80%, 65%)", visible: true },
];
