import { useCallback, useEffect, useState } from "react";
import { FileUpload } from "@/components/FileUpload";
import { SliceViewer } from "@/components/SliceViewer";
import { SliceSlider } from "@/components/SliceSlider";
import { Volume3DViewer } from "@/components/Volume3DViewer";
import { ClassTogglePanel } from "@/components/ClassTogglePanel";
import { parseDicomFiles } from "@/lib/dicomParser";
import { fetchModelNames, runInference } from "@/lib/inferenceApi";
import { SEGMENTATION_CLASSES } from "@/types/segmentation";
import type {
  BoundingBoxPrompt,
  DicomSlice,
  SegmentationClass,
  SegmentationMask,
} from "@/types/segmentation";
import { Box, Loader2, Scan, Settings } from "lucide-react";
import { toast } from "sonner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const Index = () => {
  const [slices, setSlices] = useState<DicomSlice[]>([]);
  const [currentSliceIdx, setCurrentSliceIdx] = useState(0);
  const [classes, setClasses] = useState<SegmentationClass[]>(SEGMENTATION_CLASSES);
  const [masks, setMasks] = useState<SegmentationMask[]>([]);
  const [overlayOpacity, setOverlayOpacity] = useState(0.45);
  const [imageOpacity, setImageOpacity] = useState(1.0);
  const [isLoading, setIsLoading] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000/segment");
  const [showSettings, setShowSettings] = useState(false);
  const [showConfidences, setShowConfidences] = useState(false);
  const [showErrors, setShowErrors] = useState(true);
  const [rawFiles, setRawFiles] = useState<File[]>([]);
  const [viewMode, setViewMode] = useState<"2D" | "3D">("2D");
  const [surfaceSmoothness, setSurfaceSmoothness] = useState(0);
  const [sliceSpacingScale, setSliceSpacingScale] = useState(4);
  const [promptBoxes, setPromptBoxes] = useState<BoundingBoxPrompt[]>([]);
  const [activePromptClassId, setActivePromptClassId] = useState(SEGMENTATION_CLASSES[0].id);
  const [modelNames, setModelNames] = useState<string[]>([]);
  const [selectedModelName, setSelectedModelName] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetchModelNames(apiUrl)
      .then((models) => {
        if (cancelled) return;
        setModelNames(models);
        if (!selectedModelName && models.length > 0) {
          setSelectedModelName(models[0]);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setModelNames([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [apiUrl, selectedModelName]);

  const handleFilesSelected = useCallback(async (files: File[]) => {
    setIsLoading(true);
    try {
      const parsed = await parseDicomFiles(files);
      setSlices(parsed);
      setRawFiles(files);
      setCurrentSliceIdx(0);
      setMasks([]);
      setPromptBoxes([]);
      setViewMode("2D");
      toast.success(`Loaded ${parsed.length} DICOM slices`);
    } catch (err) {
      toast.error(`Failed to parse DICOM: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRunInference = useCallback(async () => {
    if (rawFiles.length === 0) return;
    if (promptBoxes.length === 0) {
      toast.error("Draw at least one bbox prompt before running MedSAM2.");
      return;
    }

    setIsInferring(true);
    try {
      const result = await runInference(rawFiles, promptBoxes, apiUrl, selectedModelName || undefined);
      setMasks(result.masks);
      toast.success(`Segmentation complete${result.model ? ` (${result.model})` : ""}!`);
    } catch (err) {
      toast.error(`Inference failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsInferring(false);
    }
  }, [apiUrl, promptBoxes, rawFiles, selectedModelName]);

  const handlePromptChange = useCallback((prompt: BoundingBoxPrompt) => {
    setPromptBoxes((prev) => {
      const next = prev.filter(
        (item) => !(item.classId === prompt.classId && item.sliceIndex === prompt.sliceIndex),
      );
      next.push(prompt);
      return next.sort((a, b) => a.sliceIndex - b.sliceIndex || a.classId - b.classId);
    });
  }, []);

  const markCurrentSliceAbsent = useCallback((classId: number) => {
    handlePromptChange({
      classId,
      sliceIndex: currentSliceIdx,
      x1: 0,
      y1: 0,
      x2: 1,
      y2: 1,
    });
  }, [currentSliceIdx, handlePromptChange]);

  const clearCurrentPrompt = useCallback(() => {
    setPromptBoxes((prev) =>
      prev.filter(
        (item) => !(item.classId === activePromptClassId && item.sliceIndex === currentSliceIdx),
      ),
    );
  }, [activePromptClassId, currentSliceIdx]);

  const clearAllPrompts = useCallback(() => {
    setPromptBoxes([]);
  }, []);

  const toggleClass = useCallback((id: number) => {
    setClasses((prev) =>
      prev.map((c) => (c.id === id ? { ...c, visible: !c.visible } : c)),
    );
  }, []);

  const currentSlice = slices[currentSliceIdx];
  const currentMask = masks.find((m) => m.sliceIndex === currentSliceIdx);
  const currentSlicePrompts = promptBoxes.filter((prompt) => prompt.sliceIndex === currentSliceIdx);
  const activeClassName = classes.find((cls) => cls.id === activePromptClassId)?.name ?? "None";

  const hasAbsentPrompt = useCallback((classId: number) => {
    return promptBoxes.some(
      (item) =>
        item.classId === classId &&
        item.sliceIndex === currentSliceIdx &&
        item.x1 === 0 &&
        item.y1 === 0 &&
        item.x2 === 1 &&
        item.y2 === 1,
    );
  }, [currentSliceIdx, promptBoxes]);

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      <header className="flex items-center justify-between px-5 py-3 bg-card border-b border-border flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/15 flex items-center justify-center">
            <Scan className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-foreground tracking-tight">ProSeg Viewer</h1>
            <p className="text-[10px] font-mono text-muted-foreground">
              MedSAM2 Prostate MRI Segmentation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {slices.length > 0 && (
            <span className="text-xs font-mono text-muted-foreground px-2 py-1 bg-secondary rounded">
              {slices.length} slices
            </span>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 rounded-md hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground"
              >
                <Settings className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">API Settings</TooltipContent>
          </Tooltip>
        </div>
      </header>

      {showSettings && (
        <div className="px-5 py-3 bg-card border-b border-border flex items-center gap-3">
          <label className="text-xs font-mono text-muted-foreground">API URL</label>
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            className="flex-1 px-3 py-2 bg-background border border-border rounded-md text-sm font-mono"
          />
        </div>
      )}

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 p-4 flex flex-col gap-4 min-w-0">
          {slices.length > 0 && currentSlice ? (
            <>
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setViewMode("2D")}
                      className={`px-4 py-2 rounded-md text-sm flex items-center gap-2 border ${
                        viewMode === "2D"
                          ? "bg-primary text-primary-foreground border-primary"
                          : "border-border text-muted-foreground hover:bg-secondary"
                      }`}
                    >
                      <Scan className="w-3.5 h-3.5" />
                      2D Viewer
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">View single DICOM slices</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setViewMode("3D")}
                      className={`px-4 py-2 rounded-md text-sm flex items-center gap-2 border ${
                        viewMode === "3D"
                          ? "bg-primary text-primary-foreground border-primary"
                          : "border-border text-muted-foreground hover:bg-secondary"
                      }`}
                    >
                      <Box className="w-3.5 h-3.5" />
                      3D Volume
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">View true 3D reconstructed volume</TooltipContent>
                </Tooltip>
              </div>

              <div className="flex-1 min-h-0">
                {viewMode === "2D" ? (
                  <SliceViewer
                    slice={currentSlice}
                    mask={currentMask}
                    classes={classes}
                    overlayOpacity={overlayOpacity}
                    imageOpacity={imageOpacity}
                    showErrors={showErrors}
                    promptBoxes={currentSlicePrompts}
                    selectedPromptClassId={activePromptClassId}
                    onPromptChange={handlePromptChange}
                  />
                ) : (
                  <Volume3DViewer
                    slices={slices}
                    masks={masks}
                    classes={classes}
                    overlayOpacity={overlayOpacity}
                    imageOpacity={imageOpacity}
                    showErrors={showErrors}
                    smoothness={surfaceSmoothness}
                    sliceSpacingScale={sliceSpacingScale}
                  />
                )}
              </div>

              {viewMode === "2D" && (
                <SliceSlider
                  currentSlice={currentSliceIdx}
                  totalSlices={slices.length}
                  onChange={setCurrentSliceIdx}
                />
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="max-w-md w-full">
                <FileUpload onFilesSelected={handleFilesSelected} isLoading={isLoading} />
              </div>
            </div>
          )}
        </div>

        {slices.length > 0 && currentSlice && (
          <div className="w-72 border-l border-border bg-card p-4 flex flex-col gap-4 overflow-y-auto flex-shrink-0">
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={handleRunInference}
                  disabled={isInferring || slices.length === 0}
                  className="w-full py-2.5 px-4 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isInferring ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Running…
                    </>
                  ) : (
                    <>
                      <Scan className="w-4 h-4" />
                      Run Segmentation
                    </>
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">Run MedSAM2 inference with bbox prompts</TooltipContent>
            </Tooltip>

            <div className="panel-section space-y-3">
              <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
                Prompt Boxes
              </h3>
              {modelNames.length > 0 && (
                <div>
                  <label className="text-xs font-mono text-muted-foreground block mb-2">Model</label>
                  <select
                    value={selectedModelName}
                    onChange={(e) => setSelectedModelName(e.target.value)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground"
                  >
                    {modelNames.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <div className="space-y-2">
                {classes.map((cls) => {
                  const active = cls.id === activePromptClassId;
                  const count = promptBoxes.filter((item) => item.classId === cls.id).length;
                  const absent = hasAbsentPrompt(cls.id);
                  return (
                    <div key={cls.id} className="flex gap-2">
                      <button
                        onClick={() => setActivePromptClassId(cls.id)}
                        className={`flex-1 flex items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors ${
                          active
                            ? "border-primary bg-primary/10 text-foreground"
                            : "border-border bg-background text-muted-foreground"
                        }`}
                      >
                        <span className="flex items-center gap-2">
                          <span
                            className="inline-block h-3 w-3 rounded-full"
                            style={{ backgroundColor: cls.color }}
                          />
                          {cls.name}
                        </span>
                        <span className="text-xs font-mono">{count}</span>
                      </button>
                      <button
                        onClick={() => markCurrentSliceAbsent(cls.id)}
                        className={`rounded-md border px-2 py-2 text-[11px] font-mono transition-colors ${
                          absent
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border bg-background text-muted-foreground hover:bg-secondary"
                        }`}
                        title="Mark this class as absent on the current slice"
                      >
                        Absent
                      </button>
                    </div>
                  );
                })}
              </div>
              <div className="space-y-1 text-xs font-mono text-muted-foreground">
                <p>Drag on the 2D viewer to place one bbox per class per slice.</p>
                <p>Active class: {activeClassName}</p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={clearCurrentPrompt}
                  className="flex-1 rounded-md border border-border px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-secondary"
                >
                  Clear Slice Box
                </button>
                <button
                  onClick={clearAllPrompts}
                  className="flex-1 rounded-md border border-border px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-secondary"
                >
                  Clear All
                </button>
              </div>
            </div>

            {viewMode === "3D" && (
              <div className="panel-section space-y-3">
                <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
                  Surface Mesh
                </h3>
                <div>
                  <label className="text-xs font-mono text-muted-foreground block mb-2">
                    Smoothness: {surfaceSmoothness}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="4"
                    step="1"
                    value={surfaceSmoothness}
                    onChange={(e) => setSurfaceSmoothness(parseInt(e.target.value, 10))}
                    className="w-full accent-primary h-1"
                  />
                </div>
                <div>
                  <label className="text-xs font-mono text-muted-foreground block mb-2">
                    Slice Spacing: {sliceSpacingScale.toFixed(1)}x
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="12"
                    step="0.5"
                    value={sliceSpacingScale}
                    onChange={(e) => setSliceSpacingScale(parseFloat(e.target.value))}
                    className="w-full accent-primary h-1"
                  />
                </div>
                <p className="text-xs font-mono text-muted-foreground">
                  0 keeps the raw surface. Higher values apply light mesh smoothing.
                </p>
              </div>
            )}

            <ClassTogglePanel
              classes={classes}
              onToggle={toggleClass}
              opacity={overlayOpacity}
              onOpacityChange={setOverlayOpacity}
              imageOpacity={imageOpacity}
              onImageOpacityChange={setImageOpacity}
              showConfidences={showConfidences}
              onShowConfidencesChange={setShowConfidences}
              confidences={currentMask?.confidences}
              showErrors={showErrors}
              onShowErrorsChange={setShowErrors}
              errors={currentMask?.errors}
            />

            <div className="panel-section space-y-2">
              <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
                Slice Info
              </h3>
              <div className="space-y-1 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">File</span>
                  <span className="text-foreground truncate ml-2 max-w-[140px]">
                    {currentSlice.fileName}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Size</span>
                  <span className="text-foreground">
                    {currentSlice.width}×{currentSlice.height}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Window</span>
                  <span className="text-foreground">
                    C:{currentSlice.windowCenter} W:{currentSlice.windowWidth}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Prompt Boxes</span>
                  <span className="text-foreground">{currentSlicePrompts.length}</span>
                </div>
              </div>
            </div>

            <div className="mt-auto">
              <FileUpload onFilesSelected={handleFilesSelected} isLoading={isLoading} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Index;
