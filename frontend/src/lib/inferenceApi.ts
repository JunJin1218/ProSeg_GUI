import type { BoundingBoxPrompt, PredictionResult } from "@/types/segmentation";

const DEFAULT_API_URL = "http://localhost:8000/segment";

export async function runInference(
  files: File[],
  prompts: BoundingBoxPrompt[],
  apiUrl: string = DEFAULT_API_URL,
  modelName?: string,
): Promise<PredictionResult> {
  console.log("[runInference] sending prompts", prompts);
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("files", file);
  });
  formData.append("prompts", JSON.stringify(prompts));
  if (modelName) {
    formData.append("model_name", modelName);
  }

  const response = await fetch(apiUrl, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Inference failed (${response.status}): ${text}`);
  }

  return response.json();
}

export async function fetchModelNames(
  apiUrl: string = DEFAULT_API_URL,
): Promise<string[]> {
  const modelsUrl = apiUrl.replace(/\/segment$/, "/models");
  const response = await fetch(modelsUrl);
  if (!response.ok) {
    return [];
  }
  const data = await response.json();
  return Array.isArray(data.models) ? data.models : [];
}
