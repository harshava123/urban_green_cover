import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import { GREENERY_CLASSES } from "./classes";

export interface HeadWeights {
  model: string;
  backbone: string;
  embedding_dim: number;
  num_classes: number;
  classes: string[];
  kernel: number[][];
  bias: number[];
}

export interface PredictionResult {
  classId: string;
  confidence: number;
}

const HEAD_PATH = "/model/head_weights.json";

let mobilenetModel: mobilenet.MobileNet | null = null;
let headWeights: HeadWeights | null = null;
let kernelTensor: tf.Tensor2D | null = null;
let biasTensor: tf.Tensor1D | null = null;

export async function loadModels(): Promise<void> {
  if (!mobilenetModel) {
    mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
  }

  if (!headWeights) {
    const response = await fetch(HEAD_PATH);
    if (!response.ok) {
      throw new Error(
        "Head weights not found. Run: python scripts/export_head_weights.py"
      );
    }
    headWeights = (await response.json()) as HeadWeights;
    kernelTensor = tf.tensor2d(headWeights.kernel);
    biasTensor = tf.tensor1d(headWeights.bias);
  }
}

export async function predictGreenery(
  image: HTMLImageElement
): Promise<PredictionResult[]> {
  await loadModels();

  if (!mobilenetModel || !kernelTensor || !biasTensor || !headWeights) {
    throw new Error("Models not loaded");
  }

  const embedding = mobilenetModel.infer(image, true) as tf.Tensor2D;
  const logits = tf.matMul(embedding, kernelTensor).add(biasTensor);
  const probs = tf.softmax(logits);
  const values = await probs.data();

  embedding.dispose();
  logits.dispose();
  probs.dispose();

  const results: PredictionResult[] = GREENERY_CLASSES.map((cls, i) => ({
    classId: cls.id,
    confidence: values[i] ?? 0,
  }));

  results.sort((a, b) => b.confidence - a.confidence);
  return results;
}

export function disposeModels(): void {
  kernelTensor?.dispose();
  biasTensor?.dispose();
  kernelTensor = null;
  biasTensor = null;
  headWeights = null;
  mobilenetModel = null;
}
