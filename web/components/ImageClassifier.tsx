"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { GREENERY_CLASSES } from "@/lib/classes";
import { disposeModels, loadModels, predictGreenery } from "@/lib/predict";
import styles from "./ImageClassifier.module.css";

type ModelStatus = "loading" | "ready" | "error";

interface Prediction {
  classId: string;
  confidence: number;
}

export default function ImageClassifier() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>("loading");
  const [modelError, setModelError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const imageRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await loadModels();
        if (!cancelled) setModelStatus("ready");
      } catch (err) {
        if (!cancelled) {
          setModelStatus("error");
          setModelError(
            err instanceof Error ? err.message : "Failed to load model"
          );
        }
      }
    }

    init();

    return () => {
      cancelled = true;
      disposeModels();
    };
  }, []);

  const runPrediction = useCallback(async () => {
    const img = imageRef.current;
    if (!img || !img.complete) return;

    setIsPredicting(true);
    setModelError(null);
    try {
      const results = await predictGreenery(img);
      setPredictions(results);
    } catch (err) {
      setModelError(
        err instanceof Error ? err.message : "Prediction failed"
      );
    } finally {
      setIsPredicting(false);
    }
  }, []);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;

    const url = URL.createObjectURL(file);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return url;
    });
    setPredictions(null);
  }, []);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const onImageLoad = () => {
    if (modelStatus === "ready") runPrediction();
  };

  const topPrediction = predictions?.[0];
  const topClass = GREENERY_CLASSES.find((c) => c.id === topPrediction?.classId);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.badge}>ML Project Demo</div>
        <h1 className={styles.title}>Urban Green Cover Assessment</h1>
        <p className={styles.subtitle}>
          Upload an aerial scene image to classify urban greenery level using a
          MobileNetV2 CNN trained on the UC Merced Land Use dataset.
        </p>
        <div className={styles.stats}>
          <span>93.3% test accuracy</span>
          <span className={styles.dot} />
          <span>4 greenery classes</span>
          <span className={styles.dot} />
          <span>~12 ms inference</span>
        </div>
      </header>

      <div className={styles.statusBar}>
        {modelStatus === "loading" && (
          <span className={styles.statusLoading}>
            Loading MobileNetV2 backbone and classification head…
          </span>
        )}
        {modelStatus === "ready" && (
          <span className={styles.statusReady}>Model ready</span>
        )}
        {modelStatus === "error" && (
          <span className={styles.statusError}>
            {modelError ?? "Model failed to load"} — run{" "}
            <code>python scripts/export_head_weights.py</code>
          </span>
        )}
      </div>

      <div className={styles.grid}>
        <section className={styles.uploadSection}>
          <div
            className={`${styles.dropzone} ${dragOver ? styles.dropzoneActive : ""} ${previewUrl ? styles.dropzoneHasImage : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !previewUrl && fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className={styles.hiddenInput}
            />

            {previewUrl ? (
              <img
                ref={imageRef}
                src={previewUrl}
                alt="Uploaded aerial scene"
                className={styles.preview}
                onLoad={onImageLoad}
              />
            ) : (
              <div className={styles.dropzoneContent}>
                <div className={styles.uploadIcon}>
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 16V4m0 0L8 8m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
                  </svg>
                </div>
                <p className={styles.dropzoneTitle}>Drop an image here</p>
                <p className={styles.dropzoneHint}>or click to browse — JPG, PNG, WEBP</p>
              </div>
            )}
          </div>

          {previewUrl && (
            <div className={styles.actionRow}>
              <button
                type="button"
                className={styles.predictBtn}
                onClick={runPrediction}
                disabled={modelStatus !== "ready" || isPredicting}
              >
                {isPredicting ? "Analyzing…" : "Analyze greenery"}
              </button>
              <button
                type="button"
                className={styles.secondaryBtn}
                onClick={() => fileInputRef.current?.click()}
              >
                Change image
              </button>
            </div>
          )}
        </section>

        <section className={styles.resultsSection}>
          {topPrediction && topClass ? (
            <>
              <div
                className={styles.topResult}
                style={{ borderColor: topClass.color }}
              >
                <span className={styles.topLabel}>Predicted class</span>
                <h2 className={styles.topClassName} style={{ color: topClass.color }}>
                  {topClass.label}
                </h2>
                <p className={styles.topDescription}>{topClass.description}</p>
                <div className={styles.confidenceBadge}>
                  {(topPrediction.confidence * 100).toFixed(1)}% confidence
                </div>
              </div>

              <div className={styles.probList}>
                <h3 className={styles.probTitle}>All classes</h3>
                {GREENERY_CLASSES.map((cls) => {
                  const pred = predictions?.find((p) => p.classId === cls.id);
                  const pct = (pred?.confidence ?? 0) * 100;
                  return (
                    <div key={cls.id} className={styles.probRow}>
                      <div className={styles.probHeader}>
                        <span>{cls.label}</span>
                        <span>{pct.toFixed(1)}%</span>
                      </div>
                      <div className={styles.probBarTrack}>
                        <div
                          className={styles.probBarFill}
                          style={{ width: `${pct}%`, backgroundColor: cls.color }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className={styles.emptyResults}>
              <p>Upload an aerial image to see greenery classification results.</p>
              <ul className={styles.classList}>
                {GREENERY_CLASSES.map((cls) => (
                  <li key={cls.id}>
                    <span
                      className={styles.classDot}
                      style={{ backgroundColor: cls.color }}
                    />
                    {cls.label}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {modelError && modelStatus === "ready" && (
            <p className={styles.errorText}>{modelError}</p>
          )}
        </section>
      </div>

      <footer className={styles.footer}>
        <p>
          Penthala Harsha Vardhan · Urban Green Cover Image Assessment ·
          DSC01 Machine Learning
        </p>
        <a
          href="https://github.com/harshava123/urban_green_cover"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub repository
        </a>
      </footer>
    </div>
  );
}
