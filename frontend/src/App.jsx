import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  CheckCircle2,
  HeartPulse,
  LoaderCircle,
  ShieldAlert,
  UploadCloud,
  X,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(nextFile) {
    setError("");
    setResult(null);
    if (!nextFile) return;

    const validExtensions = [".png", ".jpg", ".jpeg", ".webp", ".dcm"];
    const fileExt = nextFile.name
      .slice(nextFile.name.lastIndexOf("."))
      .toLowerCase();

    if (!validExtensions.includes(fileExt)) {
      setError("Please choose a PNG, JPG, JPEG, WEBP, or DICOM (.dcm) image.");
      return;
    }
    if (nextFile.size > 20 * 1024 * 1024) {
      setError("The image must be 20 MB or smaller.");
      return;
    }
    setFile(nextFile);
  }

  async function analyze() {
    if (!file) {
      setError("Choose a chest X-ray image first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail || "The request failed. Is the backend running?",
        );
      }
      setResult(payload);
    } catch (err) {
      setError(err.message || "Could not connect to the API.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  const isMock = result?.model_mode === "mock_demo" || result?.model_mode === "mock";
  const pneumonia = result?.pneumonia_probability ?? 0;
  const normal = result?.normal_probability ?? 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Pneumonia AI home">
          <span className="brand-mark">
            <HeartPulse size={22} />
          </span>
          <span>
            Pneumonia<span className="brand-light">AI</span>
          </span>
        </a>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow">
            <Activity size={15} /> CHEST RADIOGRAPHS
          </div>
          <h1>
            Chest X-ray
            <br />
            <span>screening workspace</span>
          </h1>
          <p className="hero-copy">Upload a chest radiograph to test.</p>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <p className="step-label">STEP 01</p>
                <h2>Upload an image</h2>
              </div>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.dcm"
              hidden
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            {!preview ? (
              <button
                className="dropzone"
                onClick={() => inputRef.current?.click()}
              >
                <span className="upload-icon">
                  <UploadCloud size={26} />
                </span>
                <strong>Choose chest X-ray</strong>
                <span>Click to browse PNG, JPG, WEBP, or DICOM</span>
                <small>Maximum file size: 20 MB</small>
              </button>
            ) : (
              <div className="preview-wrap">
                <img
                  src={preview}
                  alt="Preview of selected image"
                  className="xray-preview"
                />
                <button
                  className="remove-file"
                  onClick={reset}
                  aria-label="Remove selected image"
                >
                  <X size={16} />
                </button>
                <div className="file-meta">
                  <span>{file?.name}</span>
                  <span className="file-size">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
              </div>
            )}

            {error && (
              <div className="error-message">
                <ShieldAlert size={17} /> {error}
              </div>
            )}

            <button
              className="primary-button"
              onClick={analyze}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <LoaderCircle className="spin" size={18} /> Analyzing…
                </>
              ) : (
                <>
                  Run demo analysis <ArrowUpRight size={18} />
                </>
              )}
            </button>
          </div>

          <div className="panel results-panel">
            <div className="panel-heading">
              <div>
                <p className="step-label">STEP 02</p>
                <h2>Analysis result</h2>
              </div>
            </div>

            {!result ? (
              <div className="empty-state">
                <div className="empty-art">
                  <Activity size={31} />
                </div>
                <h3>Waiting for an image</h3>
                <p>
                  Your prediction summary will appear here after you run the
                  analysis.
                </p>
                <div className="empty-row">
                  <span>Prediction summary</span>
                  <span>—</span>
                </div>
                <div className="empty-row">
                  <span>Image quality checks</span>
                  <span>—</span>
                </div>
              </div>
            ) : result.prediction === null ? (
              <div className="quality-fail">
                <ShieldAlert size={25} />
                <h3>Analysis not completed</h3>
                <p>{result.message || "Unable to process the image."}</p>
              </div>
            ) : (
              <div className="result-content">
                <div className="prediction-card">
                  <div>
                    <span className="result-overline">PREDICTION</span>
                    <h3>{result.prediction}</h3>
                  </div>
                  <span className="prediction-symbol">
                    <Activity size={24} />
                  </span>
                </div>

                <div className="probability-block">
                  <div className="probability-title">
                    <span>Pneumonia score</span>
                    <strong>{(pneumonia * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{ width: `${Math.min(pneumonia * 100, 100)}%` }}
                    />
                  </div>
                </div>

                <div className="probability-block">
                  <div className="probability-title">
                    <span>Normal score</span>
                    <strong>{(normal * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill normal-fill"
                      style={{ width: `${Math.min(normal * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {result.quality && (
                  <>
                    <div className="quality-heading">
                      <h4>Technical image checks</h4>
                      <span
                        className={
                          result.quality.is_usable
                            ? "quality-tag pass"
                            : "quality-tag warn"
                        }
                      >
                        {result.quality.is_usable ? "Passed heuristics" : "Review"}
                      </span>
                    </div>
                    <div className="quality-grid">
                      {Object.entries(result.quality.checks || {}).map(
                        ([name, passed]) => (
                          <div className="quality-item" key={name}>
                            {passed ? (
                              <CheckCircle2 size={16} />
                            ) : (
                              <ShieldAlert size={16} />
                            )}
                            <span>
                              {name.charAt(0).toUpperCase() + name.slice(1)}
                            </span>
                            <strong>{passed ? "Pass" : "Check"}</strong>
                          </div>
                        ),
                      )}
                    </div>
                    {result.quality.note && (
                      <p className="quality-note">{result.quality.note}</p>
                    )}
                  </>
                )}

                {result.explanation && (
                  <p className="explanation">{result.explanation}</p>
                )}

                {isMock && (
                  <p className="preview-note">
                    Preview only — trained model integration pending.
                  </p>
                )}
              </div>
            )}
          </div>
        </section>

        <section className="notice">
          <ShieldAlert size={19} />
          <div>
            <strong>Research and education only</strong>
            <p>
              This tool is not a medical device and must not be used to
              diagnose. A healthcare professional must interpret medical images.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}