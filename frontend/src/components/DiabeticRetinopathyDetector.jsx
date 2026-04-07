import React, { useState, useRef } from 'react';
import {
  Upload,
  Eye,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Camera,
  FileText,
  RefreshCw,
  Activity,
  Info
} from 'lucide-react';

const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const API_URL = `${BASE_URL}/predict`;
const QUALITY_CHECK_URL = `${BASE_URL}/check_quality`;
const HEATMAP_URL = `${BASE_URL}/predict_with_heatmap`;

const DiabeticRetinopathyDetector = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [qualityCheck, setQualityCheck] = useState(null);
  const [qualityChecking, setQualityChecking] = useState(false);
  const [heatmaps, setHeatmaps] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const fileInputRef = useRef(null);

  const getSeverityInfo = (pred) => {
    const severityMap = {
      'No_DR': {
        color: '#4caf50',
        icon: CheckCircle,
        label: 'No Diabetic Retinopathy',
        desc: 'No signs of diabetic retinopathy detected'
      },
      'Mild': {
        color: '#ffeb3b',
        icon: AlertTriangle,
        label: 'Mild DR',
        desc: 'Early stage - microaneurysms present'
      },
      'Moderate': {
        color: '#ff9800',
        icon: AlertTriangle,
        label: 'Moderate DR',
        desc: 'More extensive retinal changes detected'
      },
      'Severe': {
        color: '#f44336',
        icon: XCircle,
        label: 'Severe DR',
        desc: 'Advanced stage - requires immediate attention'
      },
      'Proliferative': {
        color: '#9c27b0',
        icon: XCircle,
        label: 'Proliferative DR',
        desc: 'Most advanced stage - urgent medical intervention needed'
      }
    };
    return severityMap[pred] || null;
  };

  const checkQuality = async (file) => {
    setQualityChecking(true);
    setQualityCheck(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(QUALITY_CHECK_URL, {
        method: "POST",
        body: formData
      });
      if (!response.ok) throw new Error("Quality check failed");
      const data = await response.json();
      console.log("Quality check response:", data);
      // Backend returns {success, quality_check} 
      const qc = data.quality_check || data;
      setQualityCheck(qc);
      return qc;
    } catch (err) {
      console.error('Quality check error:', err);
      return null;
    } finally {
      setQualityChecking(false);
    }
  };

  const handleFileSelect = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError("Please select a valid image file (JPG, PNG, etc.)");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("File size should be less than 10MB");
      return;
    }
    setSelectedFile(file);
    setError(null);
    setPrediction(null);
    setConfidence(null);
    setHeatmaps(null);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(file);
    await checkQuality(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    setConfidence(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const endpoint = showHeatmap ? HEATMAP_URL : API_URL;
      console.log("Calling endpoint:", endpoint);

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      });

      console.log("Response status:", response.status);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log("Full API Response:", data);

      // Handle failed prediction
      if (data.success === false) {
        setError(data.error || "Prediction failed");
        if (data.quality_check) setQualityCheck(data.quality_check);
        return;
      }

      // Set prediction - works even if success field missing
      const pred = data.prediction;
      const conf = data.confidence;

      console.log("Prediction:", pred, "Confidence:", conf);

      if (!pred) {
        setError("No prediction received from server");
        return;
      }

      setPrediction(pred);
      setConfidence(conf);

      // Update quality check from predict response if available
      if (data.quality_check) {
        setQualityCheck(data.quality_check);
      }

      if (data.heatmaps) {
        setHeatmaps(data.heatmaps);
      }

    } catch (err) {
      console.error('Prediction error:', err);
      setError("Unable to connect to backend. Please ensure the server is running.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setConfidence(null);
    setError(null);
    setQualityCheck(null);
    setHeatmaps(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const severityInfo = prediction ? getSeverityInfo(prediction) : null;
  const SeverityIcon = severityInfo?.icon;

  const getQualityColor = () => {
    if (!qualityCheck) return '#999';
    if (qualityCheck.status === 'passed') return '#4caf50';
    if (qualityCheck.status === 'warning') return '#ff9800';
    return '#f44336';
  };

  const getQualityIcon = () => {
    if (!qualityCheck) return Info;
    if (qualityCheck.status === 'passed') return CheckCircle;
    if (qualityCheck.status === 'warning') return AlertTriangle;
    return XCircle;
  };

  const QualityIcon = getQualityIcon();

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      position: 'relative',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        position: 'absolute',
        inset: 0,
        background: 'linear-gradient(135deg, rgba(230,245,255,0.95) 0%, rgba(255,255,255,0.98) 50%, rgba(232,234,246,0.95) 100%)',
        zIndex: 1
      }} />

      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '2rem 1.5rem',
        position: 'relative',
        zIndex: 2
      }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '1rem',
            marginBottom: '1rem'
          }}>
            <Eye size={56} color="#2196f3" strokeWidth={2} />
            <h1 style={{
              fontSize: '3rem',
              fontWeight: '800',
              background: 'linear-gradient(135deg, #1565c0 0%, #7b1fa2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              margin: 0
            }}>
              Diabetic Retinopathy Detector
            </h1>
          </div>
          <p style={{ color: '#555', fontSize: '1.1rem', margin: 0 }}>
            AI-powered retinal image analysis with quality validation
          </p>
        </div>

        {/* Cards Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
          gap: '2rem'
        }}>

          {/* Upload Card */}
          <div style={{
            background: 'white',
            borderRadius: '1.5rem',
            padding: '2.5rem',
            boxShadow: '0 20px 60px rgba(0,0,0,0.15)'
          }}>
            <h2 style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              fontSize: '1.5rem',
              marginBottom: '1.5rem',
              color: '#333'
            }}>
              <Upload size={28} color="#2196f3" />
              Upload Retinal Image
            </h2>

            {/* Upload Area */}
            <div
              style={{
                border: `3px dashed ${dragOver ? '#1976d2' : '#2196f3'}`,
                borderRadius: '1.5rem',
                padding: '3rem 2rem',
                textAlign: 'center',
                cursor: 'pointer',
                background: dragOver ? '#e3f2fd' : '#fafafa',
                transition: 'all 0.3s ease',
                minHeight: '320px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center'
              }}
              onClick={() => fileInputRef.current?.click()}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFileSelect(e.dataTransfer.files[0]);
              }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => handleFileSelect(e.target.files?.[0])}
              />
              {preview ? (
                <div>
                  <img
                    src={preview}
                    alt="preview"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '280px',
                      borderRadius: '1rem',
                      boxShadow: '0 8px 30px rgba(0,0,0,0.2)'
                    }}
                  />
                  <p style={{ marginTop: '1rem', color: '#666', fontSize: '0.9rem' }}>
                    {selectedFile?.name}
                  </p>
                </div>
              ) : (
                <div>
                  <Camera size={72} color="#2196f3" strokeWidth={1.5} />
                  <p style={{ fontSize: '1.2rem', color: '#555', marginTop: '1rem', marginBottom: '0.5rem' }}>
                    Drop retinal image here
                  </p>
                  <p style={{ fontSize: '0.95rem', color: '#888' }}>
                    or click to browse (JPG, PNG)
                  </p>
                </div>
              )}
            </div>

            {/* Quality Check Status */}
            {qualityChecking && (
              <div style={{
                marginTop: '1rem',
                padding: '1rem',
                background: '#f5f5f5',
                borderRadius: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem'
              }}>
                <Loader2 size={20} className="spin" color="#2196f3" />
                <span style={{ color: '#666' }}>Checking image quality...</span>
              </div>
            )}

            {qualityCheck && !qualityChecking && (
              <div style={{
                marginTop: '1rem',
                padding: '1rem',
                background: `${getQualityColor()}15`,
                border: `2px solid ${getQualityColor()}`,
                borderRadius: '0.75rem'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
                  <QualityIcon size={24} color={getQualityColor()} />
                  <strong style={{ color: getQualityColor() }}>
                    Quality Score: {qualityCheck.quality_score}/100
                  </strong>
                </div>
                <p style={{ margin: '0.5rem 0', fontSize: '0.9rem', color: '#666' }}>
                  {qualityCheck.message}
                </p>
                {qualityCheck.metrics && (
                  <div style={{
                    marginTop: '0.75rem',
                    fontSize: '0.85rem',
                    color: '#777',
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '0.5rem'
                  }}>
                    <div>Resolution: {qualityCheck.metrics.resolution}</div>
                    <div>Brightness: {qualityCheck.metrics.brightness}</div>
                    <div>Sharpness: {qualityCheck.metrics.sharpness}</div>
                    <div>Contrast: {qualityCheck.metrics.contrast}</div>
                  </div>
                )}
                {qualityCheck.issues && qualityCheck.issues.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <strong style={{ fontSize: '0.9rem', color: '#d32f2f' }}>Issues:</strong>
                    <ul style={{ margin: '0.25rem 0', paddingLeft: '1.5rem', fontSize: '0.85rem' }}>
                      {qualityCheck.issues.map((issue, idx) => (
                        <li key={idx} style={{ color: '#d32f2f' }}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {qualityCheck.warnings && qualityCheck.warnings.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <strong style={{ fontSize: '0.9rem', color: '#f57c00' }}>Warnings:</strong>
                    <ul style={{ margin: '0.25rem 0', paddingLeft: '1.5rem', fontSize: '0.85rem' }}>
                      {qualityCheck.warnings.map((warning, idx) => (
                        <li key={idx} style={{ color: '#f57c00' }}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Heatmap Toggle */}
            <div style={{
              marginTop: '1.5rem',
              padding: '1rem',
              background: '#f5f5f5',
              borderRadius: '0.75rem'
            }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                cursor: 'pointer',
                fontSize: '0.95rem',
                color: '#555'
              }}>
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)}
                  style={{ cursor: 'pointer' }}
                />
                <Eye size={18} />
                Generate attention heatmap (shows where model is looking)
              </label>
            </div>

            <button
              style={{
                width: '100%',
                padding: '1.1rem',
                marginTop: '1.5rem',
                borderRadius: '1rem',
                border: 'none',
                fontSize: '1.1rem',
                fontWeight: '600',
                cursor: loading || !selectedFile || (qualityCheck && qualityCheck.status === 'rejected') ? 'not-allowed' : 'pointer',
                background: loading || !selectedFile || (qualityCheck && qualityCheck.status === 'rejected')
                  ? '#ccc'
                  : 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.75rem',
                transition: 'all 0.3s ease',
                boxShadow: loading || !selectedFile ? 'none' : '0 4px 15px rgba(33,150,243,0.4)'
              }}
              disabled={loading || !selectedFile || (qualityCheck && qualityCheck.status === 'rejected')}
              onClick={handleUpload}
            >
              {loading ? (
                <><Loader2 size={20} className="spin" /> Analyzing Image...</>
              ) : (
                <><Activity size={20} /> Detect Diabetic Retinopathy</>
              )}
            </button>

            {selectedFile && !loading && (
              <button
                style={{
                  width: '100%',
                  padding: '1rem',
                  marginTop: '1rem',
                  borderRadius: '1rem',
                  border: '2px solid #e0e0e0',
                  fontSize: '1rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  background: 'white',
                  color: '#555',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.3s ease'
                }}
                onClick={reset}
              >
                <RefreshCw size={18} /> Reset
              </button>
            )}
          </div>

          {/* Results Card */}
          <div style={{
            background: 'white',
            borderRadius: '1.5rem',
            padding: '2.5rem',
            boxShadow: '0 20px 60px rgba(0,0,0,0.15)'
          }}>
            <h2 style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              fontSize: '1.5rem',
              marginBottom: '1.5rem',
              color: '#333'
            }}>
              <FileText size={28} color="#2196f3" />
              Analysis Results
            </h2>

            {error && (
              <div style={{
                background: '#ffebee',
                padding: '1.5rem',
                borderRadius: '1rem',
                color: '#c62828',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                border: '2px solid #ef5350'
              }}>
                <XCircle size={24} />
                <div><strong>Error:</strong> {error}</div>
              </div>
            )}

            {loading && (
              <div style={{ textAlign: 'center', padding: '3rem 2rem' }}>
                <Loader2 size={64} color="#2196f3" className="spin" />
                <p style={{ marginTop: '1.5rem', color: '#666', fontSize: '1.1rem' }}>
                  Processing retinal image...
                </p>
              </div>
            )}

            {!prediction && !error && !loading && (
              <div style={{ textAlign: 'center', padding: '3rem 2rem', color: '#999' }}>
                <Eye size={64} color="#ddd" strokeWidth={1.5} />
                <p style={{ marginTop: '1.5rem', fontSize: '1.1rem' }}>
                  Upload a retinal image to begin analysis
                </p>
              </div>
            )}

            {prediction && severityInfo && (
              <div style={{
                padding: '2rem',
                borderRadius: '1.25rem',
                background: `${severityInfo.color}15`,
                border: `3px solid ${severityInfo.color}`,
                animation: 'fadeIn 0.5s ease'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                  <SeverityIcon size={48} color={severityInfo.color} strokeWidth={2} />
                  <div>
                    <h3 style={{
                      fontSize: '1.8rem',
                      margin: '0 0 0.25rem 0',
                      color: severityInfo.color,
                      fontWeight: '700'
                    }}>
                      {severityInfo.label}
                    </h3>
                    <p style={{ margin: 0, color: '#666', fontSize: '0.95rem' }}>
                      {severityInfo.desc}
                    </p>
                  </div>
                </div>

                <div style={{ marginTop: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontWeight: '600', color: '#555' }}>Confidence Level</span>
                    <span style={{ fontWeight: '700', color: severityInfo.color, fontSize: '1.1rem' }}>
                      {confidence?.toFixed(1)}%
                    </span>
                  </div>
                  <div style={{
                    width: '100%',
                    height: '12px',
                    background: '#e0e0e0',
                    borderRadius: '10px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${confidence}%`,
                      height: '100%',
                      background: severityInfo.color,
                      transition: 'width 1s ease',
                      borderRadius: '10px'
                    }} />
                  </div>
                </div>

                <div style={{
                  marginTop: '1.5rem',
                  padding: '1rem',
                  background: 'rgba(255,255,255,0.7)',
                  borderRadius: '0.75rem',
                  fontSize: '0.85rem',
                  color: '#666',
                  lineHeight: '1.5'
                }}>
                  <strong>⚠️ Disclaimer:</strong> This is an AI screening tool and should not replace professional medical diagnosis. Please consult with an ophthalmologist for accurate diagnosis and treatment.
                </div>
              </div>
            )}

            {/* Heatmap Visualization */}
            {heatmaps && (
              <div style={{
                marginTop: '2rem',
                padding: '2rem',
                background: 'white',
                borderRadius: '1.25rem',
                border: '2px solid #2196f3'
              }}>
                <h3 style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  marginBottom: '1.5rem',
                  color: '#2196f3'
                }}>
                  <Eye size={24} /> Model Attention Heatmap
                </h3>
                {heatmaps.phase1 && heatmaps.phase1.success && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ color: '#555', marginBottom: '0.75rem' }}>Phase 1: DR Detection</h4>
                    <img src={heatmaps.phase1.comparison_base64} alt="Phase 1 Heatmap"
                      style={{ width: '100%', borderRadius: '0.75rem', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  </div>
                )}
                {heatmaps.phase2 && heatmaps.phase2.success && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ color: '#555', marginBottom: '0.75rem' }}>Phase 2: Severity Classification</h4>
                    <img src={heatmaps.phase2.comparison_base64} alt="Phase 2 Heatmap"
                      style={{ width: '100%', borderRadius: '0.75rem', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  </div>
                )}
                {heatmaps.phase3 && heatmaps.phase3.success && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ color: '#555', marginBottom: '0.75rem' }}>Phase 3: Advanced Stage</h4>
                    <img src={heatmaps.phase3.comparison_base64} alt="Phase 3 Heatmap"
                      style={{ width: '100%', borderRadius: '0.75rem', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default DiabeticRetinopathyDetector;