import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale } from 'chart.js';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

ChartJS.register(BarElement, CategoryScale, LinearScale);

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [duration, setDuration] = useState(0);

  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const chunks = useRef([]);

  useEffect(() => {
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#007bff',
      progressColor: '#6610f2',
      height: 100,
      barWidth: 2,
      responsive: true,
    });

    wavesurfer.current.on('play', () => setIsPlaying(true));
    wavesurfer.current.on('pause', () => setIsPlaying(false));
    wavesurfer.current.on('finish', () => setIsPlaying(false));

    return () => {
      wavesurfer.current.destroy();
    };
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setAudioFile(file);
    setResult(null);
    setError(null);
    setIsPlaying(false);

    if (file) {
      const audioUrl = URL.createObjectURL(file);
      wavesurfer.current.load(audioUrl);
      wavesurfer.current.on('ready', () => {
        setDuration(wavesurfer.current.getDuration().toFixed(2));
        URL.revokeObjectURL(audioUrl);
      });
    } else {
      wavesurfer.current.empty();
    }
  };

  const togglePlayPause = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream);
        setMediaRecorder(recorder);
        chunks.current = [];

        recorder.ondataavailable = (e) => chunks.current.push(e.data);

        recorder.onstop = () => {
          const blob = new Blob(chunks.current, { type: 'audio/webm' });
          const file = new File([blob], 'recorded_audio.webm', { type: 'audio/webm' });
          setAudioFile(file);

          const audioUrl = URL.createObjectURL(blob);
          wavesurfer.current.load(audioUrl);
          wavesurfer.current.on('ready', () => {
            setDuration(wavesurfer.current.getDuration().toFixed(2));
            URL.revokeObjectURL(audioUrl);
          });
        };

        recorder.start();
        setIsRecording(true);
      } catch (err) {
        console.error('Recording error:', err);
        setError('Microphone access denied or not supported.');
      }
    }
  };

  const handleSubmit = async () => {
    if (!audioFile) {
      setError('Please select or record an audio file');
      return;
    }

    const formData = new FormData();
    formData.append('audio', audioFile);

    try {
      const response = await axios.post('http://localhost:5000/classify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
      setError(null);
    } catch (err) {
      setError('Error classifying audio: ' + err.message);
      setResult(null);
    }
  };

  const downloadRecordedAudio = () => {
    if (!audioFile) return;
    const url = URL.createObjectURL(audioFile);
    const a = document.createElement('a');
    a.href = url;
    a.download = audioFile.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="App container py-5">
      <h1 className="text-center mb-4 text-primary fw-bold">Environmental Sound Classifier</h1>

      <div className="mb-4 text-center">
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileChange}
          className="form-control w-50 mx-auto"
        />
      </div>

      <div className="text-center mb-3">
        <button onClick={toggleRecording} className={`btn ${isRecording ? 'btn-danger' : 'btn-outline-danger'} me-2`}>
          {isRecording ? 'Stop Recording' : 'Record Audio'}
        </button>

        <button onClick={togglePlayPause} className="btn btn-outline-primary me-2" disabled={!audioFile}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>

        <button onClick={handleSubmit} className="btn btn-success me-2" disabled={!audioFile}>
          Classify Audio
        </button>

        <button onClick={downloadRecordedAudio} className="btn btn-secondary" disabled={!audioFile}>
          Download Audio
        </button>
      </div>

      <div ref={waveformRef} className="waveform-container mb-2" />

      {duration > 0 && (
        <p className="text-center text-muted">Duration: {duration} seconds</p>
      )}

      {audioFile && (
        <div className="text-center text-muted mt-3">
          <p><strong>File:</strong> {audioFile.name}</p>
          <p><strong>Size:</strong> {(audioFile.size / 1024).toFixed(2)} KB</p>
          <p><strong>Type:</strong> {audioFile.type}</p>
        </div>
      )}

      {result && (
        <>
          {/* Horizontal row with Table and Progress Bars */}
          <div className="d-flex justify-content-around align-items-start flex-wrap" style={{ gap: '30px' }}>
            {/* Table */}
            <div className="card shadow-sm p-4 flex-grow-1" style={{ minWidth: '300px', maxWidth: '400px' }}>
              <h3 className="mb-3 text-center">Model Predictions</h3>
              <div className="table-responsive" style={{ fontSize: '1.1rem' }}>
                <table className="table table-striped table-bordered text-center mb-0">
                  <thead className="table-dark">
                    <tr>
                      <th>Model</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result).map(([model, data]) => (
                      <tr key={model}>
                        <td>{model}</td>
                        <td>{data.prediction}</td>
                        <td>{(data.confidence * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Progress Bars */}
            <div className="flex-grow-1" style={{ minWidth: '300px', maxWidth: '400px' }}>
              <h5 className="text-center mb-3">Confidence Levels</h5>
              {Object.entries(result).map(([model, data]) => (
                <div key={model} className="mb-3">
                  <strong>{model}</strong>
                  <div className="progress" style={{ height: '30px' }}>
                    <div
                      className="progress-bar bg-info"
                      role="progressbar"
                      style={{ width: `${data.confidence * 100}%`, fontSize: '1rem' }}
                      aria-valuenow={data.confidence * 100}
                      aria-valuemin="0"
                      aria-valuemax="100"
                    >
                      {(data.confidence * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Bar Chart below */}
          <div className="mt-5" style={{ maxWidth: '850px', margin: '0 auto' }}>
            <h5 className="text-center mb-3">Confidence Comparison Chart</h5>
            <div style={{ width: '100%', height: '400px' }}>
              <Bar
                data={{
                  labels: Object.keys(result),
                  datasets: [
                    {
                      label: 'Confidence (%)',
                      data: Object.values(result).map((data) => data.confidence * 100),
                      backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    },
                  ],
                }}
                options={{
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                    },
                  },
                }}
              />
            </div>
          </div>
        </>
      )}

      {error && <p className="text-danger text-center mt-3">{error}</p>}
    </div>
  );
}

export default App;
