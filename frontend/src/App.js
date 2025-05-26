import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import './App.css';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const chunks = useRef([]);

  useEffect(() => {
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#4CAF50',
      progressColor: '#2196F3',
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

  return (
    <div className="App">
      <h1>Environmental Sound Classifier</h1>

      <input type="file" accept="audio/*" onChange={handleFileChange} />

      <button onClick={toggleRecording}>
        {isRecording ? 'Stop Recording' : 'Record Audio'}
      </button>

      <div ref={waveformRef} className="waveform-container" />

      <button onClick={togglePlayPause} disabled={!audioFile}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>

      <button onClick={handleSubmit} disabled={!audioFile}>
        Classify Audio
      </button>

      {result && (
        <div>
          <h3>Result: {result.prediction}</h3>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}

      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default App;
