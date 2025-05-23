import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import './App.css';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const waveformRef = useRef(null); // Ref for WaveSurfer container
  const wavesurfer = useRef(null); // Ref for WaveSurfer instance

  // Initialize WaveSurfer when component mounts
  useEffect(() => {
    // Create WaveSurfer instance
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#4CAF50', // Green waveform
      progressColor: '#2196F3', // Blue progress color
      height: 100, // Waveform height in pixels
      barWidth: 2, // Width of waveform bars
      responsive: true, // Adjusts to container size
    });

    // Cleanup on component unmount
    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }
    };
  }, []);

  // Handle file upload and load into WaveSurfer
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setAudioFile(file);
    setResult(null); // Clear previous results
    setError(null); // Clear previous errors

    // Load audio into WaveSurfer
    if (file) {
      const audioUrl = URL.createObjectURL(file);
      wavesurfer.current.load(audioUrl);
      // Revoke URL after loading to free memory
      wavesurfer.current.on('ready', () => {
        URL.revokeObjectURL(audioUrl);
      });
    } else {
      wavesurfer.current.empty(); // Clear waveform if no file is selected
    }
  };

  // Handle audio classification
  const handleSubmit = async () => {
    if (!audioFile) {
      setError('Please select an audio file');
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
      <div ref={waveformRef} style={{ width: '100%', margin: '20px 0' }} />
      <button onClick={handleSubmit}>Classify Audio</button>
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