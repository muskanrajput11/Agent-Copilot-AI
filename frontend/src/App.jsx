import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Styling file

function App() {
  const [prompt, setPrompt] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);

  const getSuggestions = async () => {
    if (!prompt) {
      alert("Please enter a customer query.");
      return;
    }
    
    setIsLoading(true);
    setSuggestions([]);
    setCopiedIndex(null); 
    
    try {
      const response = await axios.post('http://0.0.0.0:7860/generate-suggestions', {
        prompt: prompt
      });
      setSuggestions(response.data.suggestions);
    } catch (error) {
      console.error("Error fetching suggestions:", error);
      alert("Failed to get suggestions. Check the console and make sure the API server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index); 
    setTimeout(() => {
      setCopiedIndex(null);
    }, 2000);
  };

  return (
    <div className="container">
      <header className="header">
        <h1> Agent's Co-pilot</h1>
        <p>Enter a customer query to get AI-powered reply suggestions.</p>
      </header>
      
      <div className="main-content">
        <div className="input-area">
          <textarea
            rows="4"
            placeholder="Paste customer's query here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <button onClick={getSuggestions} disabled={isLoading}>
            {isLoading ? 'Generating...' : 'Get Suggestions'}
          </button>
        </div>
        
        <div className="suggestions-area">
          {isLoading && <div className="spinner"></div>}
          
          {!isLoading && suggestions.length === 0 && (
             <p style={{textAlign: 'center', color: '#777'}}>Suggestions will appear here...</p>
          )}

          {suggestions.length > 0 && (
            <div className="suggestions-list">
              <h3>Suggested Replies:</h3>
              {suggestions.map((suggestion, index) => (
                <div key={index} className="suggestion-card">
                  <p>{suggestion}</p>
                  <button 
                    className={`copy-btn ${copiedIndex === index ? 'copied' : ''}`}
                    onClick={() => handleCopy(suggestion, index)}
                  >
                    {copiedIndex === index ? 'Copied!' : 'Copy'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      
    </div>
  );
}

export default App;
