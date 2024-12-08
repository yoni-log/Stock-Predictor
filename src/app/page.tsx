"use client";
import { useState } from "react";

export default function HomePage() {
  const [ticker, setTicker] = useState("");
  const [period, setPeriod] = useState("");
  const [model, setModel] = useState("");
  const [prediction, setPrediction] = useState<any>(null);

  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, period, model }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setPrediction({ error: "Failed to fetch prediction. Please try again later." });
    } finally {
      setLoading(false);
    }
  };

  return (
      <div style={{ maxWidth: "400px", margin: "40px auto", fontFamily: "Arial, sans-serif" }}>
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "15px" }}>
          <div>
            <label style={{ display: "block", fontWeight: "bold", marginBottom: "5px" }}>Ticker:</label>
            <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                required
                style={{ width: "100%", padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
                placeholder="e.g. AAPL"
            />
          </div>
          <div>
            <label style={{ display: "block", fontWeight: "bold", marginBottom: "5px" }}>Period:</label>
            <input
                type="text"
                value={period}
                onChange={(e) => setPeriod(e.target.value)}
                required
                style={{ width: "100%", padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
                placeholder="e.g. 1y"
            />
          </div>
          <div>
            <label style={{ display: "block", fontWeight: "bold", marginBottom: "5px" }}>Model:</label>
            <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                required
                style={{ width: "100%", padding: "8px", borderRadius: "4px", border: "1px solid #ccc" }}
            >
              <option value="">--Select a model--</option>
              <option value="LSTM">LSTM</option>
              <option value="Random Forest">Random Forest</option>
              <option value="Linear Regression">Linear Regression</option>
              <option value="Gradient Boosting">Gradient Boosting</option>
            </select>
          </div>
          <button
              type="submit"
              disabled={loading}
              style={{
                padding: "10px",
                borderRadius: "4px",
                fontWeight: "bold",
                border: "none",
                background: loading ? "#ccc" : "#0070f3",
                color: "#fff",
                cursor: loading ? "not-allowed" : "pointer",
              }}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>
        {prediction && (
            <div style={{ marginTop: "20px", background: "#f9f9f9", padding: "15px", borderRadius: "5px" }}>
              <h3 style={{ marginTop: 0 }}>Prediction Result</h3>
              <pre style={{ whiteSpace: "pre-wrap", wordWrap: "break-word" }}>
            {JSON.stringify(prediction, null, 2)}
          </pre>
            </div>
        )}
      </div>
  );
}
