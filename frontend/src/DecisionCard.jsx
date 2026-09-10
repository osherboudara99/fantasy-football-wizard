function ConfidenceBar({ confidence }) {
  const pct = Math.round(confidence * 100)
  return (
    <div className="confidence">
      <div className="confidence-label">
        <span>Confidence</span>
        <span>{pct}%</span>
      </div>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function DecisionCard({ start, bench, confidence, key_factors, risk_factors }) {
  return (
    <div className="decision-card">
      <div className="verdict">
        <div className="verdict-start">
          <span className="verdict-tag">START</span>
          <span className="verdict-name">{start}</span>
        </div>
        <div className="verdict-bench">
          <span className="verdict-tag">BENCH</span>
          <span className="verdict-name">{bench}</span>
        </div>
      </div>

      <ConfidenceBar confidence={confidence} />

      {key_factors?.length > 0 && (
        <div className="factors">
          <h3>Key factors</h3>
          <ul>
            {key_factors.map((factor) => (
              <li key={factor}>{factor}</li>
            ))}
          </ul>
        </div>
      )}

      {risk_factors?.length > 0 && (
        <div className="factors risk">
          <h3>Risk factors</h3>
          <ul>
            {risk_factors.map((factor) => (
              <li key={factor}>{factor}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
