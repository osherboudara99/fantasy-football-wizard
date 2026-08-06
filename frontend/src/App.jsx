import { useEffect, useState } from 'react'
import { fetchPlayers, fetchRecommendation } from './api'
import ResultCard from './ResultCard'
import './App.css'

function App() {
  const [players, setPlayers] = useState([])
  const [playersError, setPlayersError] = useState(null)
  const [playerA, setPlayerA] = useState('')
  const [playerB, setPlayerB] = useState('')
  const [week, setWeek] = useState('')
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState(null)

  useEffect(() => {
    fetchPlayers()
      .then(setPlayers)
      .catch((err) => setPlayersError(err.message))
  }, [])

  const canSubmit = playerA && playerB && playerA !== playerB && !submitting

  async function handleSubmit(event) {
    event.preventDefault()
    if (!canSubmit) return

    setSubmitting(true)
    setSubmitError(null)
    setResult(null)
    try {
      const recommendation = await fetchRecommendation({
        players: [playerA, playerB],
        week: week ? Number(week) : null,
        question: question.trim() || null,
      })
      setResult(recommendation)
    } catch (err) {
      setSubmitError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Fantasy Football Wizard</h1>
        <p className="subtitle">Start/sit decisions, explained.</p>
      </header>

      {playersError && (
        <p className="error">Couldn't load players: {playersError}</p>
      )}

      <form onSubmit={handleSubmit}>
        <div className="player-row">
          <label>
            Player A
            <select value={playerA} onChange={(e) => setPlayerA(e.target.value)}>
              <option value="">Select a player</option>
              {players.map((name) => (
                <option key={name} value={name} disabled={name === playerB}>
                  {name}
                </option>
              ))}
            </select>
          </label>

          <span className="vs">vs</span>

          <label>
            Player B
            <select value={playerB} onChange={(e) => setPlayerB(e.target.value)}>
              <option value="">Select a player</option>
              {players.map((name) => (
                <option key={name} value={name} disabled={name === playerA}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>

        <label>
          Question <span className="optional">(optional)</span>
          <input
            type="text"
            value={question}
            maxLength={500}
            placeholder={
              playerA && playerB
                ? `Who should I start, ${playerA} or ${playerB}?`
                : 'Who should I start?'
            }
            onChange={(e) => setQuestion(e.target.value)}
          />
        </label>

        <label className="week-label">
          Week <span className="optional">(optional, defaults to upcoming)</span>
          <input
            type="number"
            min="1"
            max="22"
            value={week}
            onChange={(e) => setWeek(e.target.value)}
          />
        </label>

        <button type="submit" disabled={!canSubmit}>
          {submitting ? 'Thinking…' : 'Get recommendation'}
        </button>
      </form>

      {submitError && <p className="error">{submitError}</p>}

      {result && <ResultCard result={result} />}
    </div>
  )
}

export default App
