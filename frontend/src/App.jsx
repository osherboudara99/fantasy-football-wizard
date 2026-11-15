import { useEffect, useRef, useState } from 'react'
import { fetchChat, fetchPlayers } from './api'
import ChatInput from './ChatInput'
import MessageBubble from './MessageBubble'
import ScoringSettings, { loadStoredScoringRules } from './ScoringSettings'
import './App.css'

function App() {
  const [players, setPlayers] = useState([])
  const [playersError, setPlayersError] = useState(null)
  const [week, setWeek] = useState('')
  const [scoringRules, setScoringRules] = useState(loadStoredScoringRules())
  const [messages, setMessages] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    fetchPlayers()
      .then(setPlayers)
      .catch((err) => setPlayersError(err.message))
  }, [])

  useEffect(() => {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    scrollRef.current?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' })
  }, [messages])

  async function handleSend(message, mentionedPlayers) {
    const history = messages.map(({ role, content }) => ({ role, content }))
    setMessages((prev) => [...prev, { role: 'user', content: message }])
    setSubmitting(true)
    try {
      const response = await fetchChat({
        message,
        mentionedPlayers,
        week: week ? Number(week) : null,
        history,
        scoringRules,
      })
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          recommendation: response.recommendation,
          context: response.context,
        },
      ])
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'assistant', content: err.message, error: true }])
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="app">
      <header>
        <div className="header-title">
          <h1>🏈 Fantasy Football Wizard</h1>
          <p className="subtitle">Ask about start/sit, trades, or any player.</p>
          <p className="byline">
            Osher Boudara
            <a
              href="https://github.com/osherboudara99"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Osher Boudara on GitHub"
              className="byline-icon"
            >
              <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                  0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                  -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                  .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                  -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.03 2.2-.82
                  2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65
                  3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01
                  8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
              </svg>
            </a>
          </p>
        </div>
        <ScoringSettings onRulesChange={setScoringRules} />
        <label className="week-label" title="Optional — defaults to the upcoming week">
          Week
          <input
            type="number"
            min="1"
            max="22"
            placeholder="upcoming"
            value={week}
            onChange={(e) => setWeek(e.target.value)}
          />
        </label>
      </header>

      {playersError && <p className="error">Couldn't load players: {playersError}</p>}

      <div className="message-list">
        {messages.map((message, i) => (
          <MessageBubble key={i} {...message} />
        ))}
        <div ref={scrollRef} />
      </div>

      <ChatInput players={players} onSend={handleSend} disabled={submitting} />
    </div>
  )
}

export default App
