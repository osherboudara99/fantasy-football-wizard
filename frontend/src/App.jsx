import { useEffect, useRef, useState } from 'react'
import { fetchChat, fetchPlayers } from './api'
import ChatInput from './ChatInput'
import MessageBubble from './MessageBubble'
import ScoringSettings, { loadStoredScoringRules } from './ScoringSettings'
import SocialLinks from './SocialLinks'
import './App.css'

const SUGGESTED_QUESTIONS = [
  'Should I start Christian McCaffrey this week?',
  'Compare Puka Nacua and Nico Collins for my flex',
  "Is now a good time to trade for Ja'Marr Chase?",
]

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
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">🏈</span>
          <div className="brand-text">
            <h1>Fantasy Football Wizard</h1>
            <p className="subtitle">Ask about start/sit, trades, or any player.</p>
          </div>
        </div>
        <div className="header-controls">
          <ScoringSettings onRulesChange={setScoringRules} />
          <label className="week-label" title="Optional — defaults to the upcoming week">
            Week
            <input
              type="number"
              min="1"
              max="22"
              placeholder="auto"
              value={week}
              onChange={(e) => setWeek(e.target.value)}
            />
          </label>
        </div>
      </header>

      {playersError && <p className="error">Couldn't load players: {playersError}</p>}

      <div className="message-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <span className="empty-state-mark" aria-hidden="true">🏈</span>
            <h2>Ask your first question</h2>
            <p>Get a start/sit call backed by real stats, news, and the reasoning behind it.</p>
            <div className="suggestion-chips">
              {SUGGESTED_QUESTIONS.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => handleSend(question, [])}
                  disabled={submitting}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message, i) => (
          <MessageBubble key={i} {...message} />
        ))}
        <div ref={scrollRef} />
      </div>

      <ChatInput players={players} onSend={handleSend} disabled={submitting} />

      <footer className="app-footer">
        <span>Osher Boudara</span>
        <SocialLinks />
      </footer>
    </div>
  )
}

export default App
