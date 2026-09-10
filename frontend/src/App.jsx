import { useEffect, useRef, useState } from 'react'
import { fetchChat, fetchPlayers } from './api'
import ChatInput from './ChatInput'
import MessageBubble from './MessageBubble'
import './App.css'

function App() {
  const [players, setPlayers] = useState([])
  const [playersError, setPlayersError] = useState(null)
  const [week, setWeek] = useState('')
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
        <h1>Fantasy Football Wizard</h1>
        <p className="subtitle">Ask about start/sit, trades, or any player.</p>
        <label className="week-label">
          Week <span className="optional">(optional, defaults to upcoming)</span>
          <input type="number" min="1" max="22" value={week} onChange={(e) => setWeek(e.target.value)} />
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
