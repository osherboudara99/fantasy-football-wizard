import { useMemo, useState } from 'react'

// Matches an "@" that starts a mention token currently being typed - i.e. the
// last "@" in the text with no whitespace between it and the cursor.
function activeMentionQuery(text) {
  const at = text.lastIndexOf('@')
  if (at === -1) return null
  const afterAt = text.slice(at + 1)
  if (/\s/.test(afterAt)) return null
  return { start: at, query: afterAt }
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// Derived from the message text itself (rather than tracked as separate state
// updated only on selection) so that manually editing or deleting an "@Name"
// token also removes it from what gets sent - selecting a mention and then
// backspacing over it must not leave a stale mention behind.
function mentionsStillInText(text, players) {
  return players.filter((name) => {
    const pattern = new RegExp(`(?<!\\w)@${escapeRegExp(name)}(?!\\w)`, 'i')
    return pattern.test(text)
  })
}

export default function ChatInput({ players, onSend, disabled }) {
  const [text, setText] = useState('')

  const mention = useMemo(() => activeMentionQuery(text), [text])
  const suggestions = useMemo(() => {
    if (!mention) return []
    const query = mention.query.toLowerCase()
    return players.filter((name) => name.toLowerCase().includes(query)).slice(0, 8)
  }, [mention, players])

  function selectMention(name) {
    const before = text.slice(0, mention.start)
    setText(`${before}@${name} `)
  }

  function handleSubmit(event) {
    event.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed, mentionsStillInText(trimmed, players))
    setText('')
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      {suggestions.length > 0 && (
        <ul className="mention-suggestions">
          {suggestions.map((name) => (
            <li key={name}>
              <button type="button" onClick={() => selectMention(name)}>
                {name}
              </button>
            </li>
          ))}
        </ul>
      )}
      <textarea
        value={text}
        placeholder="Ask about start/sit, trades, or any player... (@ to mention)"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
          }
        }}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !text.trim()}>
        {disabled ? 'Thinking…' : 'Send'}
      </button>
    </form>
  )
}
