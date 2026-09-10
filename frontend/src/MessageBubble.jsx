import DecisionCard from './DecisionCard'
import SourceList from './SourceList'

export default function MessageBubble({ role, content, sources, recommendation, context, error }) {
  const isUser = role === 'user'

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'} ${error ? 'message-error' : ''}`}>
      <p className="message-content">{content}</p>

      {recommendation && <DecisionCard {...recommendation} />}
      {!isUser && !error && <SourceList sources={sources} />}

      {!isUser && !error && context && (
        <details className="debug-view">
          <summary>Show injected context</summary>
          <pre>{context}</pre>
        </details>
      )}
    </div>
  )
}
