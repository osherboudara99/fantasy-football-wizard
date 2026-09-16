export default function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <details className="source-list">
      <summary>Sources ({sources.length})</summary>
      <ul>
        {sources.map((item) => (
          <li key={item.link}>
            <a href={item.link} target="_blank" rel="noreferrer">
              {item.title}
            </a>
            {item.source && <span className="source-meta"> — {item.source}</span>}
            {item.published_at && (
              <span className="source-meta">
                {' '}
                ({new Date(item.published_at).toLocaleDateString()})
              </span>
            )}
          </li>
        ))}
      </ul>
    </details>
  )
}
