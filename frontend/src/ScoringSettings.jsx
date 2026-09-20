import { useState } from 'react'
import { fetchScoringRules } from './api'

const STORAGE_RULES_KEY = 'ffw_scoring_rules'
const STORAGE_BASE_KEY = 'ffw_scoring_base'
const LABELS = { ppr: 'PPR', half_ppr: 'Half-PPR', standard: 'Standard', custom: 'Custom' }

export function loadStoredScoringRules() {
  try {
    const raw = localStorage.getItem(STORAGE_RULES_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

function loadStoredScoringBase() {
  try {
    return localStorage.getItem(STORAGE_BASE_KEY) || 'ppr'
  } catch {
    return 'ppr'
  }
}

function ScoringSettings({ onRulesChange }) {
  // `activeBase` is the tier actually applied (what onRulesChange last fired
  // for) - it drives the highlighted button. `formBase` only controls which
  // panel is expanded. Clicking "Custom" must not highlight it as active
  // until a save actually succeeds - otherwise the UI would claim Custom is
  // in effect while chats keep using the previously saved preset underneath.
  const [activeBase, setActiveBase] = useState(loadStoredScoringBase())
  const [formBase, setFormBase] = useState(activeBase)
  const [baseHint, setBaseHint] = useState('ppr')
  const [description, setDescription] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  async function save(nextBase) {
    setSaving(true)
    setError(null)
    try {
      const { rules } = await fetchScoringRules({
        base: nextBase,
        baseHint,
        customDescription: nextBase === 'custom' ? description : null,
      })
      localStorage.setItem(STORAGE_RULES_KEY, JSON.stringify(rules))
      localStorage.setItem(STORAGE_BASE_KEY, nextBase)
      setActiveBase(nextBase)
      setFormBase(nextBase)
      onRulesChange(rules)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="scoring-settings">
      <span className="scoring-settings-label">Scoring</span>
      <div className="scoring-options">
        {Object.keys(LABELS).map((option) => (
          <button
            key={option}
            type="button"
            className={activeBase === option ? 'scoring-option active' : 'scoring-option'}
            onClick={() => (option === 'custom' ? setFormBase('custom') : save(option))}
            disabled={saving}
          >
            {LABELS[option]}
          </button>
        ))}
      </div>
      {formBase === 'custom' && (
        <div className="scoring-custom">
          <select value={baseHint} onChange={(e) => setBaseHint(e.target.value)}>
            <option value="ppr">Based on PPR</option>
            <option value="half_ppr">Based on Half-PPR</option>
            <option value="standard">Based on Standard</option>
          </select>
          <textarea
            placeholder="Describe how your league differs, e.g. 'catches are 0.5 points'"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <button type="button" onClick={() => save('custom')} disabled={saving || !description}>
            Save
          </button>
          {activeBase !== 'custom' && (
            <p className="scoring-pending-note">
              Not yet applied - click Save to switch to Custom scoring.
            </p>
          )}
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  )
}

export default ScoringSettings
