// Thin client over the FastAPI backend (../../api/main.py). No orchestration or
// business logic here - just request/response shaping for the two endpoints the
// UI needs.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function parseErrorDetail(response) {
  try {
    const body = await response.json()
    return body.detail || response.statusText
  } catch {
    return response.statusText
  }
}

export async function fetchPlayers() {
  const response = await fetch(`${API_BASE_URL}/players`)
  if (!response.ok) {
    throw new Error(await parseErrorDetail(response))
  }
  return response.json()
}

export async function fetchRecommendation({ players, week, question }) {
  const response = await fetch(`${API_BASE_URL}/recommendation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      players,
      week: week ?? null,
      question: question || null,
    }),
  })
  if (!response.ok) {
    throw new Error(await parseErrorDetail(response))
  }
  return response.json()
}
