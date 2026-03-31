// web/src/api.js
// Dev: Vite proxy handles /api -> localhost:8000
// Prod: set VITE_API_BASE to backend URL (e.g. "http://gpu-server:8000")
const API_BASE = import.meta.env.VITE_API_BASE || ''

export async function setMemory(memoryText, nMem = null) {
  const res = await fetch(`${API_BASE}/api/memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ memory_text: memoryText, n_mem: nMem }),
  })
  if (!res.ok) throw new Error(`Memory API error: ${res.status}`)
  return res.json()
}

export async function* streamChat(messages, params = {}) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, params }),
  })
  if (!res.ok) throw new Error(`Chat API error: ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop()

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const data = JSON.parse(line.slice(6))
      if (data.done) return
      if (data.token !== undefined) yield data.token
    }
  }
}

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/api/health`)
  return res.json()
}
