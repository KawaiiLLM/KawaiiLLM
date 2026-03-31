# KawaiiLLM Inference Web App Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web-based inference app with FastAPI backend + React frontend that faithfully replicates the training forward pass for generation, with multi-turn conversation and streaming output.

**Architecture:** Python inference engine wraps `KawaiiLLMModel` and replicates `encode_context()` + LLM input assembly from `model.py` exactly. FastAPI serves SSE streaming. React frontend reuses Claude.ai warm color design with a collapsible memory context panel (方案A) above the chat composer.

**Tech Stack:** Python (FastAPI, uvicorn, transformers), React 19, Vite, Tailwind CSS 4

---

## File Structure

```
src/inference/
├── __init__.py            # Empty
├── engine.py              # KawaiiInferenceEngine: load, encode memory, generate
└── server.py              # FastAPI app with SSE streaming

web/
├── package.json           # React + Vite + Tailwind deps
├── vite.config.js         # Vite config with /api proxy
├── index.html             # HTML entry
└── src/
    ├── main.jsx           # React entry
    ├── index.css          # Design tokens (Claude warm colors)
    ├── App.jsx            # Layout shell
    ├── api.js             # API client with SSE async generator
    ├── components/
    │   ├── Sidebar.jsx    # Collapsible sidebar with gen params
    │   ├── MemoryPanel.jsx # Collapsible memory context textarea
    │   ├── ChatInput.jsx  # Composer with dynamic textarea
    │   └── MessageBubble.jsx # User/assistant message rendering
    └── icons/
        └── index.jsx      # SVG icon components

scripts/
└── serve.sh               # Launch inference server
```

---

## Critical Reference Files

| File | Why |
|:-----|:----|
| `src/train/model.py:271-367` | `encode_context()` — MemE encoding pipeline to replicate exactly |
| `src/train/model.py:488-589` | `forward()` non-NTP path — LLM input assembly to replicate exactly |
| `src/train/model.py:621-662` | `from_checkpoint()` — checkpoint loading |
| `src/train/model.py:136-175` | `set_special_token_ids()` — special token registration |
| `src/train/train.py:58-126` | Tokenizer setup, special tokens, embedding resize sequence |
| `C3-Context-Cascade-Compression/C3-master/C3/model/C3.py:249-307` | `prepare_inputs_for_generation` pattern for `inputs_embeds` |

---

## Task 1: Inference Engine

**Files:**
- Create: `src/inference/__init__.py`
- Create: `src/inference/engine.py`

This is the most critical piece. The engine must replicate training `forward()` exactly for B=1 inference.

- [ ] **Step 1: Create empty `__init__.py`**

```python
# src/inference/__init__.py
```

- [ ] **Step 2: Write `KawaiiInferenceEngine.__init__` — model loading**

Replicate the exact loading sequence from `train.py:58-126`:

```python
# src/inference/engine.py
"""KawaiiLLM inference engine — faithful replica of training forward pass."""

import logging
import os
from threading import Thread
from typing import Iterator, Optional

import torch
from transformers import AutoTokenizer, TextIteratorStreamer

from src.train.model import KawaiiLLMModel, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


class KawaiiInferenceEngine:
    """Load a KawaiiLLM checkpoint and generate text with memory context."""

    def __init__(
        self,
        checkpoint_dir: str,
        num_mem_tokens: int = 128,
        device: str = "cuda",
        attn_implementation: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_mem_tokens = num_mem_tokens

        # --- Tokenizer (same sequence as train.py:58-74) ---
        llm_dir = os.path.join(checkpoint_dir, "llm")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        logger.info("Registered %d special tokens: %s", num_added, SPECIAL_TOKENS)

        # --- Model (same sequence as train.py:93-126) ---
        self.model = KawaiiLLMModel.from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            num_mem_tokens=num_mem_tokens,
            freeze_meme=True,
            freeze_llm=True,
            freeze_projector=True,
            attn_implementation=attn_implementation,
        )
        self.model.meme.resize_token_embeddings(len(self.tokenizer))
        self.model.llm.resize_token_embeddings(len(self.tokenizer))

        special_token_ids = {
            tok: self.tokenizer.convert_tokens_to_ids(tok)
            for tok in SPECIAL_TOKENS
        }
        special_token_ids["pad_token_id"] = self.tokenizer.pad_token_id
        self.model.set_special_token_ids(special_token_ids)

        self.model.to(self.device).eval()
        logger.info("Model loaded on %s", self.device)

        # Monkey-patch prepare_inputs_for_generation if needed
        self._patch_prepare_inputs()

        # Cached memory state
        self._memory_prefix_embeds: Optional[torch.Tensor] = None  # [1, n_mem+2, 4096]
        self._memory_prefix_mask: Optional[torch.Tensor] = None    # [1, n_mem+2]
        self._active_n_mem: int = 0
```

- [ ] **Step 3: Write `_patch_prepare_inputs` — ensure `inputs_embeds` works with `generate()`**

Following C3.py:249-307 pattern. Qwen3's native method may not handle `inputs_embeds` correctly on the first generation step:

```python
    def _patch_prepare_inputs(self):
        """Monkey-patch LLM's prepare_inputs_for_generation for inputs_embeds support."""
        original = self.model.llm.prepare_inputs_for_generation

        def patched(input_ids, past_key_values=None, attention_mask=None,
                    inputs_embeds=None, **kwargs):
            # First step: use inputs_embeds if provided and no cache yet
            if inputs_embeds is not None and past_key_values is None:
                position_ids = None
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                return {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache", True),
                }
            # Subsequent steps: delegate to original (uses input_ids + KV cache)
            return original(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=None,  # force None so original uses input_ids
                **kwargs,
            )

        self.model.llm.prepare_inputs_for_generation = patched
```

- [ ] **Step 4: Write `set_memory` — MemE encoding + projection (replicates model.py:271-367, 496-520)**

This is the exact replica of `encode_context()` + projection + residual + boundary token assembly:

```python
    @torch.no_grad()
    def set_memory(self, memory_text: str, n_mem: Optional[int] = None) -> None:
        """Encode memory_text through MemE → Projector, cache prefix embeddings."""
        if not memory_text.strip():
            self._memory_prefix_embeds = None
            self._memory_prefix_mask = None
            self._active_n_mem = 0
            logger.info("Memory cleared")
            return

        n_mem = n_mem or self.num_mem_tokens
        self._active_n_mem = n_mem

        # Tokenize context (same as collator: MemE uses left-padding)
        ctx_enc = self.tokenizer(
            memory_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096,
        )
        context_ids = ctx_enc["input_ids"].to(self.device)              # [1, L]
        context_mask = ctx_enc["attention_mask"].to(self.device)        # [1, L]

        # --- Replicate encode_context (model.py:306-361) ---
        meme_embed = self.model.meme.get_input_embeddings()
        text_embeds = meme_embed(context_ids)                           # [1, L, 2560]

        mem_embeds = self.model.mem_embeddings.weight[:n_mem]           # [n_mem, 2560]
        mem_embeds = mem_embeds.unsqueeze(0)                            # [1, n_mem, 2560]

        combined = torch.cat([text_embeds, mem_embeds], dim=1)          # [1, L+n_mem, 2560]

        # Extend attention mask with ones for MEM tokens (model.py:319-326)
        extra_mask = torch.ones(1, n_mem, dtype=context_mask.dtype, device=self.device)
        extended_mask = torch.cat([context_mask, extra_mask], dim=1)    # [1, L+n_mem]

        # Fix position 0 to prevent NaN under causal mask (model.py:337)
        extended_mask[:, 0] = 1

        # Build position_ids (model.py:340-341)
        position_ids = extended_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(extended_mask == 0, 0)

        # Run MemE (model.py:350-356)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            outputs = self.model.meme(
                inputs_embeds=combined,
                attention_mask=extended_mask,
                position_ids=position_ids,
            )

        mem_hidden = outputs.last_hidden_state[:, -n_mem:, :]          # [1, n_mem, 2560]

        # --- Replicate projection (model.py:496-516) ---
        projected = self.model.projector(mem_hidden)                   # [1, n_mem, 4096]

        # Add pad_embed residual base (model.py:511-516)
        llm_embed = self.model.llm.get_input_embeddings()
        pad_embed_vec = llm_embed(self.model._pad_id_buf).detach().squeeze(0)  # [4096]
        projected = projected + pad_embed_vec.unsqueeze(0).unsqueeze(0)

        # --- Assemble prefix: [<mem>] [projected] [</mem>] (model.py:519-537) ---
        mem_start_emb = llm_embed(self.model._mem_id_buf).squeeze(0)   # [4096]
        mem_end_emb = llm_embed(self.model._mem_end_id_buf).squeeze(0) # [4096]

        # For B=1 no left-padding needed
        prefix_embeds = torch.cat([
            mem_start_emb.unsqueeze(0),     # [1, 4096]
            projected.squeeze(0),           # [n_mem, 4096]
            mem_end_emb.unsqueeze(0),       # [1, 4096]
        ], dim=0).unsqueeze(0)              # [1, n_mem+2, 4096]

        prefix_mask = torch.ones(1, n_mem + 2, dtype=torch.long, device=self.device)

        self._memory_prefix_embeds = prefix_embeds
        self._memory_prefix_mask = prefix_mask
        logger.info("Memory encoded: %d tokens → %d mem tokens", context_ids.shape[1], n_mem)
```

- [ ] **Step 5: Write `generate` — LLM input assembly + HF generate() with streaming**

Replicates model.py:557-589 for building the final LLM input, then calls `generate()`:

```python
    def _build_llm_inputs(self, conversation_ids: list[int]) -> dict:
        """Build inputs_embeds, attention_mask, position_ids for LLM generation.

        Replicates model.py forward() lines 557-581 for B=1.
        """
        llm_embed = self.model.llm.get_input_embeddings()
        target_ids = torch.tensor([conversation_ids], dtype=torch.long, device=self.device)
        target_embeds = llm_embed(target_ids)  # [1, T, 4096]

        if self._memory_prefix_embeds is not None:
            # Concat prefix + target (model.py:557)
            inputs_embeds = torch.cat(
                [self._memory_prefix_embeds, target_embeds], dim=1
            )
            attention_mask = torch.cat(
                [self._memory_prefix_mask, torch.ones_like(target_ids)], dim=1
            )
        else:
            # No memory: pure NTP path (model.py:479-486)
            inputs_embeds = target_embeds
            attention_mask = torch.ones_like(target_ids)

        # Build position_ids (model.py:579-581)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        return {
            "inputs_embeds": inputs_embeds.to(dtype=self.model.llm.dtype),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format multi-turn messages into a single string for tokenization.

        Qwen3-8B-Base is a base model (no chat template), so we use a simple format.
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        # Append prompt for assistant response
        parts.append("Assistant:")
        return "\n".join(parts)

    def _truncate_conversation(self, conversation_ids: list[int], max_len: int) -> list[int]:
        """Truncate from the left (oldest turns) to fit within max_len."""
        prefix_len = (self._active_n_mem + 2) if self._memory_prefix_embeds is not None else 0
        available = max_len - prefix_len - 1  # reserve 1 for at least one generated token
        if len(conversation_ids) > available:
            conversation_ids = conversation_ids[-available:]
        return conversation_ids

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate a response given conversation messages.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}.
            stream: If True, returns an Iterator yielding token strings.
        """
        conversation_text = self._format_conversation(messages)
        conversation_ids = self.tokenizer.encode(
            conversation_text, add_special_tokens=False
        )
        conversation_ids = self._truncate_conversation(conversation_ids, max_len=32768)

        llm_inputs = self._build_llm_inputs(conversation_ids)

        gen_kwargs = {
            **llm_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-7),
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if stream:
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs["streamer"] = streamer

            thread = Thread(
                target=self.model.llm.generate, kwargs=gen_kwargs, daemon=True
            )
            thread.start()
            return streamer  # caller iterates: for token in streamer: ...
        else:
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                output_ids = self.model.llm.generate(**gen_kwargs)
            # Strip prompt from output
            prompt_len = llm_inputs["inputs_embeds"].shape[1]
            generated = output_ids[0, prompt_len:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
```

- [ ] **Step 6: Verify engine loads and generates**

```bash
cd /Users/zhaoqixuan/Projects/KawaiiLLM
python -c "
from src.inference.engine import KawaiiInferenceEngine
engine = KawaiiInferenceEngine('output/kawaii_v1', device='cuda')
engine.set_memory('这是一段测试记忆文本。')
result = engine.generate([{'role': 'user', 'content': '你好'}], max_new_tokens=32, stream=False)
print('Generated:', result)
"
```

Expected: Loads without error, prints generated text (may be nonsensical if model is undertrained, but must not NaN or crash).

- [ ] **Step 7: Commit**

```bash
git add src/inference/__init__.py src/inference/engine.py
git commit -m "feat(inference): add KawaiiInferenceEngine with training-consistent forward pass"
```

---

## Task 2: FastAPI Server with SSE Streaming

**Files:**
- Create: `src/inference/server.py`
- Create: `scripts/serve.sh`

- [ ] **Step 1: Write FastAPI server**

```python
# src/inference/server.py
"""FastAPI server for KawaiiLLM inference with SSE streaming."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .engine import KawaiiInferenceEngine

logger = logging.getLogger(__name__)

engine: KawaiiInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "output/kawaii_v1")
    device = os.environ.get("DEVICE", "cuda")
    attn_impl = os.environ.get("ATTN_IMPL", None)
    num_mem = int(os.environ.get("NUM_MEM_TOKENS", "128"))

    logger.info("Loading model from %s on %s ...", checkpoint_dir, device)
    engine = KawaiiInferenceEngine(
        checkpoint_dir=checkpoint_dir,
        num_mem_tokens=num_mem,
        device=device,
        attn_implementation=attn_impl,
    )
    logger.info("Model ready.")
    yield
    engine = None


app = FastAPI(title="KawaiiLLM Inference", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/api/memory")
async def set_memory(request: Request):
    body = await request.json()
    memory_text = body.get("memory_text", "")
    n_mem = body.get("n_mem", None)
    engine.set_memory(memory_text, n_mem=n_mem)
    return {"status": "ok", "n_mem": engine._active_n_mem}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    params = body.get("params", {})

    async def event_stream():
        loop = asyncio.get_event_loop()
        streamer = await loop.run_in_executor(
            None,
            lambda: engine.generate(messages, stream=True, **params),
        )
        for token_text in streamer:
            data = json.dumps({"token": token_text}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 2: Write launch script**

```bash
#!/bin/bash
# scripts/serve.sh — Launch KawaiiLLM inference server
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-output/kawaii_v1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
NUM_MEM_TOKENS="${NUM_MEM_TOKENS:-128}"

echo "Starting KawaiiLLM inference server..."
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Device:     ${DEVICE}"
echo "  Listen:     ${HOST}:${PORT}"

CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
DEVICE="${DEVICE}" \
NUM_MEM_TOKENS="${NUM_MEM_TOKENS}" \
python -m uvicorn src.inference.server:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1 \
  --log-level info
```

- [ ] **Step 3: Verify server endpoints with curl**

```bash
bash scripts/serve.sh &
sleep 30  # wait for model load

# Health
curl http://localhost:8000/api/health

# Set memory
curl -X POST http://localhost:8000/api/memory \
  -H "Content-Type: application/json" \
  -d '{"memory_text": "这是记忆上下文。"}'

# Chat (streaming)
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}], "params": {"max_new_tokens": 64}}'
```

Expected: Health returns 200. Memory returns `{"status": "ok"}`. Chat streams SSE `data: {"token": "..."}` lines.

- [ ] **Step 4: Commit**

```bash
git add src/inference/server.py scripts/serve.sh
git commit -m "feat(inference): add FastAPI server with SSE streaming"
```

---

## Task 3: Frontend Project Setup

**Files:**
- Create: `web/package.json`
- Create: `web/vite.config.js`
- Create: `web/index.html`
- Create: `web/src/main.jsx`
- Create: `web/src/index.css`
- Create: `web/src/App.jsx`

- [ ] **Step 1: Initialize frontend project**

`web/package.json`:
```json
{
  "name": "kawaii-llm-web",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.2.0",
    "react-dom": "^19.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^5.1.1",
    "@tailwindcss/vite": "^4.2.1",
    "tailwindcss": "^4.2.1",
    "vite": "^7.3.1"
  }
}
```

`web/vite.config.js`:
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

`web/index.html`:
```html
<!doctype html>
<html lang="zh">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>KawaiiLLM</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 2: Write design tokens and entry files**

`web/src/index.css` — Claude.ai warm color palette:
```css
@import "tailwindcss";

@theme {
  --color-bg-100: #faf9f5;
  --color-bg-200: #f0eee6;
  --color-bg-300: #e8e6dc;
  --color-sidebar: #f5f4ed;
  --color-accent: #d97757;
  --color-accent-hover: #c9654a;
  --color-text-100: #141413;
  --color-text-200: #3d3d3a;
  --color-text-300: #73726c;
  --color-border-100: rgba(31, 30, 29, 0.15);
  --color-border-200: rgba(31, 30, 29, 0.25);
  --shadow-composer: 0 4px 20px 0 rgba(0, 0, 0, 0.035),
    0 0 0 0.5px rgba(31, 30, 29, 0.15);
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  --font-serif: Georgia, "Times New Roman", Times, serif;
}

body {
  background-color: var(--color-bg-100);
  color: var(--color-text-100);
  font-family: var(--font-sans);
  margin: 0;
}

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--color-bg-300); border-radius: 3px; }
```

`web/src/main.jsx`:
```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

`web/src/App.jsx` — shell (placeholder, wired in Task 5):
```jsx
export default function App() {
  return (
    <div className="h-screen flex bg-bg-100">
      <div className="flex-1 flex items-center justify-center text-text-300">
        KawaiiLLM
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Install dependencies and verify**

```bash
cd web && npm install && npm run dev
```

Expected: Vite starts, browser shows warm beige background with "KawaiiLLM" centered.

- [ ] **Step 4: Commit**

```bash
git add web/
git commit -m "feat(web): scaffold React + Vite + Tailwind with Claude warm colors"
```

---

## Task 4: API Client

**Files:**
- Create: `web/src/api.js`

- [ ] **Step 1: Write API client with SSE async generator**

```javascript
// web/src/api.js
const API_BASE = ''

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
```

- [ ] **Step 2: Commit**

```bash
git add web/src/api.js
git commit -m "feat(web): add API client with SSE streaming"
```

---

## Task 5: Chat UI Components

**Files:**
- Create: `web/src/icons/index.jsx`
- Create: `web/src/components/MessageBubble.jsx`
- Create: `web/src/components/ChatInput.jsx`
- Create: `web/src/components/MemoryPanel.jsx`
- Create: `web/src/components/Sidebar.jsx`
- Modify: `web/src/App.jsx`

- [ ] **Step 1: Write icon components**

`web/src/icons/index.jsx` — minimal SVG icon set:

```jsx
export function SendIcon({ className = 'w-4 h-4' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
    </svg>
  )
}

export function ChevronIcon({ className = 'w-4 h-4', direction = 'down' }) {
  const rotation = { up: 180, down: 0, left: 90, right: -90 }[direction]
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" style={{ transform: `rotate(${rotation}deg)` }}>
      <path d="M6 9l6 6 6-6" />
    </svg>
  )
}

export function SidebarIcon({ className = 'w-5 h-5' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M9 3v18" />
    </svg>
  )
}

export function PlusIcon({ className = 'w-5 h-5' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 5v14M5 12h14" />
    </svg>
  )
}

export function MemoryIcon({ className = 'w-5 h-5' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 2a10 10 0 0 1 10 10 10 10 0 0 1-10 10A10 10 0 0 1 2 12 10 10 0 0 1 12 2z" />
      <path d="M12 6v6l4 2" />
    </svg>
  )
}

export function TrashIcon({ className = 'w-4 h-4' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
    </svg>
  )
}
```

- [ ] **Step 2: Write MessageBubble component**

`web/src/components/MessageBubble.jsx`:

```jsx
export default function MessageBubble({ role, content, isStreaming }) {
  if (role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[85%] bg-bg-200 rounded-[12px] px-4 py-2.5">
          <p className="text-[15px] leading-[1.5] whitespace-pre-wrap">{content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="mb-6">
      <div className="max-w-[672px]">
        <p className="font-serif text-[15px] leading-[1.6] whitespace-pre-wrap">
          {content}
          {isStreaming && (
            <span className="inline-block w-[2px] h-[1em] bg-text-100 ml-0.5 animate-pulse" />
          )}
        </p>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Write ChatInput component (composer)**

`web/src/components/ChatInput.jsx`:

```jsx
import { useState, useRef } from 'react'
import { SendIcon } from '../icons'

export default function ChatInput({ onSend, disabled, placeholder = 'Send a message...' }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  function handleSubmit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  function handleInput(e) {
    setValue(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }

  const hasText = value.trim().length > 0

  return (
    <div className="bg-white rounded-[20px]" style={{ boxShadow: 'var(--shadow-composer)' }}>
      <div className="m-3.5 flex flex-col gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="w-full resize-none bg-transparent text-[16px] leading-[1.5]
                     placeholder:text-text-300 focus:outline-none"
          style={{ maxHeight: '200px' }}
        />
        <div className="flex justify-end">
          <button
            onClick={handleSubmit}
            disabled={!hasText || disabled}
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors
              ${hasText && !disabled
                ? 'bg-accent text-white hover:bg-accent-hover cursor-pointer'
                : 'bg-bg-300 text-text-300 cursor-not-allowed'}`}
          >
            <SendIcon />
          </button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Write MemoryPanel component**

`web/src/components/MemoryPanel.jsx`:

```jsx
import { useState } from 'react'
import { ChevronIcon, MemoryIcon } from '../icons'

export default function MemoryPanel({ memoryText, onMemoryChange, onApply, dirty }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="mb-3">
      {/* Toggle header */}
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 text-[13px] text-text-200
                   hover:text-text-100 transition-colors cursor-pointer"
      >
        <MemoryIcon className="w-4 h-4" />
        <span>Memory Context</span>
        {dirty && <span className="w-1.5 h-1.5 rounded-full bg-accent" />}
        <ChevronIcon className="w-3 h-3" direction={open ? 'up' : 'down'} />
      </button>

      {/* Collapsible content */}
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-in-out"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden min-h-0">
          <div className="px-3 pb-3">
            <textarea
              value={memoryText}
              onChange={(e) => onMemoryChange(e.target.value)}
              placeholder="Paste memory context text here... This text will be compressed through MemE and injected as memory tokens."
              className="w-full min-h-[100px] max-h-[300px] resize-y bg-bg-200 rounded-[12px]
                         px-3 py-2.5 text-[14px] leading-[1.5] placeholder:text-text-300
                         focus:outline-none border border-border-100"
            />
            <div className="flex justify-end mt-2">
              <button
                onClick={onApply}
                disabled={!dirty}
                className={`px-3 py-1 rounded-[8px] text-[13px] font-medium transition-colors
                  ${dirty
                    ? 'bg-accent text-white hover:bg-accent-hover cursor-pointer'
                    : 'bg-bg-300 text-text-300 cursor-not-allowed'}`}
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Write Sidebar component**

`web/src/components/Sidebar.jsx`:

```jsx
import { SidebarIcon, PlusIcon, TrashIcon } from '../icons'

export default function Sidebar({ open, onToggle, genParams, onParamsChange, onNewChat }) {
  return (
    <div
      className="h-full bg-sidebar border-r border-border-100 flex flex-col transition-[width] duration-300"
      style={{ width: open ? '300px' : '52px' }}
    >
      {/* Header */}
      <div className="p-3 flex items-center justify-between">
        {open && <span className="text-[14px] font-semibold text-text-100">KawaiiLLM</span>}
        <button
          onClick={onToggle}
          className="w-8 h-8 flex items-center justify-center rounded-[8px]
                     hover:bg-bg-200 transition-colors cursor-pointer"
        >
          <SidebarIcon />
        </button>
      </div>

      {/* New chat button */}
      <div className="px-3 mb-4">
        <button
          onClick={onNewChat}
          className={`flex items-center gap-2 rounded-[8px] hover:bg-bg-200
                      transition-colors cursor-pointer ${open ? 'px-3 py-2 w-full' : 'w-8 h-8 justify-center'}`}
        >
          <PlusIcon className="w-4 h-4 flex-shrink-0" />
          {open && <span className="text-[14px]">New Chat</span>}
        </button>
      </div>

      {/* Generation parameters (visible only when sidebar is open) */}
      {open && (
        <div className="flex-1 overflow-y-auto px-3">
          <div className="text-[12px] text-text-300 mb-3">Generation</div>

          <ParamSlider
            label="Temperature" value={genParams.temperature}
            min={0} max={2} step={0.05}
            onChange={(v) => onParamsChange({ ...genParams, temperature: v })}
          />
          <ParamSlider
            label="Top-p" value={genParams.top_p}
            min={0} max={1} step={0.05}
            onChange={(v) => onParamsChange({ ...genParams, top_p: v })}
          />
          <ParamSlider
            label="Max Tokens" value={genParams.max_new_tokens}
            min={64} max={4096} step={64}
            onChange={(v) => onParamsChange({ ...genParams, max_new_tokens: v })}
          />
          <ParamSlider
            label="Repetition Penalty" value={genParams.repetition_penalty}
            min={1.0} max={2.0} step={0.05}
            onChange={(v) => onParamsChange({ ...genParams, repetition_penalty: v })}
          />
        </div>
      )}
    </div>
  )
}

function ParamSlider({ label, value, min, max, step, onChange }) {
  return (
    <div className="mb-4">
      <div className="flex justify-between text-[13px] mb-1">
        <span className="text-text-200">{label}</span>
        <span className="text-text-300">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 bg-bg-300 rounded-full appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                   [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-accent"
      />
    </div>
  )
}
```

- [ ] **Step 6: Wire everything in App.jsx**

`web/src/App.jsx`:

```jsx
import { useState, useRef, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import MemoryPanel from './components/MemoryPanel'
import ChatInput from './components/ChatInput'
import MessageBubble from './components/MessageBubble'
import { setMemory as setMemoryApi, streamChat } from './api'

const DEFAULT_PARAMS = {
  temperature: 0.7,
  top_p: 0.9,
  max_new_tokens: 2048,
  repetition_penalty: 1.0,
}

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [messages, setMessages] = useState([])
  const [memoryText, setMemoryText] = useState('')
  const [appliedMemory, setAppliedMemory] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [genParams, setGenParams] = useState(DEFAULT_PARAMS)
  const scrollRef = useRef(null)

  const memoryDirty = memoryText !== appliedMemory

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  async function handleApplyMemory() {
    if (!memoryDirty) return
    await setMemoryApi(memoryText)
    setAppliedMemory(memoryText)
  }

  async function handleSend(userMessage) {
    if (isStreaming) return

    // Apply memory if dirty
    if (memoryDirty && memoryText.trim()) {
      await setMemoryApi(memoryText)
      setAppliedMemory(memoryText)
    }

    const newMessages = [...messages, { role: 'user', content: userMessage }]
    setMessages([...newMessages, { role: 'assistant', content: '' }])
    setIsStreaming(true)

    try {
      const assistantIdx = newMessages.length
      for await (const token of streamChat(newMessages, genParams)) {
        setMessages((prev) => {
          const updated = [...prev]
          updated[assistantIdx] = {
            ...updated[assistantIdx],
            content: updated[assistantIdx].content + token,
          }
          return updated
        })
      }
    } catch (err) {
      console.error('Stream error:', err)
      setMessages((prev) => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'assistant',
          content: updated[updated.length - 1].content + '\n\n[Error: ' + err.message + ']',
        }
        return updated
      })
    } finally {
      setIsStreaming(false)
    }
  }

  function handleNewChat() {
    setMessages([])
  }

  return (
    <div className="h-screen flex bg-bg-100">
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        genParams={genParams}
        onParamsChange={setGenParams}
        onNewChat={handleNewChat}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Messages area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-[672px] mx-auto">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-text-300">
                <p className="text-[24px] font-serif mb-2">KawaiiLLM</p>
                <p className="text-[14px]">Set memory context and start chatting</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <MessageBubble
                key={i}
                role={msg.role}
                content={msg.content}
                isStreaming={isStreaming && i === messages.length - 1 && msg.role === 'assistant'}
              />
            ))}
          </div>
        </div>

        {/* Bottom: memory panel + input */}
        <div className="border-t border-border-100 bg-bg-100">
          <div className="max-w-[672px] mx-auto px-4 py-3">
            <MemoryPanel
              memoryText={memoryText}
              onMemoryChange={setMemoryText}
              onApply={handleApplyMemory}
              dirty={memoryDirty}
            />
            <ChatInput
              onSend={handleSend}
              disabled={isStreaming}
              placeholder={messages.length === 0 ? 'Start a conversation...' : 'Reply...'}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 7: Verify UI renders and interacts**

```bash
cd web && npm run dev
```

Open browser: sidebar visible, memory panel toggles, input accepts text, send button activates.
(Without backend running, send will show an error — that's expected.)

- [ ] **Step 8: Commit**

```bash
git add web/src/
git commit -m "feat(web): add chat UI with memory panel, sidebar, and streaming"
```

---

## Task 6: End-to-End Integration and Verification

**Files:**
- No new files. Verify everything works together.

- [ ] **Step 1: Start backend**

```bash
CHECKPOINT_DIR=output/kawaii_v1 bash scripts/serve.sh
```

Wait for "Model ready." log message.

- [ ] **Step 2: Start frontend**

```bash
cd web && npm run dev
```

- [ ] **Step 3: Full flow test**

1. Open `http://localhost:5173`
2. Expand memory panel, paste a paragraph of text, click "Apply"
3. Type "你好" in the chat input, press Enter
4. Verify: tokens stream in one-by-one on the assistant message
5. Send a follow-up message to verify multi-turn works
6. Click "New Chat" to verify conversation clears
7. Toggle sidebar to verify collapse/expand

- [ ] **Step 4: Test without memory**

1. Click "New Chat"
2. Do NOT set any memory
3. Send a message
4. Verify: response generates (pure NTP path, no memory prefix)

- [ ] **Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: KawaiiLLM inference web app — engine + server + frontend"
```

---

## Verification Summary

| Check | How | Expected |
|:------|:----|:---------|
| Engine loads checkpoint | `python -c "from src.inference.engine import ..."` | No errors, model on GPU |
| Memory encoding | `engine.set_memory("text")` | `_memory_prefix_embeds` shape `[1, 130, 4096]` |
| Generation (no stream) | `engine.generate([...], stream=False)` | Returns non-empty string |
| Generation (stream) | `for tok in engine.generate(..., stream=True)` | Yields token strings |
| Server health | `curl /api/health` | `{"status": "ok"}` |
| Server SSE | `curl -N /api/chat` | Streams `data: {"token": ...}` |
| Frontend renders | Open browser | Warm beige UI with sidebar |
| Memory panel | Toggle + paste + Apply | Network request succeeds |
| Chat streaming | Send message | Tokens appear one-by-one |
| Multi-turn | Send follow-up | Conversation history preserved |
| No-memory path | Chat without setting memory | Works (pure NTP) |
