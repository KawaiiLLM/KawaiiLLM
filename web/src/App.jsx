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
