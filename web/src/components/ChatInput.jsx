import { useState, useRef } from 'react'
import { SendIcon, StopIcon } from '../icons'

export default function ChatInput({ onSend, onStop, disabled, placeholder = 'Send a message...' }) {
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
          {disabled ? (
            <button
              onClick={onStop}
              className="w-8 h-8 rounded-full flex items-center justify-center
                         bg-accent text-white hover:bg-accent-hover cursor-pointer transition-colors"
            >
              <StopIcon />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!hasText}
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors
                ${hasText
                  ? 'bg-accent text-white hover:bg-accent-hover cursor-pointer'
                  : 'bg-bg-300 text-text-300 cursor-not-allowed'}`}
            >
              <SendIcon />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
