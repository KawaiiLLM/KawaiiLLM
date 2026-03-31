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
