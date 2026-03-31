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
