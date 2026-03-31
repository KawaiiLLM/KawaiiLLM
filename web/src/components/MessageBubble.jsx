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
