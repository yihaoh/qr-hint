import React from 'react';

const LEVEL_CONFIG = [
  { level: 1, label: 'Level 1: Direction', description: 'A gentle nudge toward the problem area', color: 'blue' },
  { level: 2, label: 'Level 2: Pinpoint', description: 'Identifies the type of issue', color: 'amber' },
  { level: 3, label: 'Level 3: Near-Answer', description: 'Detailed guidance, almost the answer', color: 'red' },
];

const HintSection = ({ systemHint, aiHints, loadingHintLevel, onRequestHint, hasRepairs }) => {
  // Determine the highest revealed level (any level with content)
  const highestRevealed = [3, 2, 1].find(l => aiHints[l]) || 0;

  return (
    <section className="mt-8">
      <div className="flex items-center gap-2 mb-4">
        <svg
          className="w-6 h-6 text-gray-700"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
        >
          <path
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <h3 className="text-lg font-semibold text-gray-800">Hints</h3>
      </div>

      {/* System message */}
      {systemHint && (
        <div className="hint-box mb-4">
          <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{systemHint}</p>
        </div>
      )}

      {/* Multi-level hints */}
      {hasRepairs && (
        <div className="space-y-3">
          {LEVEL_CONFIG.map(({ level, label, description, color }) => {
            const isRevealed = !!aiHints[level];
            const isLoading = loadingHintLevel === level;
            const isLocked = level > highestRevealed + 1;
            const canReveal = !isRevealed && !isLocked && !loadingHintLevel;

            return (
              <div key={level} className="rounded-lg border border-gray-200 overflow-hidden bg-white">
                {/* Level header */}
                <div className="flex items-center justify-between px-4 py-3 bg-gray-50">
                  <div className="flex items-center gap-3">
                    <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold
                      ${isRevealed
                        ? `bg-${color}-100 text-${color}-700`
                        : isLocked
                          ? 'bg-gray-100 text-gray-400'
                          : `bg-${color}-50 text-${color}-600`
                      }`}
                      style={isRevealed ? {
                        backgroundColor: color === 'blue' ? '#dbeafe' : color === 'amber' ? '#fef3c7' : '#fee2e2',
                        color: color === 'blue' ? '#1d4ed8' : color === 'amber' ? '#b45309' : '#b91c1c',
                      } : isLocked ? {} : {
                        backgroundColor: color === 'blue' ? '#eff6ff' : color === 'amber' ? '#fffbeb' : '#fef2f2',
                        color: color === 'blue' ? '#2563eb' : color === 'amber' ? '#d97706' : '#dc2626',
                      }}
                    >
                      {isRevealed ? (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                        </svg>
                      ) : level}
                    </span>
                    <div>
                      <span className={`text-sm font-medium ${isLocked ? 'text-gray-400' : 'text-gray-700'}`}>
                        {label}
                      </span>
                      <span className={`text-xs ml-2 ${isLocked ? 'text-gray-300' : 'text-gray-400'}`}>
                        {description}
                      </span>
                    </div>
                  </div>

                  {/* Action button */}
                  {!isRevealed && (
                    <button
                      onClick={() => onRequestHint(level)}
                      disabled={!canReveal || isLoading}
                      className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 flex items-center gap-1.5
                        ${isLocked
                          ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                          : isLoading
                            ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                            : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 shadow-sm hover:shadow-md'
                        }`}
                    >
                      {isLoading ? (
                        <>
                          <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Thinking...
                        </>
                      ) : isLocked ? (
                        <>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                            <path d="M7 11V7a5 5 0 0110 0v4" />
                          </svg>
                          Locked
                        </>
                      ) : (
                        <>
                          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                          Reveal
                        </>
                      )}
                    </button>
                  )}
                </div>

                {/* Hint content */}
                {(isRevealed || isLoading) && (
                  <div className="px-4 py-3 border-t border-gray-100">
                    <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                      {aiHints[level] || (isLoading ? '' : '')}
                      {isLoading && !aiHints[level] && (
                        <span className="text-gray-400 italic">Generating hint...</span>
                      )}
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Fallback when no repairs and no system hint */}
      {!hasRepairs && !systemHint && (
        <div className="hint-box">
          <p className="text-gray-500 italic">Run Repair to get analysis and hints.</p>
        </div>
      )}
    </section>
  );
};

export default HintSection;
