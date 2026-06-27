import React from 'react';

const ProgressBar = ({ stage, hasDuplicateMismatch }) => {
  if (!stage) return null;

  const stages = [
    { id: 'from', name: 'FROM' },
    { id: 'where', name: 'WHERE' },
    { id: 'group_by', name: 'GROUP BY' },
    { id: 'having', name: 'HAVING' },
    { id: 'select', name: 'SELECT' },
  ];

  const completedStages = stage.completed || [];
  const isComplete = (stage.current === 'complete' || completedStages.length === stages.length) && !hasDuplicateMismatch;

  return (
    <div className="mb-4">
      <div className={`bg-white rounded-xl shadow-sm px-5 py-4 border ${hasDuplicateMismatch ? 'border-amber-300' : 'border-slate-200'}`}>
        {/* Header row */}
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-600">Repair Progress</h3>
          {isComplete && (
            <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
              All Passed
            </span>
          )}
          {hasDuplicateMismatch && (
            <span className="text-xs font-medium text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full flex items-center gap-1">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Duplicate Mismatch
            </span>
          )}
        </div>

        {/* Stage pills */}
        <div className="flex gap-1.5">
          {stages.map((stageItem) => {
            const isCompleted = completedStages.includes(stageItem.id);
            const isCurrent = stageItem.id === stage.current;

            return (
              <div
                key={stageItem.id}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-semibold transition-all duration-300
                  ${isCompleted
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : isCurrent
                      ? 'bg-blue-50 text-blue-700 border border-blue-200 shadow-sm'
                      : 'bg-slate-50 text-slate-400 border border-slate-100'
                  }`}
              >
                {isCompleted && (
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
                {isCurrent && (
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                )}
                {stageItem.name}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ProgressBar;
