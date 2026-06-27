import React from 'react';
import { format } from 'sql-formatter';
import SqlEditor from './SqlEditor';

const QueryCard = ({ question, query, onQueryChange, onRepair, isLoading, correctQuery, isTeacherMode, schema, highlights }) => {
  const handleFormat = () => {
    if (!query.trim()) return;
    try {
      const formatted = format(query, {
        language: 'postgresql',
        keywordCase: 'upper',
        indentStyle: 'standard',
        logicalOperatorNewline: 'before',
      });
      onQueryChange(formatted);
    } catch {
      // If formatting fails, keep original
    }
  };

  return (
    <section className="card p-6 border border-slate-200">
      <div className="flex items-start gap-4 mb-6">
        <svg
          className="w-8 h-8 text-gray-600 flex-shrink-0 mt-1"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <div className="flex-1">
          <p className="text-2xl text-gray-800 leading-relaxed">{question}</p>
        </div>
      </div>

      {/* Correct Answer - Only visible in TA mode */}
      {isTeacherMode && correctQuery && (
        <div className="mb-6 p-4 bg-green-50 border-l-4 border-green-500 rounded-r-lg">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
            >
              <path
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <div className="flex-1">
              <h4 className="text-sm font-semibold text-green-800 mb-2">
                Correct Answer (TA Only)
              </h4>
              <div className="p-3 bg-green-100 rounded border border-green-200 overflow-x-auto">
                <code className="text-xs text-green-900 font-mono whitespace-pre-wrap break-all">
                  {correctQuery}
                </code>
              </div>
            </div>
          </div>
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-2">
          <label
            htmlFor="query-input"
            className="block text-sm font-medium text-gray-700"
          >
            Your query
          </label>
          <button
            onClick={handleFormat}
            disabled={isLoading || !query.trim()}
            className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 hover:text-gray-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title="Format SQL (Ctrl+Shift+F)"
          >
            <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 10H7" /><path d="M21 6H3" /><path d="M21 14H3" /><path d="M21 18H7" />
            </svg>
            Format
          </button>
        </div>
        <SqlEditor
          value={query}
          onChange={onQueryChange}
          schema={schema}
          disabled={isLoading}
          highlights={highlights}
        />
      </div>

      <div className="mt-4 flex justify-end">
        <button
          onClick={onRepair}
          disabled={isLoading}
          className="px-6 py-2.5 bg-gradient-to-r from-slate-700 to-slate-800 text-white rounded-lg hover:from-slate-600 hover:to-slate-700 transition-all font-medium disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed flex items-center gap-2 shadow-sm hover:shadow-md"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </>
          ) : (
            'Repair'
          )}
        </button>
      </div>
    </section>
  );
};

export default QueryCard;
