import React from 'react';

const QuestionList = ({ questions, activeQuestion, onSelectQuestion }) => {
  return (
    <aside className="flex-1 overflow-y-auto bg-slate-50">
      <div className="px-4 pt-4 pb-2">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Questions</h2>
      </div>

      <nav className="px-2 pb-2 space-y-0.5">
        {questions.map((question, index) => {
          const isActive = activeQuestion === question.id;
          return (
            <button
              key={question.id}
              onClick={() => onSelectQuestion(question.id)}
              className={`w-full text-left flex items-center gap-2.5 px-3 py-2.5 rounded-lg transition-all duration-150 group
                ${isActive
                  ? 'bg-white shadow-sm border border-slate-200 text-slate-900'
                  : 'text-slate-600 hover:bg-white/60 hover:text-slate-800'
                }`}
            >
              <span className={`inline-flex items-center justify-center w-6 h-6 rounded-md text-xs font-bold flex-shrink-0 transition-colors
                ${isActive
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-200 text-slate-500 group-hover:bg-slate-300'
                }`}
              >
                {index + 1}
              </span>
              <span className={`text-sm leading-snug ${isActive ? 'font-medium' : ''}`}>
                {question.label}
              </span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
};

export default QuestionList;
