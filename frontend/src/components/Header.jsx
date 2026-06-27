import React from 'react';

const Header = ({ isTeacherMode, onToggleMode }) => {
  return (
    <header className="bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 text-white px-6 py-3.5 flex items-center justify-between shadow-lg">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-white/15 flex items-center justify-center backdrop-blur-sm">
          <svg
            className="w-5 h-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M4 7C4 5.89543 4.89543 5 6 5H18C19.1046 5 20 5.89543 20 7V17C20 18.1046 19.1046 19 18 19H6C4.89543 19 4 18.1046 4 17V7Z" />
            <path d="M4 10H20M8 5V19" />
          </svg>
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight leading-tight">QR-Hint</h1>
          <p className="text-xs text-white/50 font-medium">SQL Query Repair Tool</p>
        </div>
      </div>

      <div className="flex items-center gap-5">
        {/* TA/Student Mode Toggle */}
        <div className="flex items-center gap-2.5 bg-white/10 rounded-full px-3 py-1.5">
          <span className={`text-xs font-medium transition-opacity ${!isTeacherMode ? 'opacity-100' : 'opacity-40'}`}>
            Student
          </span>
          <button
            onClick={onToggleMode}
            className={`relative w-11 h-6 rounded-full transition-colors duration-300 ${
              isTeacherMode ? 'bg-emerald-500' : 'bg-white/25'
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-all duration-300 ${
                isTeacherMode ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
          <span className={`text-xs font-medium transition-opacity ${isTeacherMode ? 'opacity-100' : 'opacity-40'}`}>
            TA
          </span>
        </div>
      </div>
    </header>
  );
};

export default Header;
