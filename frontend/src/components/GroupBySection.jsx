import React from 'react';

const GroupBySection = ({ groupBy }) => {
  // Don't render if there are no GROUP BY issues
  if (!groupBy || !groupBy.has_issues) {
    return null;
  }

  const { incorrect, missing } = groupBy;

  return (
    <section className="mt-8">
      <div className="flex items-center gap-2 mb-4">
        <svg
          className="w-6 h-6 text-purple-600"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
        >
          <path
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <h3 className="text-lg font-semibold text-gray-800">GROUP BY Issues</h3>
      </div>

      <div className="space-y-4">
        {/* Incorrect GROUP BY columns */}
        {incorrect && incorrect.length > 0 && (
          <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-red-800 mb-2">
                  Incorrect GROUP BY Columns
                </h4>
                <p className="text-sm text-red-700 mb-2">
                  The following columns should not be in the GROUP BY clause:
                </p>
                <ul className="space-y-1">
                  {incorrect.map((col, index) => (
                    <li
                      key={index}
                      className="text-sm font-mono bg-red-100 text-red-900 px-3 py-2 rounded border border-red-300"
                    >
                      {col}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Missing GROUP BY columns */}
        {missing && missing.length > 0 && (
          <div className="bg-green-50 border-l-4 border-green-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-green-800 mb-2">
                  Missing GROUP BY Columns
                </h4>
                <p className="text-sm text-green-700 mb-2">
                  You need to add the following columns to the GROUP BY clause:
                </p>
                <ul className="space-y-1">
                  {missing.map((col, index) => (
                    <li
                      key={index}
                      className="text-sm font-mono bg-green-100 text-green-900 px-3 py-2 rounded border border-green-300"
                    >
                      {col}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Helper text */}
      <div className="mt-4 p-3 bg-purple-50 border border-purple-200 rounded-lg">
        <div className="flex items-start gap-2">
          <svg
            className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <p className="text-sm text-purple-700">
            <strong>Tip:</strong> When using aggregate functions (COUNT, SUM, AVG, etc.),
            all non-aggregated columns in the SELECT clause must appear in the GROUP BY clause.
          </p>
        </div>
      </div>
    </section>
  );
};

export default GroupBySection;
