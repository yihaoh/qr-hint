import React from 'react';

const FromSection = ({ fromClause }) => {
  // Don't render if there are no FROM issues
  if (!fromClause || !fromClause.has_issues) {
    return null;
  }

  const { missing, redundant, wrong_count } = fromClause;

  return (
    <section className="mt-8">
      <div className="flex items-center gap-2 mb-4">
        <svg
          className="w-6 h-6 text-orange-600"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
        >
          <path
            d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <h3 className="text-lg font-semibold text-gray-800">FROM Clause Issues</h3>
      </div>

      <div className="space-y-4">
        {/* Missing tables */}
        {missing && missing.length > 0 && (
          <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-red-800 mb-2">
                  Missing Tables
                </h4>
                <p className="text-sm text-red-700 mb-2">
                  The following tables are missing from your FROM clause:
                </p>
                <ul className="space-y-1">
                  {missing.map((table, index) => (
                    <li
                      key={index}
                      className="text-sm font-mono bg-red-100 text-red-900 px-3 py-2 rounded border border-red-300"
                    >
                      {table}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Redundant tables */}
        {redundant && redundant.length > 0 && (
          <div className="bg-orange-50 border-l-4 border-orange-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-orange-800 mb-2">
                  Redundant Tables
                </h4>
                <p className="text-sm text-orange-700 mb-2">
                  The following tables should not be in your FROM clause:
                </p>
                <ul className="space-y-1">
                  {redundant.map((table, index) => (
                    <li
                      key={index}
                      className="text-sm font-mono bg-orange-100 text-orange-900 px-3 py-2 rounded border border-orange-300"
                    >
                      {table}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Wrong count of table references */}
        {wrong_count && wrong_count.length > 0 && (
          <div className="bg-yellow-50 border-l-4 border-yellow-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-yellow-800 mb-2">
                  Table Reference Count Issue
                </h4>
                <ul className="space-y-1">
                  {wrong_count.map((msg, index) => (
                    <li
                      key={index}
                      className="text-sm bg-yellow-100 text-yellow-900 px-3 py-2 rounded border border-yellow-300"
                    >
                      {msg}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Helper text */}
      <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
        <div className="flex items-start gap-2">
          <svg
            className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5"
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
          <p className="text-sm text-orange-700">
            <strong>Tip:</strong> Make sure your FROM clause includes all the tables mentioned in the question,
            and only those tables. Each table should be referenced the correct number of times.
          </p>
        </div>
      </div>
    </section>
  );
};

export default FromSection;
