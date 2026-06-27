import React from 'react';

const SelectSection = ({ selectClause }) => {
  // Don't render if there are no SELECT issues and no DISTINCT mismatches
  const hasDistinctMismatches = selectClause?.distinct_mismatches?.length > 0;

  if (!selectClause || (!selectClause.has_issues && !hasDistinctMismatches)) {
    return null;
  }

  const { incorrect, wrong_order, missing, distinct_mismatches } = selectClause;

  // Check for query-level DISTINCT mismatch
  const queryLevelDistinct = distinct_mismatches?.find(m => m.type === 'query_level');
  // Check for aggregate-level DISTINCT mismatches
  const aggregateDistinctMismatches = distinct_mismatches?.filter(m => m.type !== 'query_level') || [];

  return (
    <section className="mt-8">
      <div className="flex items-center gap-2 mb-4">
        <svg
          className="w-6 h-6 text-cyan-600"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
        >
          <path
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <h3 className="text-lg font-semibold text-gray-800">SELECT Clause Issues</h3>
      </div>

      <div className="space-y-4">
        {/* Incorrect SELECT columns */}
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
                  Incorrect SELECT Columns
                </h4>
                <p className="text-sm text-red-700 mb-2">
                  The following columns in your SELECT clause are incorrect or should not be included:
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

        {/* Wrong order columns */}
        {wrong_order && wrong_order.length > 0 && (
          <div className="bg-yellow-50 border-l-4 border-yellow-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-yellow-800 mb-2">
                  Columns in Wrong Order
                </h4>
                <p className="text-sm text-yellow-700 mb-2">
                  These columns are correct but appear in the wrong position:
                </p>
                <ul className="space-y-1">
                  {wrong_order.map((col, index) => (
                    <li
                      key={index}
                      className="text-sm font-mono bg-yellow-100 text-yellow-900 px-3 py-2 rounded border border-yellow-300"
                    >
                      {col}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Missing SELECT columns */}
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
                  Missing SELECT Columns
                </h4>
                <p className="text-sm text-green-700 mb-2">
                  You need to add the following columns to your SELECT clause:
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

        {/* Query-level DISTINCT mismatch */}
        {queryLevelDistinct && (
          <div className="bg-purple-50 border-l-4 border-purple-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-purple-500 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-purple-800 mb-2">
                  DISTINCT Keyword Mismatch
                </h4>
                <p className="text-sm text-purple-700">
                  {queryLevelDistinct.q1_distinct
                    ? 'The correct query uses SELECT DISTINCT, but your query does not. Consider adding the DISTINCT keyword to eliminate duplicate rows.'
                    : 'Your query uses SELECT DISTINCT, but it is not needed. Consider removing the DISTINCT keyword.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Aggregate-level DISTINCT mismatches */}
        {aggregateDistinctMismatches.length > 0 && (
          <div className="bg-indigo-50 border-l-4 border-indigo-500 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-indigo-500 flex-shrink-0 mt-0.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-indigo-800 mb-2">
                  Aggregate DISTINCT Mismatch
                </h4>
                <p className="text-sm text-indigo-700 mb-2">
                  The following aggregate functions have mismatched DISTINCT usage:
                </p>
                <ul className="space-y-1">
                  {aggregateDistinctMismatches.map((mismatch, index) => (
                    <li
                      key={index}
                      className="text-sm font-mono bg-indigo-100 text-indigo-900 px-3 py-2 rounded border border-indigo-300"
                    >
                      {mismatch.q1_distinct
                        ? `Expected ${mismatch.q1_aggregate}(DISTINCT ...) but got ${mismatch.q2_aggregate}(...)`
                        : `Expected ${mismatch.q1_aggregate}(...) but got ${mismatch.q2_aggregate}(DISTINCT ...)`}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Helper text */}
      <div className="mt-4 p-3 bg-cyan-50 border border-cyan-200 rounded-lg">
        <div className="flex items-start gap-2">
          <svg
            className="w-5 h-5 text-cyan-600 flex-shrink-0 mt-0.5"
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
          <p className="text-sm text-cyan-700">
            <strong>Tip:</strong> The SELECT clause determines which columns appear in the query result.
            Make sure you select the correct columns in the correct order as specified in the question.
            If using aggregate functions, remember that non-aggregated columns must appear in the GROUP BY clause.
          </p>
        </div>
      </div>
    </section>
  );
};

export default SelectSection;
