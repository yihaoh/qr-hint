import React from 'react';

const HavingSection = ({ having }) => {
  // Check for DISTINCT mismatches in HAVING
  const hasDistinctMismatches = having?.distinct_mismatches?.length > 0;

  // Don't render if there are no HAVING issues and no DISTINCT mismatches
  if (!having || (!having.has_issues && !hasDistinctMismatches)) {
    return null;
  }

  const { repairs, distinct_mismatches } = having;

  return (
    <section className="mt-8">
      <div className="flex items-center gap-2 mb-4">
        <svg
          className="w-6 h-6 text-indigo-600"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
        >
          <path
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <h3 className="text-lg font-semibold text-gray-800">
          HAVING Clause Issues ({repairs.length})
        </h3>
      </div>

      <div className="space-y-6">
        {repairs.map((repair, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
            <div className="flex items-center gap-2 mb-3">
              <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-indigo-100 text-indigo-600 text-xs font-semibold">
                {index + 1}
              </span>
              <h4 className="text-sm font-semibold text-gray-700">HAVING Repair Suggestion</h4>
            </div>

            <div className="space-y-3">
              {/* Repair Site */}
              <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
                <p className="text-xs font-semibold text-red-800 mb-1">Problem Found:</p>
                <code className="text-sm text-red-900 font-mono break-all">
                  {repair.repair_site || 'N/A'}
                </code>
              </div>

              {/* Arrow indicator */}
              <div className="flex justify-center">
                <svg
                  className="w-5 h-5 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={3}
                    d="M19 14l-7 7m0 0l-7-7m7 7V3"
                  />
                </svg>
              </div>

              {/* Fix */}
              <div className="bg-green-50 border-l-4 border-green-400 p-4 rounded">
                <p className="text-xs font-semibold text-green-800 mb-1">Suggested Fix:</p>
                <code className="text-sm text-green-900 font-mono break-all">
                  {repair.fix || 'N/A'}
                </code>
              </div>
            </div>

            {/* Metadata (optional) */}
            {(repair.repair_site_size || repair.fix_size) && (
              <div className="mt-3 pt-3 border-t border-gray-200 flex gap-4 text-xs text-gray-500">
                <span>Repair Size: {repair.repair_site_size}</span>
                <span>Fix Size: {repair.fix_size}</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* HAVING DISTINCT mismatches */}
      {distinct_mismatches && distinct_mismatches.length > 0 && (
        <div className="mt-6 bg-purple-50 border-l-4 border-purple-500 rounded-lg p-4 shadow-sm">
          <div className="flex items-start gap-3">
            <svg
              className="w-5 h-5 text-purple-500 flex-shrink-0 mt-0.5"
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
              <h4 className="text-sm font-semibold text-purple-800 mb-2">
                HAVING Aggregate DISTINCT Mismatch
              </h4>
              <p className="text-sm text-purple-700 mb-2">
                The following aggregate functions in your HAVING clause have mismatched DISTINCT usage:
              </p>
              <ul className="space-y-1">
                {distinct_mismatches.map((mismatch, index) => (
                  <li
                    key={index}
                    className="text-sm font-mono bg-purple-100 text-purple-900 px-3 py-2 rounded border border-purple-300"
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

      {/* Helper text */}
      <div className="mt-4 p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
        <div className="flex items-start gap-2">
          <svg
            className="w-5 h-5 text-indigo-600 flex-shrink-0 mt-0.5"
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
          <p className="text-sm text-indigo-700">
            <strong>Tip:</strong> The HAVING clause filters grouped results based on aggregate functions
            (COUNT, SUM, AVG, MAX, MIN). Make sure your HAVING conditions correctly use these aggregate functions
            to filter the grouped data.
          </p>
        </div>
      </div>
    </section>
  );
};

export default HavingSection;
