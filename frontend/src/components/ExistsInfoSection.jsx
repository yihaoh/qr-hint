import React from 'react';

const ExistsInfoSection = ({ existsInfo }) => {
  // Don't render if no subquery info or no subquery in query
  if (!existsInfo || (!existsInfo.has_exists && !existsInfo.has_some)) {
    return null;
  }

  const { original_query, rewritten_query, tables_added, has_exists, has_some } = existsInfo;

  // Determine what type of subquery was detected
  const getSubqueryType = () => {
    if (has_exists && has_some) return 'EXISTS and SOME/ANY';
    if (has_exists) return 'EXISTS';
    if (has_some) return 'SOME/ANY';
    return 'Subquery';
  };

  return (
    <section className="mt-4">
      <div className="bg-blue-50 border-l-4 border-blue-500 rounded-lg p-4 shadow-sm">
        <div className="flex items-start gap-3">
          {/* Subquery Icon */}
          <svg
            className="w-6 h-6 text-blue-600 flex-shrink-0 mt-0.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>

          <div className="flex-1">
            <h4 className="text-sm font-semibold text-blue-800 mb-2">
              {getSubqueryType()} Subquery Detected
            </h4>
            <p className="text-sm text-blue-700 mb-3">
              Your query contains {has_exists && has_some ? 'EXISTS and SOME/ANY subqueries' : has_exists ? 'an EXISTS subquery' : 'a SOME/ANY subquery'}. For analysis, it has been converted to an equivalent form.
            </p>

            {/* Original Query */}
            <div className="mb-3">
              <span className="text-xs font-medium text-blue-600 uppercase tracking-wide">
                Your Original Query:
              </span>
              <div className="mt-1 p-2 bg-blue-100 rounded border border-blue-200 overflow-x-auto">
                <code className="text-xs text-blue-900 font-mono whitespace-pre-wrap break-all">
                  {original_query}
                </code>
              </div>
            </div>

            {/* Rewritten Query */}
            <div className="mb-3">
              <span className="text-xs font-medium text-blue-600 uppercase tracking-wide">
                Equivalent Rewritten Query (for analysis):
              </span>
              <div className="mt-1 p-2 bg-blue-100 rounded border border-blue-200 overflow-x-auto">
                <code className="text-xs text-blue-900 font-mono whitespace-pre-wrap break-all">
                  {rewritten_query}
                </code>
              </div>
            </div>

            {/* Tables Added */}
            {tables_added && tables_added.length > 0 && (
              <div>
                <span className="text-xs font-medium text-blue-600 uppercase tracking-wide">
                  Tables from subquery:
                </span>
                <div className="mt-1 flex flex-wrap gap-2">
                  {tables_added.map((table, index) => (
                    <span
                      key={index}
                      className="text-xs font-mono bg-blue-200 text-blue-800 px-2 py-1 rounded border border-blue-300"
                    >
                      {table}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Helper text */}
        {/* <div className="mt-3 pt-3 border-t border-blue-200">
          <div className="flex items-start gap-2">
            <svg
              className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5"
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
            <p className="text-xs text-blue-600">
              <strong>Note:</strong> The EXISTS subquery checks if any rows match the condition.
              The rewritten form is semantically equivalent and used internally for analysis.
              Any hints will be explained in terms of your original EXISTS syntax.
            </p>
          </div>
        </div> */}
      </div>
    </section>
  );
};

export default ExistsInfoSection;
