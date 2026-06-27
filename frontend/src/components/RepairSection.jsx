// where stage
import React from 'react';

const RepairSection = ({ repairs, selectedSolution, setSelectedSolution }) => {
  // If no repair solutions, don't render
  if (!repairs || repairs.length === 0) {
    return null;
  }

  const currentSolution = repairs[selectedSolution];

  return (
    <section className="mt-8">
      <div className="flex items-center gap-2 mb-4">
        <svg
          className="w-5 h-5 text-pink-500"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4m0 4h.01" stroke="white" strokeWidth="2" strokeLinecap="round" />
        </svg>
        <h3 className="text-lg font-semibold text-gray-800">
          Where Clause Repairs ({repairs.length} {repairs.length === 1 ? 'Solution' : 'Solutions'})
        </h3>
      </div>

      {/* Tab selector for multiple solutions */}
      {repairs.length > 1 && (
        <div className="mb-4">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-3">
            <p className="text-xs font-semibold text-gray-600 mb-2">
              Available Repair Solutions (ranked by cost):
            </p>
            <div className="flex gap-2">
              {repairs.map((solution, index) => (
                <button
                  key={index}
                  onClick={() => setSelectedSolution(index)}
                  className={`
                    flex-1 px-3 py-2 rounded-md text-sm font-medium transition-all
                    ${selectedSolution === index
                      ? 'bg-pink-500 text-white shadow-sm'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }
                  `}
                >
                  <div className="flex flex-col items-center">
                    <span className="font-semibold">Solution {index + 1}</span>
                    <span className="text-xs opacity-90">Cost: {solution.cost?.toFixed(3) || 'N/A'}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Display selected solution's repairs */}
      <div className="space-y-6">
        {currentSolution.repairs.map((repair, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-5">
            <div className="flex items-center gap-2 mb-3">
              <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-pink-100 text-pink-600 text-xs font-semibold">
                {index + 1}
              </span>
              <h4 className="text-sm font-semibold text-gray-700">Repair Suggestion</h4>
            </div>

            <div className="space-y-3">
              {/* Repair Site */}
              <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
                <p className="text-xs font-semibold text-red-800 mb-1">Problem Found:</p>
                <code className="text-sm text-red-900 font-mono break-all">
                  {repair.site}
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
                  {repair.fix}
                </code>
              </div>
            </div>

            {/* Metadata (optional) */}
            {(repair.repairSiteSize || repair.fixSize) && (
              <div className="mt-3 pt-3 border-t border-gray-200 flex gap-4 text-xs text-gray-500">
                <span>Repair Size: {repair.repairSiteSize}</span>
                <span>Fix Size: {repair.fixSize}</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Helper text */}
      {repairs.length > 1 && (
        <div className="mt-4 p-3 bg-pink-50 border border-pink-200 rounded-lg">
          <div className="flex items-start gap-2">
            <svg
              className="w-5 h-5 text-pink-600 flex-shrink-0 mt-0.5"
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
            <p className="text-sm text-pink-700">
              <strong>Tip:</strong> Multiple repair solutions are available, ranked by cost (lower is better).
              Each solution may suggest different approaches to fix your query. Try the lowest cost solution first.
            </p>
          </div>
        </div>
      )}
    </section>
  );
};

export default RepairSection;
