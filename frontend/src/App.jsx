import { useState } from 'react';
import Header from './components/Header';
import QuestionList from './components/QuestionList';
import QueryCard from './components/QueryCard';
import RepairSection from './components/RepairSection';
import HintSection from './components/HintSection';
import GroupBySection from './components/GroupBySection';
import FromSection from './components/FromSection';
import HavingSection from './components/HavingSection';
import SelectSection from './components/SelectSection';
import ProgressBar from './components/ProgressBar';
import ExistsInfoSection from './components/ExistsInfoSection';
import SchemaSelector, { DB_SCHEMAS } from './components/SchemaSelector';
import { questionsBySchema } from './data/questions';
import { repairQuery, generateAIHintStream, generateFromHintStream, generateGroupByHintStream, generateHavingHintStream, generateSelectHintStream } from './services/api';

// Current stage types for AI hint
const HINT_STAGE = {
  FROM: 'from',
  WHERE: 'where',
  GROUP_BY: 'group_by',
  HAVING: 'having',
  SELECT: 'select'
};

function App() {
  // State management
  const [activeSchema, setActiveSchema] = useState('beers'); // Current schema
  const [activeQuestion, setActiveQuestion] = useState('q1');
  const [query, setQuery] = useState('');
  const [repairs, setRepairs] = useState([]);
  const [selectedSolution, setSelectedSolution] = useState(0); // Track which solution is selected
  const [fromClause, setFromClause] = useState(null);
  const [groupBy, setGroupBy] = useState(null);
  const [having, setHaving] = useState(null);
  const [selectClause, setSelectClause] = useState(null);
  const [stage, setStage] = useState(null);
  const [systemHint, setSystemHint] = useState(''); // System message (non-AI)
  const [aiHints, setAiHints] = useState({ 1: '', 2: '', 3: '' }); // Multi-level AI hints
  const [loadingHintLevel, setLoadingHintLevel] = useState(null); // Which level is loading (null | 1 | 2 | 3)
  const [currentHintStage, setCurrentHintStage] = useState(null); // Track which stage needs AI hint
  const [subqueryInfo, setSubqueryInfo] = useState(null); // Subquery (EXISTS/SOME/ANY) rewrite information
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isTeacherMode, setIsTeacherMode] = useState(false); // TA/Student mode toggle (default to Student mode)
  const [clauseHighlights, setClauseHighlights] = useState({ passed: [], failed: null });
  const [hasDuplicateMismatch, setHasDuplicateMismatch] = useState(false); // EXISTS vs DISTINCT style mismatch

  // Handle query change — clear red highlight on user edit
  const handleQueryChange = (newQuery) => {
    setQuery(newQuery);
    if (clauseHighlights.failed) {
      setClauseHighlights(prev => ({ ...prev, failed: null }));
    }
  };

  // Get questions for current schema
  const questionsData = questionsBySchema[activeSchema] || [];

  // Get current question data
  const currentQuestionData = questionsData.find(q => q.id === activeQuestion);

  // Handle schema change
  const handleSchemaChange = (schemaId) => {
    setActiveSchema(schemaId);
    // Reset to first question of new schema
    const newQuestions = questionsBySchema[schemaId] || [];
    setActiveQuestion(newQuestions.length > 0 ? newQuestions[0].id : '');
    // Clear all state
    setQuery('');
    setRepairs([]);
    setSelectedSolution(0);
    setFromClause(null);
    setGroupBy(null);
    setHaving(null);
    setSelectClause(null);
    setStage(null);
    setSystemHint('');
    setAiHints({ 1: '', 2: '', 3: '' });
    setCurrentHintStage(null);
    setSubqueryInfo(null);
    setClauseHighlights({ passed: [], failed: null });
    setHasDuplicateMismatch(false);
    setError(null);
  };

  // Handle question selection
  const handleSelectQuestion = (questionId) => {
    setActiveQuestion(questionId);
    setQuery(''); // Clear query when switching questions
    setRepairs([]); // Clear previous repairs
    setSelectedSolution(0); // Reset solution selection
    setFromClause(null); // Clear previous FROM issues
    setGroupBy(null); // Clear previous GROUP BY issues
    setHaving(null); // Clear previous HAVING issues
    setSelectClause(null); // Clear previous SELECT issues
    setStage(null); // Clear stage progress
    setSystemHint('');
    setAiHints({ 1: '', 2: '', 3: '' }); // Clear previous hint
    setCurrentHintStage(null); // Reset hint stage
    setSubqueryInfo(null); // Clear subquery info
    setClauseHighlights({ passed: [], failed: null });
    setHasDuplicateMismatch(false);
    setError(null); // Clear previous errors
  };

  // Handle repair button click
  const handleRepair = async () => {
    // Validate user query
    if (!query.trim()) {
      setError('Please enter a SQL query');
      return;
    }

    setIsLoading(true);
    setError(null);
    setRepairs([]);
    setSelectedSolution(0); // Reset solution selection
    setFromClause(null); // Clear previous FROM issues
    setGroupBy(null); // Clear previous GROUP BY issues
    setHaving(null); // Clear previous HAVING issues
    setSelectClause(null); // Clear previous SELECT issues
    setStage(null); // Clear previous stage
    setSystemHint('');
    setAiHints({ 1: '', 2: '', 3: '' }); // Clear previous hint
    setCurrentHintStage(null); // Reset hint stage
    setSubqueryInfo(null); // Clear subquery info
    setClauseHighlights({ passed: [], failed: null }); // Clear highlights during analysis
    setHasDuplicateMismatch(false);

    try {
      // Call repair API with schema parameter
      const result = await repairQuery(
        currentQuestionData.correctQuery,
        query,
        activeSchema
      );

      if (result.ok) {
        // Set stage information
        if (result.stage) {
          setStage(result.stage);
        }

        // Store subquery rewrite info if present
        if (result.subquery_info) {
          setSubqueryInfo(result.subquery_info);
        }

        // Process FROM clause results (only if tested)
        if (result.from_clause) {
          setFromClause(result.from_clause);
        }

        // Process GROUP BY results (only if tested)
        if (result.group_by) {
          setGroupBy(result.group_by);
        }

        // Process HAVING results (only if tested)
        if (result.having) {
          setHaving(result.having);
        }

        // Process SELECT results (only if tested)
        if (result.select) {
          setSelectClause(result.select);
        }

        // Determine hint based on current stage
        const hasFromIssues = result.from_clause && result.from_clause.has_issues;
        const hasWhereIssues = result.repairs && result.repairs.length > 0;
        const hasGroupByIssues = result.group_by && result.group_by.has_issues;
        const hasHavingIssues = result.having && result.having.has_issues;
        const hasSelectIssues = result.select && result.select.has_issues;

        if (hasFromIssues) {
          // FROM issues - fix FROM first
          setSystemHint('Fix the FROM clause issues above first. WHERE, GROUP BY, HAVING, and SELECT will be checked after FROM is correct.');
          setCurrentHintStage(HINT_STAGE.FROM);
          setClauseHighlights({ passed: [], failed: 'from' });
        } else if (hasWhereIssues) {
          // WHERE issues - fix WHERE next
          // New format: result.repairs is an array of solutions, each with cost and repairs array
          const formattedRepairSolutions = result.repairs.map(solution => ({
            solutionIndex: solution.solution_index,
            cost: solution.cost,
            repairs: solution.repairs.map(repair => ({
              site: repair.repair_site || 'N/A',
              fix: repair.fix || 'N/A',
              repairSiteSize: repair.repair_site_size,
              fixSize: repair.fix_size
            }))
          }));
          setRepairs(formattedRepairSolutions);
          setSystemHint('FROM clause is correct! Now fix the WHERE clause issues above. GROUP BY, HAVING, and SELECT will be checked after WHERE is correct.');
          setCurrentHintStage(HINT_STAGE.WHERE);
          setClauseHighlights({ passed: ['from'], failed: 'where' });
        } else if (hasGroupByIssues) {
          // GROUP BY issues - fix GROUP BY
          setRepairs([]);
          setSystemHint('FROM and WHERE clauses are correct! Now fix the GROUP BY issues above. HAVING and SELECT will be checked after GROUP BY is correct.');
          setCurrentHintStage(HINT_STAGE.GROUP_BY);
          setClauseHighlights({ passed: ['from', 'where'], failed: 'group_by' });
        } else if (hasHavingIssues) {
          // HAVING issues - fix HAVING
          setRepairs([]);
          setSystemHint('FROM, WHERE, and GROUP BY clauses are correct! Now fix the HAVING clause issues above. SELECT will be checked after HAVING is correct.');
          setCurrentHintStage(HINT_STAGE.HAVING);
          setClauseHighlights({ passed: ['from', 'where', 'group_by'], failed: 'having' });
        } else if (hasSelectIssues) {
          // SELECT issues - fix SELECT last
          setRepairs([]);
          setSystemHint('FROM, WHERE, GROUP BY, and HAVING clauses are correct! Now fix the SELECT clause issues above.');
          setCurrentHintStage(HINT_STAGE.SELECT);
          setClauseHighlights({ passed: ['from', 'where', 'group_by', 'having'], failed: 'select' });
        } else {
          // All structural stages correct!
          setRepairs([]);
          setClauseHighlights({ passed: ['select', 'from', 'where', 'group_by', 'having'], failed: null });

          const subqInfo = result.subquery_info;
          const hasStyleMismatch = subqInfo?.subquery_style_mismatch;

          if (hasStyleMismatch) {
            // One uses EXISTS, the other uses DISTINCT/JOIN — potential duplicate row issue
            setHasDuplicateMismatch(true);
            setSystemHint('⚠️ Duplicate Mismatch: Your query uses a different structural approach (EXISTS vs JOIN+DISTINCT). While both may produce the same result set (set semantics), they differ in bag semantics — a JOIN without DISTINCT introduces duplicate rows, causing row count mismatches. Make sure duplicates are handled correctly.');
          } else {
            // Same subquery style or no subquery — clean pass
            setSystemHint('Congratulations! Your query is correct.');
          }
        }
      } else {
        setError(result.error || 'Failed to repair query');
      }
    } catch (err) {
      setError(err.message || 'Failed to connect to the backend');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle AI hint request (multi-level) - supports FROM, WHERE, GROUP BY, HAVING, and SELECT stages
  const handleRequestAIHint = async (level) => {
    const hasFromIssues = fromClause && fromClause.has_issues;
    const hasWhereIssues = repairs.length > 0;
    const hasGroupByIssues = groupBy && groupBy.has_issues;
    const hasHavingIssues = having && having.has_issues;
    const hasSelectIssues = selectClause && selectClause.has_issues;

    if (!hasFromIssues && !hasWhereIssues && !hasGroupByIssues && !hasHavingIssues && !hasSelectIssues) {
      return;
    }

    setLoadingHintLevel(level);
    setError(null);

    try {
      let streamedHint = '';
      const onChunk = (chunk) => {
        streamedHint += chunk;
        setAiHints(prev => ({ ...prev, [level]: streamedHint }));
      };

      if (currentHintStage === HINT_STAGE.FROM && hasFromIssues) {
        const fromIssues = {
          missing_tables: fromClause.missing || [],
          redundant_tables: fromClause.redundant || [],
          wrong_count: fromClause.wrong_count && fromClause.wrong_count.length > 0
        };
        await generateFromHintStream(currentQuestionData.question, query, fromIssues, onChunk, subqueryInfo, level);

      } else if (currentHintStage === HINT_STAGE.WHERE && hasWhereIssues) {
        const selectedSolutionData = repairs[selectedSolution];
        const apiRepairs = selectedSolutionData.repairs.map(repair => ({
          repair_site: repair.site,
          fix: repair.fix,
          repair_site_size: repair.repairSiteSize,
          fix_size: repair.fixSize
        }));
        await generateAIHintStream(currentQuestionData.question, query, apiRepairs, onChunk, subqueryInfo, level);

      } else if (currentHintStage === HINT_STAGE.GROUP_BY && hasGroupByIssues) {
        const groupByIssues = {
          incorrect: groupBy.incorrect || [],
          missing: groupBy.missing || []
        };
        await generateGroupByHintStream(currentQuestionData.question, query, groupByIssues, onChunk, level);

      } else if (currentHintStage === HINT_STAGE.HAVING && hasHavingIssues) {
        const havingIssues = {
          repairs: having.repairs || [],
          distinct_mismatches: having.distinct_mismatches || []
        };
        await generateHavingHintStream(currentQuestionData.question, query, havingIssues, onChunk, level);

      } else if (currentHintStage === HINT_STAGE.SELECT && hasSelectIssues) {
        const selectIssues = {
          incorrect: selectClause.incorrect || [],
          wrong_order: selectClause.wrong_order || [],
          missing: selectClause.missing || [],
          distinct_mismatches: selectClause.distinct_mismatches || []
        };
        await generateSelectHintStream(currentQuestionData.question, query, selectIssues, onChunk, subqueryInfo, level);
      }

    } catch (err) {
      setError(err.message || 'Failed to generate AI hint');
    } finally {
      setLoadingHintLevel(null);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <Header
        isTeacherMode={isTeacherMode}
        onToggleMode={() => setIsTeacherMode(!isTeacherMode)}
      />

      {/* Main content area with sidebar and content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Schema Selector and Questions List */}
        <div className="flex flex-col w-72 bg-slate-50 border-r border-slate-200">
          <SchemaSelector
            activeSchema={activeSchema}
            onSchemaChange={handleSchemaChange}
          />
          <QuestionList
            questions={questionsData}
            activeQuestion={activeQuestion}
            onSelectQuestion={handleSelectQuestion}
          />
        </div>

        {/* Main content area */}
        <main className="flex-1 overflow-y-auto bg-slate-100">
          <div className="max-w-5xl mx-auto p-6 lg:p-8">
            {/* Query Card */}
            <QueryCard
              question={currentQuestionData?.question}
              query={query}
              onQueryChange={handleQueryChange}
              onRepair={handleRepair}
              isLoading={isLoading}
              correctQuery={currentQuestionData?.correctQuery}
              isTeacherMode={isTeacherMode}
              schema={DB_SCHEMAS[activeSchema]}
              highlights={clauseHighlights}
            />

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow-sm">
                <div className="flex items-start gap-3">
                  {/* Error Icon */}
                  <svg
                    className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5"
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

                  {/* Error Message */}
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-red-800 mb-1">Query Error</h4>
                    <p className="text-sm text-red-700 leading-relaxed">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Progress Bar - Show when there are repairs or results */}
            {(stage || repairs.length > 0 || (fromClause && fromClause.tested) || (groupBy && groupBy.tested) || (having && having.tested) || (selectClause && selectClause.tested)) && (
              <ProgressBar stage={stage} hasDuplicateMismatch={hasDuplicateMismatch} />
            )}

            {/* Subquery Info Section - Show when query contains EXISTS/SOME/ANY */}
            {/* <ExistsInfoSection existsInfo={subqueryInfo} /> */}

            {/* FROM Section - Only show if tested and has issues (TA mode only) */}
            {isTeacherMode && fromClause && fromClause.tested && (
              <FromSection fromClause={fromClause} />
            )}

            {/* Repair Section - WHERE repairs (TA mode only) */}
            {isTeacherMode && (
              <RepairSection
                repairs={repairs}
                selectedSolution={selectedSolution}
                setSelectedSolution={setSelectedSolution}
              />
            )}

            {/* GROUP BY Section - Only show if tested (WHERE is correct) (TA mode only) */}
            {isTeacherMode && groupBy && groupBy.tested && (
              <GroupBySection groupBy={groupBy} />
            )}

            {/* HAVING Section - Only show if tested (GROUP BY is correct) (TA mode only) */}
            {isTeacherMode && having && having.tested && (
              <HavingSection having={having} />
            )}

            {/* SELECT Section - Only show if tested (HAVING is correct) (TA mode only) */}
            {isTeacherMode && selectClause && selectClause.tested && (
              <SelectSection selectClause={selectClause} />
            )}

            {/* Hint Section - Always show when there are issues or a hint */}
            {(repairs.length > 0 || (fromClause && fromClause.has_issues) || (groupBy && groupBy.has_issues) || (having && having.has_issues) || (selectClause && selectClause.has_issues) || systemHint) && (
              <HintSection
                systemHint={systemHint}
                aiHints={aiHints}
                loadingHintLevel={loadingHintLevel}
                onRequestHint={handleRequestAIHint}
                hasRepairs={repairs.length > 0 || (fromClause && fromClause.has_issues) || (groupBy && groupBy.has_issues) || (having && having.has_issues) || (selectClause && selectClause.has_issues)}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
