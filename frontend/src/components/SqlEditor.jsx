import { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import { EditorView, keymap } from '@codemirror/view';
import { EditorState, StateField, StateEffect, Compartment, RangeSetBuilder } from '@codemirror/state';
import { Decoration } from '@codemirror/view';
import { basicSetup } from 'codemirror';
import { sql, PostgreSQL } from '@codemirror/lang-sql';
import { format } from 'sql-formatter';
import { DB_SCHEMAS } from './SchemaSelector';

// Clause keywords we can locate in a query. order_by/limit are boundary-only:
// they end the preceding clause but are never highlighted themselves.
const CLAUSE_KEYWORDS = [
  { key: 'select', re: /^select\b/i },
  { key: 'from', re: /^from\b/i },
  { key: 'where', re: /^where\b/i },
  { key: 'group_by', re: /^group\s+by\b/i },
  { key: 'having', re: /^having\b/i },
  { key: 'order_by', re: /^order\s+by\b/i },
  { key: 'limit', re: /^limit\b/i },
];

/**
 * Locate the top-level clauses of a query.
 * Subqueries are skipped by tracking paren depth, so only the outer query's
 * clauses are reported. The first occurrence of each clause wins, which keeps
 * UNION-ed queries from producing overlapping ranges.
 *
 * @returns {Object} clause key -> { from, to } offsets into sql
 */
const findClauseRanges = (sqlText) => {
  const hits = [];
  let depth = 0;

  for (let i = 0; i < sqlText.length; i++) {
    const ch = sqlText[i];

    // skip string literals so a keyword inside quotes is never matched
    if (ch === "'" || ch === '"') {
      const quote = ch;
      i++;
      while (i < sqlText.length && sqlText[i] !== quote) i++;
      continue;
    }
    // skip line comments
    if (ch === '-' && sqlText[i + 1] === '-') {
      while (i < sqlText.length && sqlText[i] !== '\n') i++;
      continue;
    }
    if (ch === '(') { depth++; continue; }
    if (ch === ')') { depth--; continue; }
    if (depth !== 0) continue;

    // only test at a word boundary
    const prev = i > 0 ? sqlText[i - 1] : ' ';
    if (/[\w$]/.test(prev)) continue;

    const rest = sqlText.slice(i);
    for (const { key, re } of CLAUSE_KEYWORDS) {
      const m = rest.match(re);
      if (m) {
        if (!hits.some((h) => h.key === key)) hits.push({ key, from: i });
        i += m[0].length - 1;
        break;
      }
    }
  }

  hits.sort((a, b) => a.from - b.from);

  const ranges = {};
  hits.forEach((hit, idx) => {
    const next = hits[idx + 1];
    let to = next ? next.from : sqlText.length;
    // don't let the highlight bleed into the whitespace before the next clause
    while (to > hit.from && /\s/.test(sqlText[to - 1])) to--;
    if (to > hit.from) ranges[hit.key] = { from: hit.from, to };
  });

  return ranges;
};

const setHighlights = StateEffect.define();

const highlightField = StateField.define({
  create: () => Decoration.none,
  update(decorations, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setHighlights)) return effect.value;
    }
    return decorations.map(tr.changes);
  },
  provide: (f) => EditorView.decorations.from(f),
});

const PASSED_MARK = Decoration.mark({ class: 'cm-clause-passed' });
const FAILED_MARK = Decoration.mark({ class: 'cm-clause-failed' });

const buildDecorations = (sqlText, highlights) => {
  const { passed = [], failed = null } = highlights || {};
  if (passed.length === 0 && !failed) return Decoration.none;

  const ranges = findClauseRanges(sqlText);
  const builder = new RangeSetBuilder();

  // RangeSetBuilder requires ranges added in ascending order of `from`
  const marks = [];
  passed.forEach((clause) => {
    if (ranges[clause]) marks.push({ ...ranges[clause], deco: PASSED_MARK });
  });
  if (failed && ranges[failed]) {
    marks.push({ ...ranges[failed], deco: FAILED_MARK });
  }
  marks.sort((a, b) => a.from - b.from);
  marks.forEach(({ from, to, deco }) => builder.add(from, to, deco));

  return builder.finish();
};

const editorTheme = EditorView.theme({
  '&': {
    fontSize: '13px',
    border: '1px solid rgb(226 232 240)',
    borderRadius: '0.5rem',
    overflow: 'hidden',
  },
  '&.cm-focused': {
    outline: 'none',
    borderColor: 'rgb(100 116 139)',
    boxShadow: '0 0 0 3px rgb(100 116 139 / 0.1)',
  },
  '.cm-scroller': {
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
    minHeight: '180px',
    maxHeight: '420px',
  },
  '.cm-content': { padding: '10px 0' },
  '.cm-gutters': {
    backgroundColor: 'rgb(248 250 252)',
    borderRight: '1px solid rgb(226 232 240)',
    color: 'rgb(148 163 184)',
  },
  '.cm-clause-passed': {
    backgroundColor: 'rgb(220 252 231)',
    borderBottom: '2px solid rgb(34 197 94)',
  },
  '.cm-clause-failed': {
    backgroundColor: 'rgb(254 226 226)',
    borderBottom: '2px solid rgb(239 68 68)',
  },
  '&.cm-editor.cm-disabled': { opacity: 0.6 },
});

/** Turn DB_SCHEMAS[schema] into the { table: [col, ...] } shape lang-sql wants. */
const toCompletionSchema = (schemaId) => {
  const tables = DB_SCHEMAS[schemaId] || DB_SCHEMAS.beers || {};
  return Object.fromEntries(
    Object.entries(tables).map(([table, columns]) => [
      table,
      columns.map((c) => c.name),
    ])
  );
};

const SqlEditor = ({ value, onChange, schema = 'beers', disabled = false, highlights }) => {
  const hostRef = useRef(null);
  const viewRef = useRef(null);
  const onChangeRef = useRef(onChange);
  const languageComp = useRef(new Compartment());
  const editableComp = useRef(new Compartment());

  // keep the latest onChange without tearing down the editor
  useEffect(() => { onChangeRef.current = onChange; }, [onChange]);

  // create the editor once
  useEffect(() => {
    const formatKeymap = keymap.of([
      {
        key: 'Mod-Shift-f',
        run: (view) => {
          try {
            const formatted = format(view.state.doc.toString(), {
              language: 'postgresql',
              keywordCase: 'upper',
              indentStyle: 'standard',
              logicalOperatorNewline: 'before',
            });
            view.dispatch({
              changes: { from: 0, to: view.state.doc.length, insert: formatted },
            });
          } catch {
            // leave the query untouched if it can't be parsed
          }
          return true;
        },
      },
    ]);

    const view = new EditorView({
      state: EditorState.create({
        doc: value || '',
        extensions: [
          basicSetup,
          formatKeymap,
          languageComp.current.of(
            sql({ dialect: PostgreSQL, schema: toCompletionSchema(schema), upperCaseKeywords: true })
          ),
          editableComp.current.of(EditorView.editable.of(!disabled)),
          highlightField,
          editorTheme,
          EditorView.lineWrapping,
          EditorView.updateListener.of((update) => {
            if (update.docChanged) {
              onChangeRef.current?.(update.state.doc.toString());
            }
          }),
        ],
      }),
      parent: hostRef.current,
    });

    viewRef.current = view;
    return () => { view.destroy(); viewRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // push external value changes (question switch, Format button) into the doc
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    if (value !== current) {
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: value || '' },
      });
    }
  }, [value]);

  // swap autocompletion when the schema selector changes
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({
      effects: languageComp.current.reconfigure(
        sql({ dialect: PostgreSQL, schema: toCompletionSchema(schema), upperCaseKeywords: true })
      ),
    });
  }, [schema]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({
      effects: editableComp.current.reconfigure(EditorView.editable.of(!disabled)),
    });
  }, [disabled]);

  // recompute clause marks whenever the verdict or the query text changes
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({
      effects: setHighlights.of(buildDecorations(view.state.doc.toString(), highlights)),
    });
  }, [highlights, value]);

  return <div ref={hostRef} className={disabled ? 'cm-disabled' : undefined} />;
};

SqlEditor.propTypes = {
  value: PropTypes.string,
  onChange: PropTypes.func.isRequired,
  schema: PropTypes.string,
  disabled: PropTypes.bool,
  highlights: PropTypes.shape({
    passed: PropTypes.arrayOf(PropTypes.string),
    failed: PropTypes.string,
  }),
};

export default SqlEditor;
