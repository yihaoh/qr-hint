import PropTypes from 'prop-types';

// Available schemas with their tables and columns
// Key columns are marked with isKey: true
export const SCHEMAS = [
  { id: 'beers', name: 'Beers', description: 'Bar and drinker database' },
  // { id: 'dblp', name: 'DBLP', description: 'Publication database' },
  { id: 'tpc', name: 'TPC', description: 'TPC benchmark database' }
];

// Database schema details
export const DB_SCHEMAS = {
  beers: {
    Drinker: [
      { name: 'name', type: 'string', isKey: true },
      { name: 'address', type: 'string', isKey: false }
    ],
    Bar: [
      { name: 'name', type: 'string', isKey: true },
      { name: 'address', type: 'string', isKey: false }
    ],
    Beer: [
      { name: 'name', type: 'string', isKey: true },
      { name: 'brewer', type: 'string', isKey: false }
    ],
    Frequents: [
      { name: 'drinker', type: 'string', isKey: true },
      { name: 'bar', type: 'string', isKey: true },
      { name: 'times_a_week', type: 'int', isKey: false }
    ],
    Serves: [
      { name: 'bar', type: 'string', isKey: true },
      { name: 'beer', type: 'string', isKey: true },
      { name: 'price', type: 'float', isKey: false }
    ],
    Likes: [
      { name: 'drinker', type: 'string', isKey: true },
      { name: 'beer', type: 'string', isKey: true }
    ]
  },
  dblp: {
    inproceedings: [
      { name: 'pubkey', type: 'string', isKey: true },
      { name: 'title', type: 'string', isKey: false },
      { name: 'booktitle', type: 'string', isKey: false },
      { name: 'yearx', type: 'int', isKey: false },
      { name: 'area', type: 'string', isKey: false }
    ],
    article: [
      { name: 'pubkey', type: 'string', isKey: true },
      { name: 'title', type: 'string', isKey: false },
      { name: 'journal', type: 'string', isKey: false },
      { name: 'yearx', type: 'int', isKey: false }
    ],
    authorship: [
      { name: 'pubkey', type: 'string', isKey: true },
      { name: 'author', type: 'string', isKey: true }
    ]
  },
  tpc: {
    part: [
      { name: 'p_partkey', type: 'int', isKey: true },
      { name: 'p_name', type: 'string', isKey: false },
      { name: 'p_mfgr', type: 'string', isKey: false },
      { name: 'p_brand', type: 'string', isKey: false },
      { name: 'p_type', type: 'string', isKey: false },
      { name: 'p_size', type: 'int', isKey: false },
      { name: 'p_container', type: 'string', isKey: false },
      { name: 'p_retailprice', type: 'float', isKey: false },
      { name: 'p_comment', type: 'string', isKey: false }
    ],
    supplier: [
      { name: 's_suppkey', type: 'int', isKey: true },
      { name: 's_name', type: 'string', isKey: false },
      { name: 's_address', type: 'string', isKey: false },
      { name: 's_nationkey', type: 'int', isKey: false },
      { name: 's_phone', type: 'string', isKey: false },
      { name: 's_acctbal', type: 'float', isKey: false },
      { name: 's_comment', type: 'string', isKey: false }
    ],
    partsupp: [
      { name: 'ps_partkey', type: 'int', isKey: true },
      { name: 'ps_suppkey', type: 'int', isKey: true },
      { name: 'ps_availqty', type: 'int', isKey: false },
      { name: 'ps_supplycost', type: 'float', isKey: false },
      { name: 'ps_comment', type: 'string', isKey: false }
    ],
    customer: [
      { name: 'c_custkey', type: 'int', isKey: true },
      { name: 'c_name', type: 'string', isKey: false },
      { name: 'c_address', type: 'string', isKey: false },
      { name: 'c_nationkey', type: 'int', isKey: false },
      { name: 'c_phone', type: 'string', isKey: false },
      { name: 'c_acctbal', type: 'float', isKey: false },
      { name: 'c_mktsegment', type: 'string', isKey: false },
      { name: 'c_comment', type: 'string', isKey: false }
    ],
    nation: [
      { name: 'n_nationkey', type: 'int', isKey: true },
      { name: 'n_name', type: 'string', isKey: false },
      { name: 'n_regionkey', type: 'int', isKey: false },
      { name: 'n_comment', type: 'string', isKey: false }
    ],
    region: [
      { name: 'r_regionkey', type: 'int', isKey: true },
      { name: 'r_name', type: 'string', isKey: false },
      { name: 'r_comment', type: 'string', isKey: false }
    ],
    lineitem: [
      { name: 'l_orderkey', type: 'int', isKey: true },
      { name: 'l_partkey', type: 'int', isKey: false },
      { name: 'l_suppkey', type: 'int', isKey: false },
      { name: 'l_linenumber', type: 'int', isKey: true },
      { name: 'l_quantity', type: 'float', isKey: false },
      { name: 'l_extendedprice', type: 'float', isKey: false },
      { name: 'l_discount', type: 'float', isKey: false },
      { name: 'l_tax', type: 'float', isKey: false },
      { name: 'l_returnflag', type: 'string', isKey: false },
      { name: 'l_linestatus', type: 'string', isKey: false },
      { name: 'l_shipdate', type: 'int', isKey: false },
      { name: 'l_commitdate', type: 'int', isKey: false },
      { name: 'l_receiptdate', type: 'int', isKey: false },
      { name: 'l_shipinstruct', type: 'string', isKey: false },
      { name: 'l_shipmode', type: 'string', isKey: false },
      { name: 'l_comment', type: 'string', isKey: false }
    ],
    orders: [
      { name: 'o_orderkey', type: 'int', isKey: true },
      { name: 'o_custkey', type: 'int', isKey: false },
      { name: 'o_orderstatus', type: 'string', isKey: false },
      { name: 'o_totalprice', type: 'float', isKey: false },
      { name: 'o_orderdate', type: 'int', isKey: false },
      { name: 'o_orderpriority', type: 'string', isKey: false },
      { name: 'o_clerk', type: 'string', isKey: false },
      { name: 'o_shippriority', type: 'int', isKey: false },
      { name: 'o_comment', type: 'string', isKey: false }
    ]
  }
};

function SchemaSelector({ activeSchema, onSchemaChange }) {
  const currentDbSchema = DB_SCHEMAS[activeSchema] || {};

  return (
    <div className="px-3 py-3 border-b border-slate-200">
      <div className="flex items-center gap-2 mb-2.5">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Schema</span>
        <div className="flex gap-1.5 ml-auto">
          {SCHEMAS.map((schema) => (
            <button
              key={schema.id}
              onClick={() => onSchemaChange(schema.id)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all duration-150 ${
                activeSchema === schema.id
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-white text-slate-600 border border-slate-200 hover:border-slate-300 hover:bg-slate-50'
              }`}
              title={schema.description}
            >
              {schema.name}
            </button>
          ))}
        </div>
      </div>

      {/* Schema Tables Display */}
      <div className="bg-white rounded-lg border border-slate-200 p-2.5 max-h-56 overflow-y-auto">
        <div className="space-y-1">
          {Object.entries(currentDbSchema).map(([tableName, columns]) => (
            <div key={tableName} className="text-xs leading-relaxed">
              <span className="font-bold text-slate-700">{tableName}</span>
              <span className="text-slate-400">(</span>
              {columns.map((col, idx) => (
                <span key={col.name}>
                  <span
                    className={`${col.isKey ? 'underline font-semibold text-blue-700' : 'text-slate-500'}`}
                  >
                    {col.name}
                  </span>
                  {idx < columns.length - 1 && <span className="text-slate-300">, </span>}
                </span>
              ))}
              <span className="text-slate-400">)</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

SchemaSelector.propTypes = {
  activeSchema: PropTypes.string.isRequired,
  onSchemaChange: PropTypes.func.isRequired
};

export default SchemaSelector;
