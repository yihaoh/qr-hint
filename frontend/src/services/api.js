// API service for communicating with the backend

const API_BASE_URL = '/api';

/**
 * Sanitize SQL query by removing trailing semicolons
 * The Calcite parser used by QR-Hint doesn't support semicolons
 * @param {string} query - The SQL query
 * @returns {string} The sanitized query
 */
const sanitizeQuery = (query) => {
  return query.trim().replace(/;+$/, '');
};

/**
 * Repair a SQL query by comparing it with the correct query
 * @param {string} correctQuery - The correct SQL query
 * @param {string} incorrectQuery - The user's SQL query to be checked
 * @param {string} schema - The database schema to use (default: 'beers')
 * @returns {Promise<Object>} The repair result
 */
export const repairQuery = async (correctQuery, incorrectQuery, schema = 'beers') => {
  try {
    // Sanitize both queries to remove trailing semicolons
    const sanitizedCorrectQuery = sanitizeQuery(correctQuery);
    const sanitizedIncorrectQuery = sanitizeQuery(incorrectQuery);

    const response = await fetch(`${API_BASE_URL}/repair`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        correct_query: sanitizedCorrectQuery,
        incorrect_query: sanitizedIncorrectQuery,
        schema: schema,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to repair query');
    }

    return data;
  } catch (error) {
    console.error('Error calling repair API:', error);
    throw error;
  }
};

// /**
//  * Parse a SQL query
//  * @param {string} query - The SQL query to parse
//  * @returns {Promise<Object>} The parsed query information
//  */
// export const parseQuery = async (query) => {
//   try {
//     const response = await fetch(`${API_BASE_URL}/query/parse`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ query }),
//     });

//     const data = await response.json();

//     if (!response.ok) {
//       throw new Error(data.error || 'Failed to parse query');
//     }

//     return data;
//   } catch (error) {
//     console.error('Error calling parse API:', error);
//     throw error;
//   }
// };

// /**
//  * Generate a hint for a SQL query
//  * @param {string} query - The SQL query
//  * @param {Object} options - Optional hint generation options
//  * @returns {Promise<Object>} The generated hint
//  */
// export const generateHint = async (query, options = {}) => {
//   try {
//     const response = await fetch(`${API_BASE_URL}/query/hint`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ query, options }),
//     });

//     const data = await response.json();

//     if (!response.ok) {
//       throw new Error(data.error || 'Failed to generate hint');
//     }

//     return data;
//   } catch (error) {
//     console.error('Error calling hint API:', error);
//     throw error;
//   }
// };

/**
 * Generate AI-powered hint based on repair results (non-streaming)
 * @param {string} question - The SQL question/requirement
 * @param {string} incorrectQuery - The user's incorrect SQL query
 * @param {Array} repairs - Array of repair suggestions
 * @returns {Promise<Object>} The AI-generated hint
 */
export const generateAIHint = async (question, incorrectQuery, repairs) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-hint`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        incorrect_query: incorrectQuery,
        repairs,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to generate AI hint');
    }

    return data;
  } catch (error) {
    console.error('Error calling AI hint API:', error);
    throw error;
  }
};

/**
 * Generate AI-powered hint with streaming response
 * @param {string} question - The SQL question/requirement
 * @param {string} incorrectQuery - The user's incorrect SQL query
 * @param {Array} repairs - Array of repair suggestions
 * @param {Function} onChunk - Callback function called for each chunk of text
 * @param {Object} subqueryInfo - Optional subquery rewrite information from repair response
 * @returns {Promise<void>}
 */
export const generateAIHintStream = async (question, incorrectQuery, repairs, onChunk, subqueryInfo = null, level = 2) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-hint-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        incorrect_query: incorrectQuery,
        repairs,
        subquery_info: subqueryInfo,
        level,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to connect to AI hint stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // Decode the chunk
      const chunk = decoder.decode(value, { stream: true });

      // Process each line (SSE format)
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6); // Remove "data: " prefix

          try {
            const data = JSON.parse(jsonStr);

            if (data.chunk) {
              // Call the callback with the text chunk
              onChunk(data.chunk);
            } else if (data.done) {
              // Stream complete
              return;
            } else if (data.error) {
              // Error occurred
              throw new Error(data.error);
            }
          } catch (e) {
            // Skip malformed JSON
            if (e.message !== 'Failed to generate AI hint') {
              console.warn('Failed to parse SSE data:', jsonStr);
            } else {
              throw e;
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error calling AI hint stream API:', error);
    throw error;
  }
};

/**
 * Generate AI-powered hint for FROM clause issues with streaming response
 * @param {string} question - The SQL question/requirement
 * @param {string} incorrectQuery - The user's incorrect SQL query
 * @param {Object} fromIssues - Object with missing_tables, redundant_tables, wrong_count
 * @param {Function} onChunk - Callback function called for each chunk of text
 * @param {Object} subqueryInfo - Optional subquery rewrite information from repair response
 * @returns {Promise<void>}
 */
export const generateFromHintStream = async (question, incorrectQuery, fromIssues, onChunk, subqueryInfo = null, level = 2) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-hint-from-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        incorrect_query: incorrectQuery,
        from_issues: fromIssues,
        subquery_info: subqueryInfo,
        level,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to connect to FROM AI hint stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // Decode the chunk
      const chunk = decoder.decode(value, { stream: true });

      // Process each line (SSE format)
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6); // Remove "data: " prefix

          try {
            const data = JSON.parse(jsonStr);

            if (data.chunk) {
              // Call the callback with the text chunk
              onChunk(data.chunk);
            } else if (data.done) {
              // Stream complete
              return;
            } else if (data.error) {
              // Error occurred
              throw new Error(data.error);
            }
          } catch (e) {
            // Skip malformed JSON
            if (e.message !== 'Failed to generate FROM AI hint') {
              console.warn('Failed to parse SSE data:', jsonStr);
            } else {
              throw e;
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error calling FROM AI hint stream API:', error);
    throw error;
  }
};

/**
 * Generate AI-powered hint for GROUP BY clause issues with streaming response
 * @param {string} question - The SQL question/requirement
 * @param {string} incorrectQuery - The user's incorrect SQL query
 * @param {Object} groupByIssues - Object with incorrect and missing columns
 * @param {Function} onChunk - Callback function called for each chunk of text
 * @returns {Promise<void>}
 */
export const generateGroupByHintStream = async (question, incorrectQuery, groupByIssues, onChunk, level = 2) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-hint-groupby-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        incorrect_query: incorrectQuery,
        group_by_issues: groupByIssues,
        level,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to connect to GROUP BY AI hint stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // Decode the chunk
      const chunk = decoder.decode(value, { stream: true });

      // Process each line (SSE format)
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6); // Remove "data: " prefix

          try {
            const data = JSON.parse(jsonStr);

            if (data.chunk) {
              // Call the callback with the text chunk
              onChunk(data.chunk);
            } else if (data.done) {
              // Stream complete
              return;
            } else if (data.error) {
              // Error occurred
              throw new Error(data.error);
            }
          } catch (e) {
            // Skip malformed JSON
            if (e.message !== 'Failed to generate GROUP BY AI hint') {
              console.warn('Failed to parse SSE data:', jsonStr);
            } else {
              throw e;
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error calling GROUP BY AI hint stream API:', error);
    throw error;
  }
};

/**
 * Generate AI-powered hint for HAVING clause issues with streaming response
 * @param {string} question - The SQL question/requirement
 * @param {string} incorrectQuery - The user's incorrect SQL query
 * @param {Object} havingIssues - Object with repairs array
 * @param {Function} onChunk - Callback function called for each chunk of text
 * @returns {Promise<void>}
 */
export const generateHavingHintStream = async (question, incorrectQuery, havingIssues, onChunk, level = 2) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-hint-having-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        incorrect_query: incorrectQuery,
        having_issues: havingIssues,
        level,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to connect to HAVING AI hint stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // Decode the chunk
      const chunk = decoder.decode(value, { stream: true });

      // Process each line (SSE format)
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6); // Remove "data: " prefix

          try {
            const data = JSON.parse(jsonStr);

            if (data.chunk) {
              // Call the callback with the text chunk
              onChunk(data.chunk);
            } else if (data.done) {
              // Stream complete
              return;
            } else if (data.error) {
              // Error occurred
              throw new Error(data.error);
            }
          } catch (e) {
            // Skip malformed JSON
            if (e.message !== 'Failed to generate HAVING AI hint') {
              console.warn('Failed to parse SSE data:', jsonStr);
            } else {
              throw e;
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error calling HAVING AI hint stream API:', error);
    throw error;
  }
};

/**
 * Generate AI-powered hint for SELECT clause issues with streaming response
 * @param {string} question - The SQL question/requirement
 * @param {string} incorrectQuery - The user's incorrect SQL query
 * @param {Object} selectIssues - Object with incorrect, wrong_order, and missing columns
 * @param {Function} onChunk - Callback function called for each chunk of text
 * @param {Object} subqueryInfo - Optional subquery rewrite information from repair response
 * @returns {Promise<void>}
 */
export const generateSelectHintStream = async (question, incorrectQuery, selectIssues, onChunk, subqueryInfo = null, level = 2) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-hint-select-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        incorrect_query: incorrectQuery,
        select_issues: selectIssues,
        subquery_info: subqueryInfo,
        level,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to connect to SELECT AI hint stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // Decode the chunk
      const chunk = decoder.decode(value, { stream: true });

      // Process each line (SSE format)
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6); // Remove "data: " prefix

          try {
            const data = JSON.parse(jsonStr);

            if (data.chunk) {
              // Call the callback with the text chunk
              onChunk(data.chunk);
            } else if (data.done) {
              // Stream complete
              return;
            } else if (data.error) {
              // Error occurred
              throw new Error(data.error);
            }
          } catch (e) {
            // Skip malformed JSON
            if (e.message !== 'Failed to generate SELECT AI hint') {
              console.warn('Failed to parse SSE data:', jsonStr);
            } else {
              throw e;
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error calling SELECT AI hint stream API:', error);
    throw error;
  }
};

/**
 * Generate AI-powered duplicate check for subquery rewriting with streaming response
 * This checks if EXISTS/SOME subquery rewriting might cause duplicate row issues
 * @param {string} question - The SQL question/requirement
 * @param {string} userQuery - The user's SQL query
 * @param {string} correctQuery - The correct SQL query
 * @param {Object} subqueryInfo - Subquery rewrite information from repair response
 * @param {Function} onChunk - Callback function called for each chunk of text
 * @returns {Promise<void>}
 */
export const generateDuplicateCheckStream = async (question, userQuery, correctQuery, subqueryInfo, onChunk) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-hint-duplicate-check-stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        user_query: userQuery,
        correct_query: correctQuery,
        subquery_info: subqueryInfo,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to connect to duplicate check stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      // Decode the chunk
      const chunk = decoder.decode(value, { stream: true });

      // Process each line (SSE format)
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6); // Remove "data: " prefix

          try {
            const data = JSON.parse(jsonStr);

            if (data.chunk) {
              // Call the callback with the text chunk
              onChunk(data.chunk);
            } else if (data.done) {
              // Stream complete
              return;
            } else if (data.error) {
              // Error occurred
              throw new Error(data.error);
            }
          } catch (e) {
            // Skip malformed JSON
            if (e.message !== 'Failed to generate duplicate check') {
              console.warn('Failed to parse SSE data:', jsonStr);
            } else {
              throw e;
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error calling duplicate check stream API:', error);
    throw error;
  }
};

/**
 * Test endpoint to check if backend is running
 * @returns {Promise<Object>} The test response
 */
export const testBackend = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/test-print`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error testing backend:', error);
    throw error;
  }
};
