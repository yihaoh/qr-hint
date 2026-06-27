"""
DeepSeek AI Service for generating SQL query repair hints.
This service provides intelligent, educational hints without directly revealing the answer.
Supports streaming responses for better user experience.
"""

import os
import json
import requests
from typing import List, Dict, Optional, Generator


class DeepSeekService:
    """Service for generating AI-powered hints using DeepSeek API."""

    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.api_base = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
        self.model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

    def generate_hint(
        self,
        question: str,
        incorrect_query: str,
        repairs: List[Dict]
    ) -> str:
        """
        Generate an educational hint based on repair suggestions.

        Args:
            question: The SQL question/requirement
            incorrect_query: The user's incorrect SQL query
            repairs: List of repair suggestions with format:
                     {
                         'repair_site': str or None,
                         'fix': str or None,
                         'repair_site_size': int,
                         'fix_size': int
                     }

        Returns:
            A hint string that guides the user without revealing the answer
        """
        if not repairs or len(repairs) == 0:
            return "Your query looks correct! Well done!"

        # Build the prompt for DeepSeek
        prompt = self._build_hint_prompt(question, incorrect_query, repairs)

        try:
            response = self._call_deepseek_api(prompt)
            return response
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            # Fallback to basic hint if API fails
            return self._generate_fallback_hint(repairs)

    def _build_exists_context(self, exists_info: Optional[Dict]) -> str:
        """Build context string explaining EXISTS rewrite if applicable."""
        if not exists_info or not exists_info.get('has_exists', False):
            return ""

        original = exists_info.get('original_query', '')
        rewritten = exists_info.get('rewritten_query', '')

        return f"""
Note: The student used an EXISTS subquery. For analysis, the query has been transformed:
- Original query with EXISTS: {original}
- Equivalent rewritten query (EXISTS converted to JOIN): {rewritten}

When providing hints, explain in terms of the ORIGINAL EXISTS syntax that the student wrote.
Help them understand how EXISTS works and what conditions it checks."""

    def _get_hint_instructions(self, level: int, clause: str = 'WHERE', exists_info=None) -> tuple:
        """Return (task_instructions, max_tokens) for the given hint level and clause type."""
        exists_note = ""
        if exists_info and exists_info.get('has_exists'):
            exists_note = f"\n- If the issue relates to EXISTS, explain in terms of the EXISTS subquery syntax the student used"

        if level == 1:
            instructions = f"""Your task: Provide a very brief directional hint (1 sentence max) that:
- Only indicates the {clause} clause needs attention, without any specifics
- Uses vague guidance like "revisit", "reconsider", "take another look at"
- Does NOT mention any specific columns, operators, values, or table names
- Does NOT explain what's wrong - just point to where to look{exists_note}

Response (one sentence only):"""
            return instructions, 80

        elif level == 2:
            instructions = f"""Your task: Provide a moderate hint (2-3 sentences) that:
- Identifies the TYPE of issue in the {clause} clause (e.g., "wrong operator", "missing condition") without the exact fix
- Guides the student toward the right thinking direction
- Uses language like "consider", "think about", "check if"
- Does NOT show corrected SQL or reveal exact values that need changing{exists_note}

Response (hint only, no extra explanation):"""
            return instructions, 150

        else:  # level 3
            instructions = f"""Your task: Provide a detailed, near-answer hint (3-4 sentences) that:
- Clearly describes what's wrong in the {clause} clause and why it produces incorrect results
- Explains the correct approach in plain English
- May reference specific parts of the student's query that need changing
- Makes the fix obvious WITHOUT writing the actual corrected SQL code{exists_note}

Response (hint only, no extra explanation):"""
            return instructions, 300

# where
    def _build_hint_prompt(
        self,
        question: str,
        incorrect_query: str,
        repairs: List[Dict],
        exists_info: Optional[Dict] = None,
        level: int = 2
    ) -> str:
        """Build the prompt for DeepSeek API."""

        # Extract repair information
        repair_descriptions = []
        for i, repair in enumerate(repairs, 1):
            repair_site = repair.get('repair_site', '')
            fix = repair.get('fix', '')

            # Handle None values
            if repair_site == 'None' or repair_site is None:
                repair_site = '(missing clause)'
            if fix == 'None' or fix is None:
                fix = '(should be removed)'

            repair_descriptions.append(
                f"Issue {i}: The query has '{repair_site}' but it should be '{fix}'"
            )

        repairs_text = "\n".join(repair_descriptions)

        # Build EXISTS context if applicable
        exists_context = self._build_exists_context(exists_info)

        # Use original query for display if EXISTS was rewritten
        display_query = incorrect_query
        if exists_info and exists_info.get('has_exists', False):
            display_query = exists_info.get('original_query', incorrect_query)

        # Get level-specific instructions
        instructions, _ = self._get_hint_instructions(level, 'WHERE', exists_info)

        # Create the prompt
        prompt = f"""You are a helpful SQL tutor. A student is learning SQL and made a mistake in their query.

Question: {question}

Student's Query:
{display_query}
{exists_context}
Detected Issues:
{repairs_text}

{instructions}"""

        return prompt

    def generate_hint_stream(
        self,
        question: str,
        incorrect_query: str,
        repairs: List[Dict],
        exists_info: Optional[Dict] = None,
        level: int = 2
    ) -> Generator[str, None, None]:
        """
        Generate an educational hint with streaming response.

        Args:
            question: The SQL question/requirement
            incorrect_query: The user's incorrect SQL query
            repairs: List of repair suggestions
            exists_info: Optional dict with EXISTS rewrite information
            level: Hint detail level (1=direction, 2=pinpoint, 3=near-answer)

        Yields:
            Chunks of the hint text as they are generated
        """
        if not repairs or len(repairs) == 0:
            yield "Your query looks correct! Well done!"
            return

        # Build the prompt for DeepSeek
        prompt = self._build_hint_prompt(question, incorrect_query, repairs, exists_info, level)
        _, max_tokens = self._get_hint_instructions(level, 'WHERE', exists_info)

        try:
            yield from self._call_deepseek_api_stream(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"Error calling DeepSeek API: {str(e)}")
            # Fallback to basic hint if API fails
            yield self._generate_fallback_hint(repairs)

    def _call_deepseek_api(self, prompt: str) -> str:
        """Make non-streaming API call to DeepSeek (deprecated, kept for compatibility)."""

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a patient SQL tutor who gives hints without revealing answers directly. Keep responses concise and educational.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': 150
        }

        response = requests.post(
            f'{self.api_base}/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        hint = result['choices'][0]['message']['content'].strip()
        return hint

    def _call_deepseek_api_stream(self, prompt: str, max_tokens: int = 150) -> Generator[str, None, None]:
        """Make streaming API call to DeepSeek."""

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a patient SQL tutor who gives hints without revealing answers directly. Keep responses concise and educational.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': max_tokens,
            'stream': True  # Enable streaming
        }

        response = requests.post(
            f'{self.api_base}/chat/completions',
            headers=headers,
            json=data,
            timeout=30,
            stream=True  # Enable streaming response
        )

        response.raise_for_status()

        # Process the streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')

                # Skip empty lines and comments
                if not line.strip() or line.startswith(':'):
                    continue

                # Remove "data: " prefix
                if line.startswith('data: '):
                    line = line[6:]

                # Check for end of stream
                if line.strip() == '[DONE]':
                    break

                try:
                    # Parse JSON chunk
                    chunk = json.loads(line)

                    # Extract content delta
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')

                        if content:
                            yield content

                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue

    def _generate_fallback_hint(self, repairs: List[Dict]) -> str:
        """Generate a basic hint if API call fails."""

        if not repairs:
            return "Your query looks correct!"

        repair = repairs[0]  # Focus on first repair
        repair_site = repair.get('repair_site', 'this part')

        if repair_site == 'None' or repair_site is None:
            return "Something is missing in your query. Check if all required clauses are present."

        # Generic educational hint
        return f"Look carefully at the part containing '{repair_site[:30]}...' - think about the SQL syntax and logic here. Does it correctly express what the question is asking for?"

    # ========== FROM Clause Hint Methods ==========

    def _build_from_hint_prompt(
        self,
        question: str,
        incorrect_query: str,
        from_issues: Dict,
        exists_info: Optional[Dict] = None,
        level: int = 2
    ) -> str:
        """Build the prompt for FROM clause hints."""

        missing_tables = from_issues.get('missing_tables', [])
        redundant_tables = from_issues.get('redundant_tables', [])
        wrong_count = from_issues.get('wrong_count', False)

        # Build issue description with specific table names
        issue_descriptions = []
        issue_num = 1

        for table in missing_tables:
            issue_descriptions.append(f"Issue {issue_num}: The FROM clause is missing table '{table}'")
            issue_num += 1

        for table in redundant_tables:
            issue_descriptions.append(f"Issue {issue_num}: The FROM clause has extra table '{table}' that shouldn't be there")
            issue_num += 1

        if wrong_count and not missing_tables and not redundant_tables:
            issue_descriptions.append(f"Issue {issue_num}: The number of tables in the FROM clause doesn't match what's expected")

        issues_description = "\n".join(issue_descriptions) if issue_descriptions else "- FROM clause structure issue detected"

        # Build EXISTS context if applicable
        exists_context = self._build_exists_context(exists_info)

        # Use original query for display if EXISTS was rewritten
        display_query = incorrect_query
        if exists_info and exists_info.get('has_exists', False):
            display_query = exists_info.get('original_query', incorrect_query)

        # Get level-specific instructions
        instructions, _ = self._get_hint_instructions(level, 'FROM', exists_info)

        prompt = f"""You are a helpful SQL tutor. A student is learning SQL and made a mistake in their FROM clause.

Question: {question}

Student's Query:
{display_query}
{exists_context}
Detected FROM Clause Issues:
{issues_description}

{instructions}"""

        return prompt

    def generate_from_hint_stream(
        self,
        question: str,
        incorrect_query: str,
        from_issues: Dict,
        exists_info: Optional[Dict] = None,
        level: int = 2
    ) -> Generator[str, None, None]:
        """
        Generate an educational hint for FROM clause issues with streaming response.
        """
        has_issues = (
            from_issues.get('missing_tables', []) or
            from_issues.get('redundant_tables', []) or
            from_issues.get('wrong_count', False)
        )

        if not has_issues:
            yield "Your FROM clause looks correct! Well done!"
            return

        prompt = self._build_from_hint_prompt(question, incorrect_query, from_issues, exists_info, level)
        _, max_tokens = self._get_hint_instructions(level, 'FROM', exists_info)

        try:
            yield from self._call_deepseek_api_stream(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"Error calling DeepSeek API for FROM hint: {str(e)}")
            yield self._generate_from_fallback_hint(from_issues)

    def _generate_from_fallback_hint(self, from_issues: Dict) -> str:
        """Generate a basic FROM clause hint if API call fails."""

        missing = from_issues.get('missing_tables', [])
        redundant = from_issues.get('redundant_tables', [])

        if missing:
            return "Your query seems to be missing some tables. Think about which tables contain the data you need to answer the question. Consider what information you're trying to retrieve and where that data is stored."
        elif redundant:
            return "Your query includes some extra tables that may not be needed. Review the question and think about which tables are actually necessary to get the required data."
        else:
            return "Check your FROM clause carefully. Make sure you have all the tables you need, and only the tables you need, to answer the question."

    # ========== GROUP BY Clause Hint Methods ==========

    def _build_group_by_hint_prompt(
        self,
        question: str,
        incorrect_query: str,
        group_by_issues: Dict,
        level: int = 2
    ) -> str:
        """Build the prompt for GROUP BY clause hints."""

        incorrect_items = group_by_issues.get('incorrect', [])
        missing_items = group_by_issues.get('missing', [])

        issue_descriptions = []
        issue_num = 1

        for column in incorrect_items:
            issue_descriptions.append(f"Issue {issue_num}: The GROUP BY clause has '{column}' which shouldn't be there")
            issue_num += 1

        for column in missing_items:
            issue_descriptions.append(f"Issue {issue_num}: The GROUP BY clause is missing '{column}'")
            issue_num += 1

        issues_description = "\n".join(issue_descriptions) if issue_descriptions else "- GROUP BY clause structure issue detected"

        instructions, _ = self._get_hint_instructions(level, 'GROUP BY')

        prompt = f"""You are a helpful SQL tutor. A student is learning SQL and made a mistake in their GROUP BY clause.

Question: {question}

Student's Query:
{incorrect_query}

Detected GROUP BY Clause Issues:
{issues_description}

{instructions}"""

        return prompt

    def generate_group_by_hint_stream(
        self,
        question: str,
        incorrect_query: str,
        group_by_issues: Dict,
        level: int = 2
    ) -> Generator[str, None, None]:
        """Generate an educational hint for GROUP BY clause issues with streaming response."""
        has_issues = (
            group_by_issues.get('incorrect', []) or
            group_by_issues.get('missing', [])
        )

        if not has_issues:
            yield "Your GROUP BY clause looks correct! Well done!"
            return

        prompt = self._build_group_by_hint_prompt(question, incorrect_query, group_by_issues, level)
        _, max_tokens = self._get_hint_instructions(level, 'GROUP BY')

        try:
            yield from self._call_deepseek_api_stream(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"Error calling DeepSeek API for GROUP BY hint: {str(e)}")
            yield self._generate_group_by_fallback_hint(group_by_issues)

    def _generate_group_by_fallback_hint(self, group_by_issues: Dict) -> str:
        """Generate a basic GROUP BY clause hint if API call fails."""

        incorrect = group_by_issues.get('incorrect', [])
        missing = group_by_issues.get('missing', [])

        if missing:
            return "Your GROUP BY clause seems to be missing some columns. Remember that all non-aggregated columns in your SELECT clause must also appear in GROUP BY. Think about which columns you're selecting and whether they need to be grouped."
        elif incorrect:
            return "Your GROUP BY clause includes some columns that may not be needed. Review which columns you're actually using in your SELECT clause and whether grouping by them makes sense for the question."
        else:
            return "Check your GROUP BY clause carefully. Make sure you're grouping by the right columns based on what the question asks for."

    # ========== HAVING Clause Hint Methods ==========

    def _build_having_hint_prompt(
        self,
        question: str,
        incorrect_query: str,
        having_issues: Dict,
        level: int = 2
    ) -> str:
        """Build the prompt for HAVING clause hints."""

        repairs = having_issues.get('repairs', [])
        distinct_mismatches = having_issues.get('distinct_mismatches', [])

        repair_descriptions = []
        issue_num = 1

        for repair in repairs:
            repair_site = repair.get('repair_site', '')
            fix = repair.get('fix', '')

            if repair_site == 'None' or repair_site is None:
                repair_site = '(missing condition)'
            if fix == 'None' or fix is None:
                fix = '(should be removed)'

            repair_descriptions.append(
                f"Issue {issue_num}: The HAVING clause has '{repair_site}' but it should be '{fix}'"
            )
            issue_num += 1

        for mismatch in distinct_mismatches:
            q1_distinct = mismatch.get('q1_distinct', False)
            q1_aggregate = mismatch.get('q1_aggregate', 'aggregate function')
            q2_aggregate = mismatch.get('q2_aggregate', 'aggregate function')

            if q1_distinct:
                repair_descriptions.append(
                    f"Issue {issue_num}: The HAVING clause should use DISTINCT inside the aggregate function (e.g., {q1_aggregate}(DISTINCT ...)) but your query uses {q2_aggregate}(...) without DISTINCT"
                )
            else:
                repair_descriptions.append(
                    f"Issue {issue_num}: Your query uses DISTINCT inside an aggregate function ({q2_aggregate}(DISTINCT ...)) but it should not use DISTINCT"
                )
            issue_num += 1

        repairs_text = "\n".join(repair_descriptions) if repair_descriptions else "- HAVING clause issue detected"

        instructions, _ = self._get_hint_instructions(level, 'HAVING')

        prompt = f"""You are a helpful SQL tutor. A student is learning SQL and made a mistake in their HAVING clause.

Question: {question}

Student's Query:
{incorrect_query}

Detected HAVING Clause Issues:
{repairs_text}

{instructions}"""

        return prompt

    def generate_having_hint_stream(
        self,
        question: str,
        incorrect_query: str,
        having_issues: Dict,
        level: int = 2
    ) -> Generator[str, None, None]:
        """Generate an educational hint for HAVING clause issues with streaming response."""
        repairs = having_issues.get('repairs', [])
        distinct_mismatches = having_issues.get('distinct_mismatches', [])

        if not repairs and not distinct_mismatches:
            yield "Your HAVING clause looks correct! Well done!"
            return

        prompt = self._build_having_hint_prompt(question, incorrect_query, having_issues, level)
        _, max_tokens = self._get_hint_instructions(level, 'HAVING')

        try:
            yield from self._call_deepseek_api_stream(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"Error calling DeepSeek API for HAVING hint: {str(e)}")
            yield self._generate_having_fallback_hint(having_issues)

    def _generate_having_fallback_hint(self, having_issues: Dict) -> str:
        """Generate a basic HAVING clause hint if API call fails."""

        repairs = having_issues.get('repairs', [])

        has_missing = any(r.get('repair_site') is None and r.get('fix') for r in repairs)
        has_incorrect = any(r.get('repair_site') and r.get('fix') is None for r in repairs)

        if has_missing:
            return "Your HAVING clause seems to be missing some conditions. Think about what aggregate conditions you need to filter your grouped results. Consider what the question is asking about the grouped data."
        elif has_incorrect:
            return "Your HAVING clause includes some conditions that may not be correct. Review the question and think about which aggregate conditions are actually needed to filter your results."
        else:
            return "Check your HAVING clause carefully. Make sure you're applying the right conditions to your grouped results based on what the question asks for."

    # ========== SELECT Clause Hint Methods ==========

    def _build_select_hint_prompt(
        self,
        question: str,
        incorrect_query: str,
        select_issues: Dict,
        exists_info: Optional[Dict] = None,
        level: int = 2
    ) -> str:
        """Build the prompt for SELECT clause hints."""

        incorrect = select_issues.get('incorrect', [])
        wrong_order = select_issues.get('wrong_order', [])
        missing = select_issues.get('missing', [])
        distinct_mismatches = select_issues.get('distinct_mismatches', [])

        issue_descriptions = []
        issue_num = 1

        for item in incorrect:
            issue_descriptions.append(
                f"Issue {issue_num}: The SELECT clause has '{item}' which is incorrect or shouldn't be there"
            )
            issue_num += 1

        for item in wrong_order:
            issue_descriptions.append(
                f"Issue {issue_num}: The column '{item}' is in the wrong position in the SELECT clause"
            )
            issue_num += 1

        for item in missing:
            issue_descriptions.append(
                f"Issue {issue_num}: The SELECT clause is missing '{item}'"
            )
            issue_num += 1

        for mismatch in distinct_mismatches:
            mismatch_type = mismatch.get('type', '')
            q1_distinct = mismatch.get('q1_distinct', False)

            if mismatch_type == 'query_level':
                if q1_distinct:
                    issue_descriptions.append(
                        f"Issue {issue_num}: The query should use SELECT DISTINCT to eliminate duplicate rows, but your query doesn't use DISTINCT"
                    )
                else:
                    issue_descriptions.append(
                        f"Issue {issue_num}: Your query uses SELECT DISTINCT, but DISTINCT is not needed here"
                    )
            else:
                q1_aggregate = mismatch.get('q1_aggregate', 'aggregate function')
                q2_aggregate = mismatch.get('q2_aggregate', 'aggregate function')

                if q1_distinct:
                    issue_descriptions.append(
                        f"Issue {issue_num}: The SELECT clause should use DISTINCT inside the aggregate function (e.g., {q1_aggregate}(DISTINCT ...)) but your query uses {q2_aggregate}(...) without DISTINCT"
                    )
                else:
                    issue_descriptions.append(
                        f"Issue {issue_num}: Your query uses DISTINCT inside an aggregate function ({q2_aggregate}(DISTINCT ...)) but it should not use DISTINCT"
                    )
            issue_num += 1

        issues_text = "\n".join(issue_descriptions) if issue_descriptions else "- SELECT clause issue detected"

        exists_context = self._build_exists_context(exists_info)

        display_query = incorrect_query
        if exists_info and exists_info.get('has_exists', False):
            display_query = exists_info.get('original_query', incorrect_query)

        instructions, _ = self._get_hint_instructions(level, 'SELECT', exists_info)

        prompt = f"""You are a helpful SQL tutor. A student is learning SQL and made a mistake in their SELECT clause.

Question: {question}

Student's Query:
{display_query}
{exists_context}
Detected SELECT Clause Issues:
{issues_text}

{instructions}"""

        return prompt

    def generate_select_hint_stream(
        self,
        question: str,
        incorrect_query: str,
        select_issues: Dict,
        exists_info: Optional[Dict] = None,
        level: int = 2
    ) -> Generator[str, None, None]:
        """Generate an educational hint for SELECT clause issues with streaming response."""
        has_issues = (
            select_issues.get('incorrect', []) or
            select_issues.get('wrong_order', []) or
            select_issues.get('missing', []) or
            select_issues.get('distinct_mismatches', [])
        )

        if not has_issues:
            yield "Your SELECT clause looks correct! Well done!"
            return

        prompt = self._build_select_hint_prompt(question, incorrect_query, select_issues, exists_info, level)
        _, max_tokens = self._get_hint_instructions(level, 'SELECT', exists_info)

        try:
            yield from self._call_deepseek_api_stream(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"Error calling DeepSeek API for SELECT hint: {str(e)}")
            yield self._generate_select_fallback_hint(select_issues)

    def _generate_select_fallback_hint(self, select_issues: Dict) -> str:
        """Generate a basic SELECT clause hint if API call fails."""

        incorrect = select_issues.get('incorrect', [])
        wrong_order = select_issues.get('wrong_order', [])
        missing = select_issues.get('missing', [])

        if missing:
            return "Your SELECT clause seems to be missing some columns or expressions. Think about what information the question is asking you to return. Consider whether you need any aggregate functions or specific columns."
        elif incorrect:
            return "Your SELECT clause includes some columns or expressions that may not be correct. Review the question and think about exactly what data should be returned."
        elif wrong_order:
            return "The columns in your SELECT clause may be in the wrong order. Check the question to see if there's a specific order expected for the output columns."
        else:
            return "Check your SELECT clause carefully. Make sure you're returning the right columns and expressions based on what the question asks for."


    # ========== Subquery Duplicate Check Methods ==========

    def _build_duplicate_check_prompt(
        self,
        question: str,
        user_query: str,
        correct_query: str,
        subquery_info: Dict
    ) -> str:
        """Build the prompt for checking duplicate row issues in subquery rewriting."""

        user_has_subquery = subquery_info.get('user_has_subquery', False)
        correct_has_subquery = subquery_info.get('correct_has_subquery', False)
        user_original = subquery_info.get('original_query', user_query)
        user_rewritten = subquery_info.get('rewritten_query', user_query)
        correct_original = subquery_info.get('correct_original', correct_query)
        correct_rewritten = subquery_info.get('correct_rewritten', correct_query)

        # Determine specific subquery type used
        has_exists = subquery_info.get('has_exists', False)
        has_some = subquery_info.get('has_some', False)

        # Get specific subquery type name for user's query
        def get_subquery_type_name(has_exists_flag, has_some_flag):
            if has_exists_flag and has_some_flag:
                return "EXISTS and SOME/ANY"
            elif has_exists_flag:
                return "EXISTS"
            elif has_some_flag:
                return "SOME/ANY"
            else:
                return "subquery"

        user_subquery_type = get_subquery_type_name(has_exists, has_some) if user_has_subquery else None
        # For correct query, we need to detect its type from the original query
        correct_subquery_type = None
        if correct_has_subquery and correct_original:
            correct_upper = correct_original.upper()
            correct_has_exists_kw = 'EXISTS' in correct_upper
            correct_has_some_kw = 'SOME' in correct_upper or 'ANY' in correct_upper
            correct_subquery_type = get_subquery_type_name(correct_has_exists_kw, correct_has_some_kw)

        # Determine the case (1-4)
        # Case 1: correct has subquery, user doesn't
        # Case 2: correct doesn't have subquery, user does
        # Case 3: both have subquery
        # Case 4: neither has subquery (shouldn't reach here)

        if correct_has_subquery and not user_has_subquery:
            case_num = 1
            case_description = f"The CORRECT answer uses {correct_subquery_type} subquery, but the USER wrote a JOIN-style query."
            suggestion_guidance = f"""
If there's a duplicate issue (not equivalent):
- Do NOT reveal the correct SQL syntax directly
- Explain WHY {correct_subquery_type} subquery style might be more appropriate (e.g., "{correct_subquery_type} preserves duplicate rows from the outer table, while JOIN+DISTINCT removes all duplicates")
- Use hints like "Consider how {correct_subquery_type} handles duplicate rows differently from JOIN" or "Think about whether the outer table might have legitimate duplicates that should be preserved"
- Guide the student to understand the semantic difference without giving the answer

If no duplicate issue (equivalent), confirm their JOIN approach produces the same results."""
        elif not correct_has_subquery and user_has_subquery:
            case_num = 2
            case_description = f"The USER used {user_subquery_type} subquery, but the CORRECT answer uses JOIN-style query."
            suggestion_guidance = f"""
If there's a duplicate issue (not equivalent):
- Do NOT reveal the correct SQL syntax directly
- Explain WHY JOIN style might be more appropriate for this particular question
- Use hints like "Consider whether you need to access columns from the inner query tables in your final result" or "Think about what information you need to return - does {user_subquery_type} give you access to it?"
- If the question requires data from the subquery tables, hint: "{user_subquery_type} only checks for existence/comparison but doesn't bring back columns from the subquery - consider what columns your SELECT needs"
- Guide the student to understand when {user_subquery_type} vs JOIN is appropriate without giving the answer

If no duplicate issue (equivalent), confirm their {user_subquery_type} approach is acceptable and produces the same results."""
        else:  # both have subquery (case 3)
            case_num = 3
            case_description = f"BOTH the correct answer and user's query use {user_subquery_type or correct_subquery_type} subquery style."
            suggestion_guidance = """
Since both use subquery style, they should be semantically equivalent. Just confirm the query is correct."""

        # Build context about the queries
        context_parts = []
        if user_has_subquery:
            context_parts.append(f"""User's query uses {user_subquery_type} subquery style:
- Original: {user_original}
- Rewritten to JOIN (with SELECT DISTINCT added): {user_rewritten}""")
        else:
            context_parts.append(f"User's query (JOIN style): {user_query}")

        if correct_has_subquery:
            context_parts.append(f"""Correct answer uses {correct_subquery_type} subquery style:
- Original: {correct_original}
- Rewritten to JOIN (with SELECT DISTINCT added): {correct_rewritten}""")
        else:
            context_parts.append(f"Correct answer (JOIN style): {correct_query}")

        queries_context = "\n\n".join(context_parts)

        prompt = f"""You are an expert SQL analyst. We have a system that rewrites EXISTS/SOME/ANY subqueries to equivalent JOINs. During this rewrite, SELECT DISTINCT is automatically added to eliminate duplicate rows that the JOIN might produce.

CASE {case_num}: {case_description}

IMPORTANT SEMANTIC ISSUE TO CHECK:
When an EXISTS/SOME subquery is used, if the OUTER query's table contains duplicate rows, EXISTS will preserve those duplicates (because EXISTS only checks for existence, it doesn't multiply rows).

However, when we rewrite to JOIN + DISTINCT:
- The JOIN might multiply rows OR not multiply rows depending on the data
- The DISTINCT then removes ALL duplicates, including duplicates that existed in the ORIGINAL outer table

This means: If the outer table has legitimate duplicate rows that should appear in the result, the subquery version preserves them, but the rewritten JOIN + DISTINCT version removes them.

Question: {question}

{queries_context}

Your task:
1. Analyze whether this semantic mismatch (duplicate row handling) could be an issue for this specific query
2. Consider if the outer table might legitimately contain duplicate rows that should be preserved
3. Determine if the two query styles would produce different results
{suggestion_guidance}

Respond with a JSON object in this exact format:
{{
    "has_duplicate_issue": true/false,
    "case": {case_num},
    "explanation": "Brief explanation of your analysis about the duplicate row handling",
    "suggestion": "Your suggestion to the student based on the analysis"
}}

Response (JSON only, no markdown):"""

        return prompt

    def _call_deepseek_api_stream_extended(self, prompt: str, max_tokens: int = 500) -> Generator[str, None, None]:
        """Make streaming API call to DeepSeek with extended token limit for JSON responses."""

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert SQL analyst. Always respond with valid JSON only, no markdown formatting.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.3,  # Lower temperature for more consistent JSON output
            'max_tokens': max_tokens,
            'stream': True
        }

        response = requests.post(
            f'{self.api_base}/chat/completions',
            headers=headers,
            json=data,
            timeout=60,  # Longer timeout for extended responses
            stream=True
        )

        response.raise_for_status()

        # Process the streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')

                if not line.strip() or line.startswith(':'):
                    continue

                if line.startswith('data: '):
                    line = line[6:]

                if line.strip() == '[DONE]':
                    break

                try:
                    chunk = json.loads(line)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    def check_subquery_duplicate_stream(
        self,
        question: str,
        user_query: str,
        correct_query: str,
        subquery_info: Dict
    ) -> Generator[str, None, None]:
        """
        Check for duplicate row issues when rewriting subqueries to JOINs.

        The rewrite adds SELECT DISTINCT, which may incorrectly remove duplicate
        rows that existed in the original outer table when using EXISTS/SOME.

        Cases:
        1. Correct has subquery, user doesn't → suggest EXISTS if not equivalent
        2. Correct doesn't have subquery, user does → suggest not using EXISTS if not equivalent
        3. Both have subquery → confirm correctness

        Args:
            question: The SQL question/requirement
            user_query: The user's SQL query
            correct_query: The correct SQL query
            subquery_info: Dict with subquery rewrite information

        Yields:
            Chunks of the analysis/hint text as they are generated
        """
        # Build the prompt for DeepSeek
        prompt = self._build_duplicate_check_prompt(question, user_query, correct_query, subquery_info)

        try:
            # Collect the full response first since we need to parse JSON
            # Use extended token limit for JSON response
            full_response = ""
            for chunk in self._call_deepseek_api_stream_extended(prompt, max_tokens=500):
                full_response += chunk

            # Try to parse the JSON response
            try:
                # Clean up the response (remove any markdown formatting if present)
                clean_response = full_response.strip()
                if clean_response.startswith("```"):
                    # Handle ```json ... ``` format
                    parts = clean_response.split("```")
                    if len(parts) >= 2:
                        clean_response = parts[1]
                        if clean_response.startswith("json"):
                            clean_response = clean_response[4:]
                    clean_response = clean_response.strip()

                result = json.loads(clean_response)

                # Extract the suggestion to yield
                suggestion = result.get('suggestion', 'Analysis complete.')
                has_duplicate_issue = result.get('has_duplicate_issue', False)
                case_num = result.get('case', 0)

                if has_duplicate_issue:
                    # There's a potential duplicate issue - yield a warning hint
                    yield "⚠️ Potential Semantic Difference Detected\n\n"
                    yield suggestion
                else:
                    # No issue - yield success message
                    if case_num == 3:
                        # Both use subquery - straightforward success
                        yield "✅ Query Correct!\n\n"
                    else:
                        # Different styles but equivalent
                        yield "✅ Query Semantically Equivalent!\n\n"
                    yield suggestion

            except json.JSONDecodeError as e:
                # If JSON parsing fails, log the error and provide a fallback message
                print(f"JSON parse error: {str(e)}")
                print(f"Raw response: {full_response[:500]}...")
                yield "✅ Your query structure matches the expected answer. "
                yield "Note: Could not complete detailed semantic analysis."

        except Exception as e:
            print(f"Error calling DeepSeek API for duplicate check: {str(e)}")
            yield "Unable to complete the semantic equivalence check. Please review your query manually."


# Singleton instance
_deepseek_service = None

def get_deepseek_service() -> DeepSeekService:
    """Get or create DeepSeek service instance."""
    global _deepseek_service
    if _deepseek_service is None:
        _deepseek_service = DeepSeekService()
    return _deepseek_service
