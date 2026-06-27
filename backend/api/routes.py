# path: backend/api/routes.py
from flask import Blueprint, request, jsonify, Response, stream_with_context
from services.qr_hint_service import QRHintService
from services.deepseek_service import get_deepseek_service
import json
# from models.query import QueryRequest, QueryResponse
# from utils.helpers import validate_request

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
qr_hint_service = QRHintService()
deepseek_service = None  # Lazy initialization


@api_bp.route('/test-print', methods=['GET'])
def test_print():
    """Test endpoint to verify backend is running"""
    return jsonify({
        'ok': True,
        'message': 'backend is up'
    }), 200


# @api_bp.route('/query/parse', methods=['POST'])
# def parse_query():
#     """Parse SQL query and return structure information"""
#     try:
#         data = request.get_json()

#         # Validate request
#         if not data or 'query' not in data:
#             return jsonify({
#                 'ok': False,
#                 'error': 'Missing query parameter'
#             }), 400

#         query_text = data['query']

#         # Parse query using service
#         result = qr_hint_service.parse_query(query_text)

#         return jsonify({
#             'ok': True,
#             'data': result
#         }), 200

#     except Exception as e:
#         return jsonify({
#             'ok': False,
#             'error': str(e)
#         }), 500


# @api_bp.route('/query/hint', methods=['POST'])
# def generate_hint():
#     """Generate query hint based on input query"""
#     try:
#         data = request.get_json()

#         # Validate request
#         if not data or 'query' not in data:
#             return jsonify({
#                 'ok': False,
#                 'error': 'Missing query parameter'
#             }), 400

#         query_text = data['query']
#         options = data.get('options', {})

#         # Generate hint using service
#         result = qr_hint_service.generate_hint(query_text, options)

#         return jsonify({
#             'ok': True,
#             'data': result
#         }), 200

#     except Exception as e:
#         return jsonify({
#             'ok': False,
#             'error': str(e)
#         }), 500


@api_bp.route('/repair', methods=['POST'])
def repair_query():
    """Analyze two SQL queries and suggest repairs"""
    try:
        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        if 'correct_query' not in data or 'incorrect_query' not in data:
            return jsonify({
                'ok': False,
                'error': 'Both correct_query and incorrect_query parameters are required'
            }), 400

        correct_query = data['correct_query']
        incorrect_query = data['incorrect_query']
        schema = data.get('schema', 'beers')  # Default to 'beers' schema

        # Validate queries are not empty
        if not correct_query or not incorrect_query:
            return jsonify({
                'ok': False,
                'error': 'Query parameters cannot be empty'
            }), 400

        # Call repair service with schema
        result = qr_hint_service.repair_query(correct_query, incorrect_query, schema)
        

        if result.get('ok', False):
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@api_bp.route('/ai-hint', methods=['POST'])
def generate_ai_hint():
    """Generate AI-powered hint based on repair results (non-streaming)"""
    try:
        global deepseek_service

        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        if 'question' not in data or 'incorrect_query' not in data or 'repairs' not in data:
            return jsonify({
                'ok': False,
                'error': 'question, incorrect_query, and repairs parameters are required'
            }), 400

        question = data['question']
        incorrect_query = data['incorrect_query']
        repairs = data['repairs']

        # Validate repairs is a list
        if not isinstance(repairs, list):
            return jsonify({
                'ok': False,
                'error': 'repairs must be a list'
            }), 400

        # Lazy initialize DeepSeek service
        if deepseek_service is None:
            try:
                deepseek_service = get_deepseek_service()
            except ValueError as e:
                return jsonify({
                    'ok': False,
                    'error': f'DeepSeek service not configured: {str(e)}'
                }), 503

        # Generate AI hint
        hint = deepseek_service.generate_hint(question, incorrect_query, repairs)

        return jsonify({
            'ok': True,
            'hint': hint
        }), 200

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@api_bp.route('/ai-hint-stream', methods=['POST'])
def generate_ai_hint_stream():
    """Generate AI-powered hint with streaming response using Server-Sent Events"""
    try:
        global deepseek_service

        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        if 'question' not in data or 'incorrect_query' not in data or 'repairs' not in data:
            return jsonify({
                'ok': False,
                'error': 'question, incorrect_query, and repairs parameters are required'
            }), 400

        question = data['question']
        incorrect_query = data['incorrect_query']
        repairs = data['repairs']
        subquery_info = data.get('subquery_info', None)  # Optional subquery rewrite info
        level = data.get('level', 2)  # Hint detail level (1=direction, 2=pinpoint, 3=near-answer)

        # Validate repairs is a list
        if not isinstance(repairs, list):
            return jsonify({
                'ok': False,
                'error': 'repairs must be a list'
            }), 400

        # Lazy initialize DeepSeek service
        if deepseek_service is None:
            try:
                deepseek_service = get_deepseek_service()
            except ValueError as e:
                return jsonify({
                    'ok': False,
                    'error': f'DeepSeek service not configured: {str(e)}'
                }), 503

        # Generator function for SSE
        def generate():
            try:
                for chunk in deepseek_service.generate_hint_stream(question, incorrect_query, repairs, subquery_info, level=level):
                    # Format as Server-Sent Event
                    sse_data = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield sse_data

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                # Send error as SSE
                error_data = f"data: {json.dumps({'error': str(e)})}\n\n"
                yield error_data

        # Return SSE response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@api_bp.route('/ai-hint-from-stream', methods=['POST'])
def generate_ai_hint_from_stream():
    """Generate AI-powered hint for FROM clause issues with streaming response"""
    try:
        global deepseek_service

        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        if 'question' not in data or 'incorrect_query' not in data or 'from_issues' not in data:
            return jsonify({
                'ok': False,
                'error': 'question, incorrect_query, and from_issues parameters are required'
            }), 400

        question = data['question']
        incorrect_query = data['incorrect_query']
        from_issues = data['from_issues']
        subquery_info = data.get('subquery_info', None)  # Optional subquery rewrite info
        level = data.get('level', 2)

        # Validate from_issues is a dict
        if not isinstance(from_issues, dict):
            return jsonify({
                'ok': False,
                'error': 'from_issues must be an object'
            }), 400

        # Lazy initialize DeepSeek service
        if deepseek_service is None:
            try:
                deepseek_service = get_deepseek_service()
            except ValueError as e:
                return jsonify({
                    'ok': False,
                    'error': f'DeepSeek service not configured: {str(e)}'
                }), 503

        # Generator function for SSE
        def generate():
            try:
                for chunk in deepseek_service.generate_from_hint_stream(question, incorrect_query, from_issues, subquery_info, level=level):
                    # Format as Server-Sent Event
                    sse_data = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield sse_data

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                # Send error as SSE
                error_data = f"data: {json.dumps({'error': str(e)})}\n\n"
                yield error_data

        # Return SSE response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@api_bp.route('/ai-hint-groupby-stream', methods=['POST'])
def generate_ai_hint_groupby_stream():
    """Generate AI-powered hint for GROUP BY clause issues with streaming response"""
    try:
        global deepseek_service

        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        if 'question' not in data or 'incorrect_query' not in data or 'group_by_issues' not in data:
            return jsonify({
                'ok': False,
                'error': 'question, incorrect_query, and group_by_issues parameters are required'
            }), 400

        question = data['question']
        incorrect_query = data['incorrect_query']
        group_by_issues = data['group_by_issues']
        level = data.get('level', 2)

        # Validate group_by_issues is a dict
        if not isinstance(group_by_issues, dict):
            return jsonify({
                'ok': False,
                'error': 'group_by_issues must be an object'
            }), 400

        # Lazy initialize DeepSeek service
        if deepseek_service is None:
            try:
                deepseek_service = get_deepseek_service()
            except ValueError as e:
                return jsonify({
                    'ok': False,
                    'error': f'DeepSeek service not configured: {str(e)}'
                }), 503

        # Generator function for SSE
        def generate():
            try:
                for chunk in deepseek_service.generate_group_by_hint_stream(question, incorrect_query, group_by_issues, level=level):
                    # Format as Server-Sent Event
                    sse_data = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield sse_data

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                # Send error as SSE
                error_data = f"data: {json.dumps({'error': str(e)})}\n\n"
                yield error_data

        # Return SSE response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@api_bp.route('/ai-hint-having-stream', methods=['POST'])
def generate_ai_hint_having_stream():
    """Generate AI-powered hint for HAVING clause issues with streaming response"""
    try:
        global deepseek_service

        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        if 'question' not in data or 'incorrect_query' not in data or 'having_issues' not in data:
            return jsonify({
                'ok': False,
                'error': 'question, incorrect_query, and having_issues parameters are required'
            }), 400

        question = data['question']
        incorrect_query = data['incorrect_query']
        having_issues = data['having_issues']
        level = data.get('level', 2)

        # Validate having_issues is a dict
        if not isinstance(having_issues, dict):
            return jsonify({
                'ok': False,
                'error': 'having_issues must be an object'
            }), 400

        # Lazy initialize DeepSeek service
        if deepseek_service is None:
            try:
                deepseek_service = get_deepseek_service()
            except ValueError as e:
                return jsonify({
                    'ok': False,
                    'error': f'DeepSeek service not configured: {str(e)}'
                }), 503

        # Generator function for SSE
        def generate():
            try:
                for chunk in deepseek_service.generate_having_hint_stream(question, incorrect_query, having_issues, level=level):
                    # Format as Server-Sent Event
                    sse_data = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield sse_data

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                # Send error as SSE
                error_data = f"data: {json.dumps({'error': str(e)})}\n\n"
                yield error_data

        # Return SSE response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@api_bp.route('/ai-hint-select-stream', methods=['POST'])
def generate_ai_hint_select_stream():
    """Generate AI-powered hint for SELECT clause issues with streaming response"""
    try:
        global deepseek_service

        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        if 'question' not in data or 'incorrect_query' not in data or 'select_issues' not in data:
            return jsonify({
                'ok': False,
                'error': 'question, incorrect_query, and select_issues parameters are required'
            }), 400

        question = data['question']
        incorrect_query = data['incorrect_query']
        select_issues = data['select_issues']
        subquery_info = data.get('subquery_info', None)  # Optional subquery rewrite info
        level = data.get('level', 2)

        # Validate select_issues is a dict
        if not isinstance(select_issues, dict):
            return jsonify({
                'ok': False,
                'error': 'select_issues must be an object'
            }), 400

        # Lazy initialize DeepSeek service
        if deepseek_service is None:
            try:
                deepseek_service = get_deepseek_service()
            except ValueError as e:
                return jsonify({
                    'ok': False,
                    'error': f'DeepSeek service not configured: {str(e)}'
                }), 503

        # Generator function for SSE
        def generate():
            try:
                for chunk in deepseek_service.generate_select_hint_stream(question, incorrect_query, select_issues, subquery_info, level=level):
                    # Format as Server-Sent Event
                    sse_data = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield sse_data

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                # Send error as SSE
                error_data = f"data: {json.dumps({'error': str(e)})}\n\n"
                yield error_data

        # Return SSE response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


@api_bp.route('/ai-hint-duplicate-check-stream', methods=['POST'])
def generate_ai_duplicate_check_stream():
    """Check for duplicate row issues when rewriting subqueries with streaming response"""
    try:
        global deepseek_service

        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({
                'ok': False,
                'error': 'Request body is required'
            }), 400

        required_fields = ['question', 'user_query', 'correct_query', 'subquery_info']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'ok': False,
                    'error': f'{field} parameter is required'
                }), 400

        question = data['question']
        user_query = data['user_query']
        correct_query = data['correct_query']
        subquery_info = data['subquery_info']

        # Validate subquery_info is a dict
        if not isinstance(subquery_info, dict):
            return jsonify({
                'ok': False,
                'error': 'subquery_info must be an object'
            }), 400

        # Lazy initialize DeepSeek service
        if deepseek_service is None:
            try:
                deepseek_service = get_deepseek_service()
            except ValueError as e:
                return jsonify({
                    'ok': False,
                    'error': f'DeepSeek service not configured: {str(e)}'
                }), 503

        # Generator function for SSE
        def generate():
            try:
                for chunk in deepseek_service.check_subquery_duplicate_stream(
                    question, user_query, correct_query, subquery_info
                ):
                    # Format as Server-Sent Event
                    sse_data = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    yield sse_data

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

            except Exception as e:
                # Send error as SSE
                error_data = f"data: {json.dumps({'error': str(e)})}\n\n"
                yield error_data

        # Return SSE response
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        return jsonify({
            'ok': False,
            'error': str(e)
        }), 500


# @api_bp.route('/query/optimize', methods=['POST'])
# def optimize_query():
#     """Optimize query with generated hints"""
#     try:
#         data = request.get_json()

#         if not data or 'query' not in data:
#             return jsonify({
#                 'ok': False,
#                 'error': 'Missing query parameter'
#             }), 400

#         query_text = data['query']
#         hint = data.get('hint')

#         # Optimize using service
#         result = qr_hint_service.optimize_query(query_text, hint)

#         return jsonify({
#             'ok': True,
#             'data': result
#         }), 200

#     except Exception as e:
#         return jsonify({
#             'ok': False,
#             'error': str(e)
#         }), 500
