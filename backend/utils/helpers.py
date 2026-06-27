# path: backend/utils/helpers.py
from functools import wraps
from flask import request, jsonify
import re


def validate_request(required_fields):
    """
    Decorator to validate request data

    Args:
        required_fields: List of required field names

    Usage:
        @validate_request(['query', 'database'])
        def my_route():
            # Your code here
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json()

            if not data:
                return jsonify({
                    'ok': False,
                    'error': 'No data provided'
                }), 400

            missing_fields = [field for field in required_fields if field not in data]

            if missing_fields:
                return jsonify({
                    'ok': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400

            return f(*args, **kwargs)

        return decorated_function
    return decorator


def sanitize_sql(query: str) -> str:
    """
    Basic SQL query sanitization

    Args:
        query: SQL query string

    Returns:
        Sanitized query string
    """
    if not query:
        return ""

    # Remove leading/trailing whitespace
    query = query.strip()

    # Remove multiple spaces
    query = re.sub(r'\s+', ' ', query)

    return query


def validate_sql_query(query: str) -> tuple[bool, str]:
    """
    Validate SQL query format

    Args:
        query: SQL query string

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Query cannot be empty"

    if len(query) < 5:
        return False, "Query is too short"

    # Check for basic SQL keywords
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
    query_upper = query.upper()

    if not any(keyword in query_upper for keyword in sql_keywords):
        return False, "Query must contain a valid SQL keyword"

    return True, ""


def format_error_response(error: Exception) -> dict:
    """
    Format error response

    Args:
        error: Exception object

    Returns:
        Dictionary with error information
    """
    return {
        'ok': False,
        'error': str(error),
        'type': type(error).__name__
    }


def format_success_response(data: dict) -> dict:
    """
    Format success response

    Args:
        data: Response data

    Returns:
        Dictionary with success response
    """
    return {
        'ok': True,
        'data': data
    }


def parse_database_url(url: str) -> dict:
    """
    Parse database URL into components

    Args:
        url: Database connection URL

    Returns:
        Dictionary with database connection components
    """
    # Simple regex pattern for postgresql://user:pass@host:port/dbname
    pattern = r'(\w+)://(?:([^:]+):([^@]+)@)?([^:]+):(\d+)/(.+)'
    match = re.match(pattern, url)

    if not match:
        return {}

    return {
        'driver': match.group(1),
        'user': match.group(2),
        'password': match.group(3),
        'host': match.group(4),
        'port': int(match.group(5)),
        'database': match.group(6)
    }
