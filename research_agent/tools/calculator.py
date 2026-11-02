"""Safe arithmetic: evaluates strings like "(255324 - 240893) / 240893 * 100" without eval()."""

import ast
import operator

from langchain.tools import tool

MAX_LENGTH = 200
ALLOWED_CHARS = set("0123456789.+-*/() ")

# AST operator node -> the Python function that does it. Anything not listed is rejected.
BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _walk(node):
    """Compute the value of one AST node, allowing only numbers and + - * /."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPS:
        return BINARY_OPS[type(node.op)](_walk(node.left), _walk(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](_walk(node.operand))
    op = getattr(
        node, "op", None
    )  # a BinOp/UnaryOp whose operator is not allowed, e.g. ** is ast.Pow
    if op is not None:
        raise ValueError(f"Operator not allowed: {type(op).__name__}. Use only + - * /")
    raise ValueError(f"Not allowed in an expression: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Return the value of an arithmetic expression as a float.

    Raises ValueError with a readable message for anything that is not plain arithmetic
    (names, calls, **, //, too long, divide by zero, ...).
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string.")
    if len(expression) > MAX_LENGTH:
        raise ValueError(f"Expression is longer than {MAX_LENGTH} characters.")

    # Step 1: cheap character check, so letters, quotes, _ and commas never reach the parser.
    bad = sorted(set(expression) - ALLOWED_CHARS)
    if bad:
        raise ValueError(
            f"Characters not allowed: {' '.join(bad)}. Use only numbers, + - * / ( ) and ."
        )

    # Step 2: parse into a tree. mode="eval" means one expression only, no statements.
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError(f"Not valid arithmetic: {expression!r}") from None

    # Step 3: walk the tree ourselves; nothing is ever passed to eval().
    try:
        return float(_walk(tree.body))
    except ZeroDivisionError:
        raise ValueError("Division by zero.") from None


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Use this for every calculation (growth, margins, ratios, sums, averages). Never do maths yourself.

    Args:
        expression: plain numbers with + - * / and parentheses, decimals allowed.
            Not allowed: ** % //, functions, variable names, units, the ₹ sign, or commas
            inside numbers (write 255324, not 2,55,324).

    Formulas:
        Growth %        = (new - old) / old * 100     e.g. (255324 - 240893) / 240893 * 100
        Margin %        = part / revenue * 100        e.g. 62293 / 255324 * 100
        Debt-to-equity  = debt / equity               e.g. 9392 / 94756

    Returns the number as text. The result has no units, so keep track of them yourself.
    If it starts with "Error:", fix the expression and try again.
    """
    try:
        return str(safe_eval(expression))
    except ValueError as e:
        return f"Error: {e}"
