"""Conservative evidence for explicit cProfile instructions.

This checks syntax in the saved answer. It never executes answer content.
"""
import ast
import re

_CPROFILE = re.compile(r"\bcprofile\b", re.I)
_TOOL_USE = re.compile(
    r"\b(?:use|using|with|via|run|invoke)\s+(?:the\s+)?`?cprofile\b|"
    r"\b(?:commands?|examples?|steps|code)\b", re.I,
)
_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)
_COMMAND = re.compile(r"^\s*(?:\$\s*)?(?:python(?:3|\.exe)?|py)\s+-m\s+cProfile\s+\S+\s*$", re.M)


def _profile_example(answer):
    for code in _CODE.findall(answer):
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            continue
        nodes = list(ast.walk(tree))
        imports = {alias.asname or alias.name for node in nodes if isinstance(node, ast.Import)
                   for alias in node.names if alias.name == "cProfile"}
        for node in nodes:
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (isinstance(node.func.value, ast.Name) and node.func.value.id in imports
                    and node.func.attr in {"run", "runctx"} and node.args
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                    and "(" in node.args[0].value):
                return True
    return bool(_COMMAND.search(answer))


def profiling_reuse_reason(original, incoming, answer):
    """Block newly introduced tool requirements and name-only answers."""
    if not (_CPROFILE.search(incoming) and _TOOL_USE.search(incoming)):
        return None
    if not (_CPROFILE.search(original) and _TOOL_USE.search(original)):
        return "added_cprofile_requirement"
    if not _profile_example(answer):
        return "answer_lacks_cprofile_usage"
    return None
