"""The compression pipeline must never read gold answers (specs.md §1 generality rule).

The pipeline modules may not import :mod:`evaluate` or mention the answers
directory in code. Docstrings are exempt, because describing the rule is fine.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PIPELINE = ["text.py", "bloom.py", "retriever.py", "driver.py", "llm.py"]


def code_strings_and_imports(path: Path) -> tuple[list[str], list[str]]:
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
    }
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    imports = [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
    imports += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    return strings, imports


@pytest.mark.parametrize("module", PIPELINE)
def test_pipeline_module_never_touches_answers(module):
    strings, imports = code_strings_and_imports(SRC / module)
    assert "evaluate" not in imports
    assert not [s for s in strings if "answers" in s.lower() or "gold" in s.lower()]
