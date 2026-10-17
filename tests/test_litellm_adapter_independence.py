"""The adapter is a standalone package: what an operator installs into LiteLLM's environment must import nothing
but its own modules and nothing beyond the standard library, httpx and — in the LiteLLM glue
only — litellm. Read from the source, so a lazy import inside a function counts too."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "collieai_guardrail"
#: The modules allowed to import litellm: the glue LiteLLM loads. The core never does.
GLUE = frozenset({"guardrail"})
ALLOWED = set(sys.stdlib_module_names) | {"httpx", "collieai_guardrail"}


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: a relative import"  # LiteLLM may load a module by file path
            assert node.module is not None
            names.add(node.module.split(".")[0])
    return names


def test_the_package_imports_only_the_standard_library_httpx_and_itself() -> None:
    modules = sorted(PACKAGE.glob("*.py"))
    assert {m.stem for m in modules} >= {
        "__init__",
        "config",
        "eligibility",
        "payload",
        "transport",
        "verdict",
        "counters",
        "observer",
        "litellm_data",
        "heartbeat",
        "guardrail",
    }
    for module in modules:
        allowed = ALLOWED | ({"litellm"} if module.stem in GLUE else set())
        foreign = _imports(module) - allowed
        assert not foreign, f"{module.name} imports {sorted(foreign)}"
    assert not [p for p in PACKAGE.iterdir() if p.is_dir() and p.name != "__pycache__"], "the package is flat"


def test_only_the_glue_imports_litellm_and_nothing_imports_the_glue() -> None:
    """The core and its helpers run without LiteLLM, so they can be tested anywhere; the glue is the one module
    LiteLLM loads, and no other module depends on it."""
    importers = {m.stem for m in PACKAGE.glob("*.py") if "litellm" in _imports(m)}
    assert importers == GLUE
    for module in PACKAGE.glob("*.py"):
        if module.stem not in GLUE:
            assert "collieai_guardrail.guardrail" not in module.read_text(encoding="utf-8"), module.name
