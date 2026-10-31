#!/usr/bin/env python3
"""Check documentation sources using only the Python standard library.

Run from any directory with ``python /path/to/repo/tools/check_docs.py``.
Neither the Sphinx configuration nor notebook cells are executed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


TUTORIALS = (
    "T1_Mouse_Embryo_E13_Spatial_ATAC_RNA",
    "T2_Mouse_Embryo_E13_Spatial_ATAC",
    "T3_Mouse_Brain_P22_Spatial_ATAC_RNA",
    "T4_Mouse_Brain_P22_Spatial_CUTTag_RNA_H3K27ac",
    "T5_Human_Tonsil_Spatial_ATAC",
    "T6_Mouse_Brain_Development_MISAR_seq",
)


def check_notebooks(root: Path) -> list[str]:
    """Return source-specific failures for linked notebooks and execution policy."""
    root = root.resolve()
    failures: list[str] = []
    config_path = root / "docs/conf.py"
    policies = {"nbsphinx_execute": "never", "nbsphinx_allow_errors": False}
    try:
        config = ast.parse(config_path.read_text(encoding="utf-8"))
        for name, expected in policies.items():
            assignments = []
            for node in config.body:
                targets = node.targets if isinstance(node, ast.Assign) else (
                    [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                    assignments.append(node)
            stores = [
                node for node in ast.walk(config)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name
            ]
            if not assignments or len(stores) != len(assignments):
                failures.append(f"docs/conf.py: {name} must be an explicit top-level literal")
                continue
            try:
                effective = ast.literal_eval(assignments[-1].value)
            except (ValueError, TypeError):
                effective = None
            if type(effective) is not type(expected) or effective != expected:
                failures.append(f"docs/conf.py: effective {name} must be {expected!r}, got {effective!r}")
    except (OSError, UnicodeError, SyntaxError) as exc:
        failures.append(f"docs/conf.py: cannot read configuration: {exc}")

    expected_links = {root / "docs/tutorials" / f"{name}.nblink" for name in TUTORIALS}
    actual_links = set((root / "docs").rglob("*.nblink"))
    for path in sorted(expected_links - actual_links):
        failures.append(f"{path.relative_to(root)}: missing notebook link")
    for path in sorted(actual_links - expected_links):
        failures.append(f"{path.relative_to(root)}: unexpected notebook link")

    for name in TUTORIALS:
        link_path = root / "docs/tutorials" / f"{name}.nblink"
        notebook_path = root / "tutorials" / f"{name}.ipynb"
        if link_path in actual_links:
            try:
                link = json.loads(link_path.read_text(encoding="utf-8"))
                target = link.get("path") if isinstance(link, dict) else None
                if not isinstance(target, str) or (link_path.parent / target).resolve() != notebook_path:
                    failures.append(f"{link_path.relative_to(root)}: target must be ../../tutorials/{name}.ipynb")
            except (OSError, UnicodeError, ValueError) as exc:
                failures.append(f"{link_path.relative_to(root)}: cannot read link: {exc}")
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
                raise ValueError("expected a notebook object with a cells list")
            locations = [("notebook", notebook)] + [
                (f"cell {index}", cell) for index, cell in enumerate(notebook["cells"], 1)
            ]
            for location, item in locations:
                if not isinstance(item, dict):
                    raise ValueError(f"{location} must be an object")
                metadata = item.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(f"{location} metadata must be an object")
                nbsphinx = metadata.get("nbsphinx", {})
                if not isinstance(nbsphinx, dict):
                    raise ValueError(f"{location} nbsphinx metadata must be an object")
                execute = nbsphinx.get("execute", "never")
                if execute != "never":
                    failures.append(f"{notebook_path.relative_to(root)}: {location} execute={execute!r} overrides the never policy")
                outputs = item.get("outputs", [])
                if not isinstance(outputs, list) or any(not isinstance(output, dict) for output in outputs):
                    raise ValueError(f"{location} outputs must be a list of objects")
                for output in outputs:
                    if output.get("output_type") == "error":
                        failures.append(f"{notebook_path.relative_to(root)}: {location} has error output ({output.get('ename', 'unknown error')})")
        except (OSError, UnicodeError, ValueError) as exc:
            failures.append(f"{notebook_path.relative_to(root)}: cannot read notebook: {exc}")
    return failures


class StaticResolutionError(ValueError):
    """A public function cannot be resolved by the bounded AST reader."""


class PublicFunctions:
    """Resolve root exports, relative reexports and simple aliases, never imports.

    Deliberately not a Python interpreter: computed exports, arbitrary decorators
    and dynamic wrappers need an explicit documentation manual-review comment.
    """

    def __init__(self, root: Path):
        self.root = root
        self.modules = {}
        tree = self.module("seareg")
        maps = [node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_EXPORTS" for t in node.targets)]
        if len(maps) != 1:
            raise StaticResolutionError("seareg._EXPORTS must be one literal map")
        self.exports = ast.literal_eval(maps[0])

    def module(self, module):
        if not module.startswith("seareg"):
            raise StaticResolutionError(f"outside package: {module}")
        if module not in self.modules:
            base = self.root.joinpath(*module.split("."))
            path = base / "__init__.py" if base.is_dir() else base.with_suffix(".py")
            self.modules[module] = ast.parse(path.read_text(encoding="utf-8"))
        return self.modules[module]

    def resolve(self, qualified):
        parts = qualified.split(".")
        if len(parts) == 2 and parts[0] == "seareg":
            if parts[1] not in self.exports:
                raise StaticResolutionError(f"not a root export: {qualified}")
            module, symbol = self.exports[parts[1]]
            return self.symbol("seareg." + module, symbol, set())
        return self.symbol(".".join(parts[:-1]), parts[-1], set())

    def symbol(self, module, symbol, seen):
        key = (module, symbol)
        if key in seen:
            raise StaticResolutionError(f"alias cycle: {module}.{symbol}")
        seen = seen | {key}
        tree = self.module(module)
        head, _, tail = symbol.partition(".")

        def binds(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.name == head
            if isinstance(node, ast.ImportFrom):
                return any((alias.asname or alias.name) == head for alias in node.names)
            if isinstance(node, ast.Assign):
                return any(isinstance(target, ast.Name) and target.id == head for target in node.targets)
            return False

        # Assignment order and captured aliases need execution to resolve in
        # general. Do not certify the first definition of a rebound symbol.
        if sum(binds(node) for node in tree.body) > 1:
            raise StaticResolutionError(f"multiple top-level bindings require manual review: {module}.{head}")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == head and not tail:
                decorators = {ast.unparse(d) for d in node.decorator_list}
                if decorators - {"validated_operation"}:
                    raise StaticResolutionError(f"unresolved decorators: {sorted(decorators)}")
                return node.args, False
            if isinstance(node, ast.ClassDef) and node.name == head and tail:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == tail:
                        decorators = {ast.unparse(d) for d in method.decorator_list}
                        if decorators - {"classmethod", "staticmethod", "validated_operation"}:
                            raise StaticResolutionError(f"unresolved method decorators: {sorted(decorators)}")
                        # Class.attribute binds cls only for @classmethod.
                        # A plain method accessed on the class is unbound.
                        return method.args, "classmethod" in decorators
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) != head:
                        continue
                    if node.level:
                        base = module.split(".")[:-node.level]
                        target = ".".join(base + (node.module or "").split("."))
                    else:
                        target = node.module or ""
                    return self.symbol(target, alias.name + ("." + tail if tail else ""), seen)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == head for t in node.targets):
                value = ast.unparse(node.value)
                if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value):
                    return self.symbol(module, value + ("." + tail if tail else ""), seen)
        raise StaticResolutionError(f"unresolved symbol: {module}.{symbol}")


def signature_parts(args, *, bound=False):
    """Represent Python parameter kinds and literal defaults semantically."""
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    parts = []

    def add(argument, kind, default=None):
        if default is None:
            value = ("required",)
        else:
            try:
                literal = ast.literal_eval(default)
            except (ValueError, TypeError) as exc:
                raise StaticResolutionError(f"nonliteral default: {ast.unparse(default)}") from exc
            value = ("default", type(literal).__name__, literal)
        parts.append((argument.arg, kind, value))

    for index, (argument, default) in enumerate(zip(positional, defaults)):
        if bound and index == 0:
            continue
        add(argument, "positional-only" if index < len(args.posonlyargs) else "positional", default)
    if args.vararg:
        add(args.vararg, "*args")
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        add(argument, "keyword-only", default)
    if args.kwarg:
        add(args.kwarg, "**kwargs")
    return parts


def check_api(root: Path) -> tuple[list[str], list[str], int]:
    """Return failures, explicit manual cases and automatically verified count.

    An RST comment ``.. api-signature-manual: qualified.name | reason`` is an
    explicit exception, not an automatic pass. Missing/stale exceptions fail.
    Repeated py:function names are overloads only when explicitly marked for
    manual review with that comment; they are never counted as automatic passes.
    """
    failures, manual, checked = [], [], 0
    try:
        functions = PublicFunctions(root)
    except (OSError, SyntaxError, ValueError, TypeError) as exc:
        return [f"API exports: {exc}"], [], 0
    for path in sorted((root / "docs").rglob("*.rst")):
        if "_build" in path.relative_to(root / "docs").parts:
            continue
        text = path.read_text(encoding="utf-8")
        exceptions = dict(re.findall(r"^\.\. api-signature-manual: ([\w.]+) \| (.+)$", text, re.M))
        used = set()
        directives = list(re.finditer(r"^[ \t]*\.\. py:function::[ \t]+([^\n]+)$", text, re.M))
        names = [match[1].split("(", 1)[0].strip() for match in directives]
        for match, name in zip(directives, names):
            if not name.startswith("seareg."):
                continue
            location = f"{path.relative_to(root)}:{text.count(chr(10), 0, match.start()) + 1} {name}"
            try:
                signature = match[1][len(name):].strip()
                documented = signature_parts(ast.parse(f"def documented{signature}: pass").body[0].args)
                source_args, bound = functions.resolve(name)
                source = signature_parts(source_args, bound=bound)
                # Variadic adapters cannot establish a composed public signature.
                variadic_wrapper = source_args.kwarg is not None and source != documented
                if name in exceptions:
                    if names.count(name) > 1:
                        used.add(name)
                        manual.append(f"{location}: declared overload: {exceptions[name]}")
                    elif not variadic_wrapper:
                        failures.append(f"{location}: stale manual exception; signature is statically comparable")
                    else:
                        used.add(name)
                        manual.append(f"{location}: {exceptions[name]}")
                    continue
                if variadic_wrapper:
                    raise StaticResolutionError("composed variadic wrapper requires an explicit manual-review comment")
                if documented != source:
                    failures.append(f"{location}: signature mismatch\n  documented: {documented}\n  source:     {source}")
                else:
                    checked += 1
            except (OSError, SyntaxError, ValueError, TypeError) as exc:
                if name in exceptions:
                    used.add(name)
                    manual.append(f"{location}: {exceptions[name]} (static limit: {exc})")
                else:
                    failures.append(f"{location}: {exc}")
        for name in exceptions.keys() - used:
            if not any(name in failure for failure in failures):
                failures.append(f"{path.relative_to(root)}: manual exception has no function directive: {name}")
    if checked == 0:
        failures.append("API: no function signatures were statically verified")
    return failures, manual, checked


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = check_notebooks(root)
    api_failures, manual, checked = check_api(root)
    failures.extend(api_failures)
    for case in manual:
        print(f"MANUAL: {case}")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(f"FAIL: documentation checks found {len(failures)} problem(s)")
        return 1
    print("PASS: six notebook links, readable notebooks, clean outputs and effective execution=never")
    print(f"PASS: {checked} API function signatures verified statically; {len(manual)} explicit manual-review case(s) remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
