#!/usr/bin/env python3
"""Index current, non-test bridge changes against the fixed upstream base."""

import argparse
import ast
import re
import subprocess
from pathlib import Path


HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
EXCLUDED_PREFIXES = ("tests/", "docs/codex-bridge/", ".agents/")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def included(path: str) -> bool:
    return not path.startswith(EXCLUDED_PREFIXES) and not path.startswith(
        "docs/codex-bridge-audit-"
    )


def symbols(root: Path, path: str, head: str) -> list[tuple[int, int, str, int]]:
    if Path(path).suffix != ".py":
        return []
    tree = ast.parse(git(root, "show", f"{head}:{path}"), filename=path)
    spans: list[tuple[int, int, str, int]] = []

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = ".".join((*parents, node.name))
            first = min([node.lineno, *(deco.lineno for deco in node.decorator_list)])
            spans.append((first, node.end_lineno, name, len(parents) + 1))
            parents = (*parents, node.name)
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree, ())
    return spans


def symbol_at(line: int, spans: list[tuple[int, int, str, int]]) -> str:
    matches = (s for s in spans if s[0] <= line <= s[1])
    best = max(matches, key=lambda s: (s[3], -(s[1] - s[0])), default=None)
    return best[2] if best else "<module>"


def changed_ranges(diff: str) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    path: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+++ /dev/null"):
            path = None
        else:
            match = HUNK.match(line)
            if match and path and included(path):
                start = int(match.group(1))
                length = int(match.group(2) or 1)
                if length:
                    result.setdefault(path, []).append((start, start + length - 1))
    return result


def build(root: Path, base: str, head: str) -> str:
    stats = git(root, "diff", "--numstat", "--no-renames", "--no-ext-diff", base, head, "--")
    expected: dict[str, int] = {}
    for row in stats.splitlines():
        added, _deleted, path = row.split("\t", 2)
        if included(path):
            if not added.isdigit():
                raise ValueError(f"Cannot index binary change: {path}")
            expected[path] = int(added)

    diff = git(root, "diff", "--no-renames", "--no-ext-diff", "--no-color", "--unified=0", base, head, "--")
    ranges = changed_ranges(diff)
    rows = ["file\tsymbol\tstart\tend\tadded_lines"]
    counted: dict[str, int] = {}
    for path, intervals in ranges.items():
        spans = symbols(root, path, head)
        for first, last in intervals:
            group_start = first
            previous_symbol = symbol_at(first, spans)
            for line in range(first + 1, last + 2):
                name = symbol_at(line, spans) if line <= last else None
                if name != previous_symbol:
                    size = line - group_start
                    rows.append(f"{path}\t{previous_symbol}\t{group_start}\t{line - 1}\t{size}")
                    counted[path] = counted.get(path, 0) + size
                    group_start = line
                    previous_symbol = name
    if counted != {path: count for path, count in expected.items() if count}:
        raise ValueError(f"Coverage mismatch: indexed={counted}, git={expected}")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="fixed upstream common ancestor")
    parser.add_argument("--head", default="HEAD", help="source revision to index")
    parser.add_argument("--output", default="docs/codex-bridge/coverage.tsv")
    parser.add_argument("--check", action="store_true", help="compare without writing")
    args = parser.parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    output = root / args.output
    content = build(root, args.base, args.head)
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Outdated coverage: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    rows = content.count("\n") - 1
    total = sum(int(row.rsplit("\t", 1)[1]) for row in content.splitlines()[1:])
    print(f"{rows} spans, {total} added lines, {output}")


if __name__ == "__main__":
    main()
