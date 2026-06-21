#!/usr/bin/env python3
"""
Workato recipe optimizer/analyzer.

The tool reads a Workato project export from a .zip file or extracted folder,
analyzes every *.recipe.json file, and produces actionable optimization
recommendations. It is intentionally conservative: by default it reports what
to improve instead of rewriting recipe logic.

Examples:
  python3 workato_recipe_optimizer.py som_package.zip
  python3 workato_recipe_optimizer.py som_package.zip --markdown report.md --json-report report.json
  python3 workato_recipe_optimizer.py som_package.zip --optimized-zip som_package.compact.zip
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import re
import statistics
import sys
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


RECIPE_SUFFIX = ".recipe.json"
VOLATILE_STEP_KEYS = {
    "as",
    "number",
    "uuid",
    "title",
    "comment",
    "description",
    "extended_input_schema",
    "extended_output_schema",
    "visible_config_fields",
}
BLOCK_KEYS = ("block",)
FORMULA_RE = re.compile(r"(?P<formula>(?:#\{)?_dp\('.*?'\)(?:\})?|lookup\([^)]*\))")
LOOKUP_RE = re.compile(r"lookup\(\s*['\"](?P<table>[^'\"]+)['\"]")
DATA_PILL_RE = re.compile(r"_dp\('(?P<payload>\{.*?\})'\)")


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: bytes


@dataclass
class Step:
    recipe_path: str
    recipe_name: str
    path: str
    depth: int
    number: Any
    keyword: str
    provider: str
    name: str
    alias: str
    description: str
    comment: str
    skip: bool
    raw: dict[str, Any]

    @property
    def label(self) -> str:
        bits = [str(self.number) if self.number not in (None, "") else "?"]
        action = "/".join(bit for bit in (self.provider, self.name) if bit)
        if action:
            bits.append(action)
        if self.keyword:
            bits.append(f"({self.keyword})")
        return " ".join(bits)


@dataclass
class Finding:
    severity: str
    category: str
    title: str
    detail: str
    recipe: str | None = None
    step: str | None = None
    impact: int = 1

    def sort_key(self) -> tuple[int, int, str, str]:
        severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        return (
            severity_rank.get(self.severity, 9),
            -self.impact,
            self.recipe or "",
            self.title,
        )


@dataclass
class RecipeStats:
    path: str
    name: str
    version: Any
    concurrency: Any
    bytes_on_disk: int
    steps: list[Step] = field(default_factory=list)
    keyword_counts: Counter[str] = field(default_factory=Counter)
    provider_counts: Counter[str] = field(default_factory=Counter)
    skipped_steps: list[Step] = field(default_factory=list)
    recipe_function_calls: list[tuple[str, str, str]] = field(default_factory=list)
    lookup_tables: Counter[str] = field(default_factory=Counter)
    data_pill_count: int = 0
    formula_count: int = 0
    max_depth: int = 0
    schema_bytes: int = 0
    long_string_fields: list[tuple[str, int, str]] = field(default_factory=list)
    duplicate_local_steps: list[tuple[str, list[Step]]] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def complexity_score(self) -> int:
        return (
            self.step_count
            + self.max_depth * 4
            + self.keyword_counts.get("repeat", 0) * 5
            + self.keyword_counts.get("foreach", 0) * 5
            + self.keyword_counts.get("if", 0) * 3
            + self.keyword_counts.get("else", 0) * 2
            + self.keyword_counts.get("catch", 0) * 4
            + self.skipped_count * 2
            + min(self.formula_count // 5, 25)
        )

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_steps)


@dataclass
class ProjectAnalysis:
    source: str
    recipe_stats: list[RecipeStats]
    findings: list[Finding]
    recipe_paths: set[str]
    available_assets: set[str]
    duplicate_step_groups: list[tuple[str, list[Step]]]


class WorkatoPackage:
    def __init__(self, source: Path):
        self.source = source

    def files(self) -> list[SourceFile]:
        if self.source.is_file() and self.source.suffix.lower() == ".zip":
            return self._zip_files()
        if self.source.is_dir():
            return self._directory_files()
        raise ValueError(f"Input must be a .zip file or directory: {self.source}")

    def _zip_files(self) -> list[SourceFile]:
        files: list[SourceFile] = []
        with zipfile.ZipFile(self.source) as package:
            for info in package.infolist():
                if info.is_dir():
                    continue
                files.append(SourceFile(info.filename, package.read(info)))
        return files

    def _directory_files(self) -> list[SourceFile]:
        files: list[SourceFile] = []
        for path in sorted(self.source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.source).as_posix()
                files.append(SourceFile(relative, path.read_bytes()))
        return files


def load_json_file(file: SourceFile) -> Any | None:
    try:
        return json.loads(file.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse {file.path}: {exc}") from exc


def is_recipe_file(path: str) -> bool:
    return path.endswith(RECIPE_SUFFIX)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def iter_steps(root: Any, recipe_path: str, recipe_name: str) -> Iterable[Step]:
    code = root.get("code") if isinstance(root, dict) else None
    if isinstance(code, dict):
        yield from _walk_step(code, recipe_path, recipe_name, "code", 0)


def _walk_step(
    step: dict[str, Any],
    recipe_path: str,
    recipe_name: str,
    path: str,
    depth: int,
) -> Iterable[Step]:
    yield Step(
        recipe_path=recipe_path,
        recipe_name=recipe_name,
        path=path,
        depth=depth,
        number=step.get("number"),
        keyword=as_text(step.get("keyword")),
        provider=as_text(step.get("provider")),
        name=as_text(step.get("name")),
        alias=as_text(step.get("as")),
        description=strip_html(as_text(step.get("description"))),
        comment=as_text(step.get("comment")),
        skip=bool(step.get("skip")),
        raw=step,
    )
    for key in BLOCK_KEYS:
        block = step.get(key)
        if isinstance(block, list):
            for index, child in enumerate(block):
                if isinstance(child, dict):
                    child_path = f"{path}.{key}[{index}]"
                    yield from _walk_step(child, recipe_path, recipe_name, child_path, depth + 1)


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def normalized_step_fingerprint(step: Step) -> str:
    cleaned = remove_volatile_fields(step.raw)
    payload = json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{step.provider}|{step.name}|{step.keyword}|{digest}"


def remove_volatile_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_volatile_fields(val)
            for key, val in sorted(value.items())
            if key not in VOLATILE_STEP_KEYS and key not in BLOCK_KEYS
        }
    if isinstance(value, list):
        return [remove_volatile_fields(item) for item in value]
    return value


def collect_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from collect_strings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from collect_strings(child, f"{path}[{index}]")


def json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def analyze_package(files: list[SourceFile], source: str) -> ProjectAnalysis:
    available_assets = {file.path for file in files}
    recipe_files = [file for file in files if is_recipe_file(file.path)]
    recipe_paths = {file.path for file in recipe_files}

    stats: list[RecipeStats] = []
    findings: list[Finding] = []
    all_fingerprints: defaultdict[str, list[Step]] = defaultdict(list)
    recipe_by_name: dict[str, str] = {}

    for file in recipe_files:
        root = load_json_file(file)
        if not isinstance(root, dict):
            findings.append(
                Finding(
                    "HIGH",
                    "parse",
                    "Recipe JSON is not an object",
                    "Workato recipe exports should have an object at the file root.",
                    recipe=file.path,
                )
            )
            continue

        recipe_name = as_text(root.get("name") or Path(file.path).stem.replace(".recipe", ""))
        recipe_by_name[recipe_name] = file.path
        recipe_stats = RecipeStats(
            path=file.path,
            name=recipe_name,
            version=root.get("version"),
            concurrency=root.get("concurrency"),
            bytes_on_disk=len(file.content),
        )
        steps = list(iter_steps(root, file.path, recipe_name))
        recipe_stats.steps = steps
        recipe_stats.max_depth = max((step.depth for step in steps), default=0)

        local_fingerprints: defaultdict[str, list[Step]] = defaultdict(list)
        for step in steps:
            recipe_stats.keyword_counts[step.keyword or "unknown"] += 1
            recipe_stats.provider_counts[step.provider or "unknown"] += 1
            if step.skip:
                recipe_stats.skipped_steps.append(step)
            fingerprint = normalized_step_fingerprint(step)
            local_fingerprints[fingerprint].append(step)
            all_fingerprints[fingerprint].append(step)
            collect_recipe_function_call(step, recipe_stats)

        for fingerprint, matching_steps in local_fingerprints.items():
            if len(matching_steps) > 1 and is_meaningful_duplicate(matching_steps):
                recipe_stats.duplicate_local_steps.append((fingerprint, matching_steps))

        for path, text in collect_strings(root):
            recipe_stats.data_pill_count += len(DATA_PILL_RE.findall(text))
            recipe_stats.formula_count += len(FORMULA_RE.findall(text))
            for table in LOOKUP_RE.findall(text):
                recipe_stats.lookup_tables[table] += 1
            if len(text) >= 2000:
                recipe_stats.long_string_fields.append((path, len(text), preview(text)))

        recipe_stats.schema_bytes = estimate_schema_bytes(root)
        stats.append(recipe_stats)

    duplicate_step_groups = [
        (fingerprint, steps)
        for fingerprint, steps in all_fingerprints.items()
        if len({step.recipe_path for step in steps}) > 1 and is_meaningful_duplicate(steps)
    ]

    findings.extend(build_findings(stats, recipe_paths, available_assets, duplicate_step_groups))
    return ProjectAnalysis(
        source=source,
        recipe_stats=sorted(stats, key=lambda item: item.complexity_score, reverse=True),
        findings=sorted(findings, key=lambda item: item.sort_key()),
        recipe_paths=recipe_paths,
        available_assets=available_assets,
        duplicate_step_groups=duplicate_step_groups,
    )


def collect_recipe_function_call(step: Step, stats: RecipeStats) -> None:
    if step.provider != "workato_recipe_function" and step.name != "call_recipe":
        return
    flow = step.raw.get("input", {}).get("flow_id") if isinstance(step.raw.get("input"), dict) else None
    if isinstance(flow, dict):
        zip_name = as_text(flow.get("zip_name"))
        name = as_text(flow.get("name"))
        folder = as_text(flow.get("folder"))
    else:
        zip_name = ""
        name = as_text(flow)
        folder = ""
    stats.recipe_function_calls.append((zip_name, name, folder))


def is_meaningful_duplicate(steps: list[Step]) -> bool:
    sample = steps[0]
    if sample.keyword in {"trigger", "if", "else", "catch"}:
        return False
    if not sample.provider and not sample.name:
        return False
    return True


def estimate_schema_bytes(root: dict[str, Any]) -> int:
    total = 0
    for path, value in find_schema_fields(root):
        total += json_size(value)
        if isinstance(value, str):
            total += len(value.encode("utf-8"))
    return total


def find_schema_fields(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    schema_names = {
        "recipe_data_schema",
        "extended_input_schema",
        "extended_output_schema",
        "job_report_schema",
        "job_report_config",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in schema_names:
                yield child_path, child
            yield from find_schema_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from find_schema_fields(child, f"{path}[{index}]")


def build_findings(
    stats: list[RecipeStats],
    recipe_paths: set[str],
    available_assets: set[str],
    duplicate_step_groups: list[tuple[str, list[Step]]],
) -> list[Finding]:
    findings: list[Finding] = []
    missing_references: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

    if not stats:
        return [
            Finding(
                "HIGH",
                "input",
                "No recipe files found",
                f"No files ending in {RECIPE_SUFFIX!r} were found in the input.",
            )
        ]

    step_counts = [item.step_count for item in stats]
    median_steps = statistics.median(step_counts) if step_counts else 0
    large_step_threshold = max(30, int(median_steps * 2))

    for recipe in stats:
        if recipe.step_count >= large_step_threshold:
            findings.append(
                Finding(
                    "MEDIUM",
                    "complexity",
                    "Large recipe should be split or simplified",
                    (
                        f"{recipe.name} has {recipe.step_count} steps, depth {recipe.max_depth}, "
                        f"and complexity score {recipe.complexity_score}. Consider extracting "
                        "reused validation/defaulting blocks into callable recipe functions."
                    ),
                    recipe=recipe.path,
                    impact=recipe.complexity_score,
                )
            )

        if recipe.max_depth >= 5:
            findings.append(
                Finding(
                    "MEDIUM",
                    "complexity",
                    "Deep nesting increases runtime and maintenance risk",
                    (
                        f"Maximum nesting depth is {recipe.max_depth}. Flatten guard clauses "
                        "or move nested blocks into dedicated recipe functions."
                    ),
                    recipe=recipe.path,
                    impact=recipe.max_depth,
                )
            )

        if recipe.skipped_steps:
            examples = ", ".join(step.label for step in recipe.skipped_steps[:5])
            findings.append(
                Finding(
                    "LOW",
                    "cleanup",
                    "Skipped steps are still stored in the recipe",
                    (
                        f"{len(recipe.skipped_steps)} skipped step(s): {examples}. Remove them "
                        "after confirming they are not needed for rollback or documentation."
                    ),
                    recipe=recipe.path,
                    impact=len(recipe.skipped_steps),
                )
            )

        if recipe.schema_bytes > 500_000:
            findings.append(
                Finding(
                    "LOW",
                    "package-size",
                    "Large inline schemas inflate package size",
                    (
                        f"Schema-like fields account for about {format_bytes(recipe.schema_bytes)}. "
                        "When editing in Workato, avoid refreshing broad connector schemas unless "
                        "the new fields are required."
                    ),
                    recipe=recipe.path,
                    impact=recipe.schema_bytes,
                )
            )

        if recipe.formula_count >= 75:
            findings.append(
                Finding(
                    "LOW",
                    "formula",
                    "Recipe has many formulas/data pills",
                    (
                        f"Found about {recipe.formula_count} formula/data-pill references. "
                        "Long repeated expressions are good candidates for variables or "
                        "small helper recipe functions."
                    ),
                    recipe=recipe.path,
                    impact=recipe.formula_count,
                )
            )

        repeated_tables = [table for table, count in recipe.lookup_tables.items() if count >= 5]
        if repeated_tables:
            table_summary = ", ".join(
                f"{table} ({recipe.lookup_tables[table]})" for table in repeated_tables[:6]
            )
            findings.append(
                Finding(
                    "MEDIUM",
                    "lookup",
                    "Repeated lookup calls can be cached",
                    (
                        f"Repeated lookup table usage: {table_summary}. Cache lookup results in "
                        "variables when the same key is reused in a job."
                    ),
                    recipe=recipe.path,
                    impact=sum(recipe.lookup_tables[table] for table in repeated_tables),
                )
            )

        for _, matching_steps in recipe.duplicate_local_steps[:5]:
            labels = ", ".join(step.label for step in matching_steps[:5])
            findings.append(
                Finding(
                    "LOW",
                    "duplication",
                    "Duplicate action inside one recipe",
                    (
                        f"{len(matching_steps)} matching actions appear in this recipe: {labels}. "
                        "If they do the same work with the same inputs, keep one path or extract a helper."
                    ),
                    recipe=recipe.path,
                    step=matching_steps[0].path,
                    impact=len(matching_steps),
                )
            )

        for zip_name, name, _folder in recipe.recipe_function_calls:
            if zip_name and zip_name not in recipe_paths and zip_name not in available_assets:
                missing_references[(zip_name, name)].append(recipe.path)

    for (zip_name, name), affected_recipes in sorted(missing_references.items()):
        unique_recipes = sorted(set(affected_recipes))
        findings.append(
            Finding(
                "HIGH",
                "reference",
                "Recipe function reference is missing from package",
                (
                    f"{len(affected_recipes)} call(s) reference {name!r} via {zip_name!r}, "
                    f"but that recipe file is not present in the package. Affected recipes: "
                    f"{', '.join(unique_recipes[:12])}"
                    + ("." if len(unique_recipes) <= 12 else f", and {len(unique_recipes) - 12} more.")
                    + " Import may require a shared/template project."
                ),
                recipe=unique_recipes[0] if unique_recipes else None,
                impact=len(affected_recipes),
            )
        )

    for _fingerprint, matching_steps in sorted(
        duplicate_step_groups,
        key=lambda item: len({step.recipe_path for step in item[1]}),
        reverse=True,
    )[:20]:
        recipe_names = sorted({step.recipe_name for step in matching_steps})
        sample = matching_steps[0]
        findings.append(
            Finding(
                "MEDIUM",
                "duplication",
                "Duplicate action appears across recipes",
                (
                    f"{sample.provider}/{sample.name} appears {len(matching_steps)} times across "
                    f"{len(recipe_names)} recipes: {', '.join(recipe_names[:8])}. Consider a "
                    "shared recipe function if this action is business logic, not simple plumbing."
                ),
                recipe=sample.recipe_path,
                step=sample.path,
                impact=len(matching_steps),
            )
        )

    return findings


def preview(text: str, length: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= length:
        return compact
    return compact[: length - 3] + "..."


def format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def render_markdown(analysis: ProjectAnalysis, max_findings: int = 80) -> str:
    recipe_count = len(analysis.recipe_stats)
    total_steps = sum(recipe.step_count for recipe in analysis.recipe_stats)
    total_bytes = sum(recipe.bytes_on_disk for recipe in analysis.recipe_stats)
    top_recipes = analysis.recipe_stats[:15]
    finding_counts = Counter(finding.severity for finding in analysis.findings)

    lines: list[str] = []
    lines.append("# Workato Recipe Optimization Report")
    lines.append("")
    lines.append(f"- Source: `{analysis.source}`")
    lines.append(f"- Recipes analyzed: {recipe_count}")
    lines.append(f"- Total steps: {total_steps}")
    lines.append(f"- Recipe JSON size: {format_bytes(total_bytes)}")
    lines.append(
        "- Findings: "
        + ", ".join(
            f"{severity}={finding_counts.get(severity, 0)}"
            for severity in ("HIGH", "MEDIUM", "LOW", "INFO")
            if finding_counts.get(severity, 0)
        )
    )
    lines.append("")

    lines.append("## Top Recipes by Complexity")
    lines.append("")
    lines.append("| Recipe | Steps | Depth | Score | Skipped | Calls | Lookups | Size |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for recipe in top_recipes:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(recipe.name),
                    str(recipe.step_count),
                    str(recipe.max_depth),
                    str(recipe.complexity_score),
                    str(recipe.skipped_count),
                    str(len(recipe.recipe_function_calls)),
                    str(sum(recipe.lookup_tables.values())),
                    format_bytes(recipe.bytes_on_disk),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    if not analysis.findings:
        lines.append("No optimization findings were detected.")
    else:
        for index, finding in enumerate(analysis.findings[:max_findings], start=1):
            location = ""
            if finding.recipe:
                location = f" Recipe: `{finding.recipe}`"
            if finding.step:
                location += f" Step path: `{finding.step}`"
            lines.append(
                f"{index}. **{finding.severity} / {finding.category}: "
                f"{markdown_escape(finding.title)}**{location}"
            )
            lines.append(f"   {markdown_escape(finding.detail)}")
        if len(analysis.findings) > max_findings:
            lines.append("")
            lines.append(
                f"_Showing {max_findings} of {len(analysis.findings)} findings. "
                "Use `--max-findings` to include more._"
            )
    lines.append("")

    lines.append("## Provider Usage")
    lines.append("")
    provider_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    for recipe in analysis.recipe_stats:
        provider_counts.update(recipe.provider_counts)
        keyword_counts.update(recipe.keyword_counts)
    lines.append("Top providers: " + comma_counts(provider_counts, 12))
    lines.append("")
    lines.append("Keywords: " + comma_counts(keyword_counts, 12))
    lines.append("")

    lines.append("## Safe Optimization Guidance")
    lines.append("")
    lines.append(
        textwrap.fill(
            "This report does not change recipe logic. Review HIGH findings first, then split "
            "large/deep recipes, cache repeated lookup results in variables, and extract repeated "
            "business logic into recipe functions. Use --optimized-zip only when you need a smaller "
            "transport package; it compacts JSON but does not improve runtime behavior.",
            width=100,
        )
    )
    lines.append("")
    return "\n".join(lines)


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def html_escape(value: Any) -> str:
    return html.escape(as_text(value), quote=True)


def render_html(analysis: ProjectAnalysis, max_findings: int = 80) -> str:
    recipe_count = len(analysis.recipe_stats)
    total_steps = sum(recipe.step_count for recipe in analysis.recipe_stats)
    total_bytes = sum(recipe.bytes_on_disk for recipe in analysis.recipe_stats)
    top_recipes = analysis.recipe_stats[:15]
    finding_counts = Counter(finding.severity for finding in analysis.findings)
    provider_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()
    for recipe in analysis.recipe_stats:
        provider_counts.update(recipe.provider_counts)
        keyword_counts.update(recipe.keyword_counts)

    def severity_class(severity: str) -> str:
        return severity.lower() if severity else "info"

    def count_tile(label: str, value: Any, tone: str = "") -> str:
        tone_class = f" {tone}" if tone else ""
        return (
            f'<div class="metric{tone_class}">'
            f'<div class="metric-label">{html_escape(label)}</div>'
            f'<div class="metric-value">{html_escape(value)}</div>'
            "</div>"
        )

    recipe_rows = []
    max_score = max((recipe.complexity_score for recipe in top_recipes), default=1)
    for recipe in top_recipes:
        width = max(4, round((recipe.complexity_score / max_score) * 100))
        recipe_rows.append(
            "<tr>"
            f"<td><strong>{html_escape(recipe.name)}</strong><span>{html_escape(recipe.path)}</span></td>"
            f"<td>{recipe.step_count}</td>"
            f"<td>{recipe.max_depth}</td>"
            f"<td><div class=\"score\"><span style=\"width:{width}%\"></span></div>{recipe.complexity_score}</td>"
            f"<td>{recipe.skipped_count}</td>"
            f"<td>{len(recipe.recipe_function_calls)}</td>"
            f"<td>{sum(recipe.lookup_tables.values())}</td>"
            f"<td>{format_bytes(recipe.bytes_on_disk)}</td>"
            "</tr>"
        )

    finding_items = []
    for index, finding in enumerate(analysis.findings[:max_findings], start=1):
        location = ""
        if finding.recipe:
            location += f'<div class="location">Recipe: <code>{html_escape(finding.recipe)}</code></div>'
        if finding.step:
            location += f'<div class="location">Step path: <code>{html_escape(finding.step)}</code></div>'
        finding_items.append(
            f'<article class="finding {severity_class(finding.severity)}">'
            f'<div class="finding-index">{index}</div>'
            '<div class="finding-body">'
            '<div class="finding-heading">'
            f'<span class="badge {severity_class(finding.severity)}">{html_escape(finding.severity)}</span>'
            f'<span class="category">{html_escape(finding.category)}</span>'
            f'<h3>{html_escape(finding.title)}</h3>'
            '</div>'
            f'{location}'
            f'<p>{html_escape(finding.detail)}</p>'
            '</div>'
            '</article>'
        )

    truncated_note = ""
    if len(analysis.findings) > max_findings:
        truncated_note = (
            f'<p class="note">Showing {max_findings} of {len(analysis.findings)} findings. '
            "Use <code>--max-findings</code> to include more.</p>"
        )

    provider_tags = "".join(
        f'<span class="tag">{html_escape(key)} <strong>{value}</strong></span>'
        for key, value in provider_counts.most_common(12)
    )
    keyword_tags = "".join(
        f'<span class="tag">{html_escape(key)} <strong>{value}</strong></span>'
        for key, value in keyword_counts.most_common(12)
    )

    findings_summary = "".join(
        count_tile(severity, finding_counts.get(severity, 0), severity.lower())
        for severity in ("HIGH", "MEDIUM", "LOW", "INFO")
        if finding_counts.get(severity, 0)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Workato Recipe Optimization Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d8dee8;
      --surface: #ffffff;
      --page: #f6f7f9;
      --accent: #116466;
      --high: #b42318;
      --medium: #a15c07;
      --low: #3563c7;
      --info: #475569;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: #0f2d2f;
      color: white;
      padding: 32px max(24px, calc((100vw - 1180px) / 2));
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: 32px;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      color: #c9d8d9;
      overflow-wrap: anywhere;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 56px;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 18px;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fbfcfe;
    }}
    .metric.high {{ border-color: #f4b5ae; }}
    .metric.medium {{ border-color: #f5c989; }}
    .metric.low {{ border-color: #b9c9f5; }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
    }}
    .metric-value {{
      margin-top: 4px;
      font-size: 25px;
      font-weight: 750;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      min-width: 850px;
      border-collapse: collapse;
      background: white;
    }}
    th, td {{
      padding: 11px 12px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
      font-size: 14px;
    }}
    th {{
      background: #eef3f4;
      color: #334155;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    td span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .score {{
      display: inline-block;
      width: 86px;
      height: 8px;
      margin-right: 8px;
      background: #e8edf3;
      border-radius: 999px;
      overflow: hidden;
      vertical-align: middle;
    }}
    .score span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .finding {{
      display: grid;
      grid-template-columns: 42px 1fr;
      gap: 12px;
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 8px;
      padding: 14px;
      margin: 10px 0;
      background: white;
    }}
    .finding.high {{ border-left-color: var(--high); }}
    .finding.medium {{ border-left-color: var(--medium); }}
    .finding.low {{ border-left-color: var(--low); }}
    .finding.info {{ border-left-color: var(--info); }}
    .finding-index {{
      color: var(--muted);
      font-weight: 800;
      font-size: 18px;
    }}
    .finding-heading {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }}
    .finding h3 {{
      flex-basis: 100%;
      margin: 2px 0 0;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .finding p {{
      margin: 8px 0 0;
      color: #334155;
    }}
    .badge {{
      border-radius: 999px;
      color: white;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 800;
    }}
    .badge.high {{ background: var(--high); }}
    .badge.medium {{ background: var(--medium); }}
    .badge.low {{ background: var(--low); }}
    .badge.info {{ background: var(--info); }}
    .category {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }}
    .location {{
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    code {{
      background: #eef2f7;
      border-radius: 4px;
      padding: 1px 4px;
      color: #1f2937;
    }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .tag {{
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fbfcfe;
      padding: 6px 10px;
      font-size: 13px;
    }}
    .note {{
      color: var(--muted);
      margin: 14px 0 0;
    }}
    .guidance {{
      color: #334155;
      margin: 0;
    }}
    @media (max-width: 720px) {{
      header {{ padding: 24px 16px; }}
      header h1 {{ font-size: 26px; }}
      main {{ width: calc(100vw - 20px); margin-top: 10px; }}
      section {{ padding: 16px; }}
      .finding {{ grid-template-columns: 1fr; }}
      .finding-index {{ font-size: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Workato Recipe Optimization Report</h1>
    <p>{html_escape(analysis.source)}</p>
  </header>
  <main>
    <section>
      <h2>Summary</h2>
      <div class="metrics">
        {count_tile("Recipes", recipe_count)}
        {count_tile("Total steps", total_steps)}
        {count_tile("Recipe JSON size", format_bytes(total_bytes))}
        {findings_summary}
      </div>
    </section>
    <section>
      <h2>Top Recipes by Complexity</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Recipe</th>
              <th>Steps</th>
              <th>Depth</th>
              <th>Score</th>
              <th>Skipped</th>
              <th>Calls</th>
              <th>Lookups</th>
              <th>Size</th>
            </tr>
          </thead>
          <tbody>
            {''.join(recipe_rows)}
          </tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>Findings</h2>
      {''.join(finding_items) if finding_items else '<p>No optimization findings were detected.</p>'}
      {truncated_note}
    </section>
    <section>
      <h2>Provider Usage</h2>
      <div class="tags">{provider_tags or '<span class="tag">none</span>'}</div>
      <h2>Keyword Usage</h2>
      <div class="tags">{keyword_tags or '<span class="tag">none</span>'}</div>
    </section>
    <section>
      <h2>Safe Optimization Guidance</h2>
      <p class="guidance">This report does not change recipe logic. Review HIGH findings first, then split large/deep recipes, cache repeated lookup results in variables, and extract repeated business logic into recipe functions. Use <code>--optimized-zip</code> only when you need a smaller transport package; it compacts JSON but does not improve runtime behavior.</p>
    </section>
  </main>
</body>
</html>
"""


def comma_counts(counter: Counter[str], limit: int) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def render_json(analysis: ProjectAnalysis) -> dict[str, Any]:
    return {
        "source": analysis.source,
        "summary": {
            "recipes": len(analysis.recipe_stats),
            "steps": sum(recipe.step_count for recipe in analysis.recipe_stats),
            "findings": Counter(finding.severity for finding in analysis.findings),
        },
        "recipes": [
            {
                "path": recipe.path,
                "name": recipe.name,
                "version": recipe.version,
                "concurrency": recipe.concurrency,
                "steps": recipe.step_count,
                "max_depth": recipe.max_depth,
                "complexity_score": recipe.complexity_score,
                "skipped_steps": recipe.skipped_count,
                "recipe_function_calls": [
                    {"zip_name": zip_name, "name": name, "folder": folder}
                    for zip_name, name, folder in recipe.recipe_function_calls
                ],
                "lookup_tables": dict(recipe.lookup_tables),
                "provider_counts": dict(recipe.provider_counts),
                "keyword_counts": dict(recipe.keyword_counts),
                "bytes_on_disk": recipe.bytes_on_disk,
                "schema_bytes_estimate": recipe.schema_bytes,
            }
            for recipe in analysis.recipe_stats
        ],
        "findings": [
            {
                "severity": finding.severity,
                "category": finding.category,
                "title": finding.title,
                "detail": finding.detail,
                "recipe": finding.recipe,
                "step": finding.step,
                "impact": finding.impact,
            }
            for finding in analysis.findings
        ],
    }


def write_compact_zip(source_files: list[SourceFile], output_path: Path) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for file in source_files:
            content = file.content
            if file.path.endswith(".json"):
                try:
                    parsed = json.loads(content.decode("utf-8"))
                    content = json.dumps(
                        parsed,
                        ensure_ascii=False,
                        sort_keys=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            package.writestr(file.path, content)


def print_console_summary(analysis: ProjectAnalysis) -> None:
    high = sum(1 for finding in analysis.findings if finding.severity == "HIGH")
    medium = sum(1 for finding in analysis.findings if finding.severity == "MEDIUM")
    low = sum(1 for finding in analysis.findings if finding.severity == "LOW")
    total_steps = sum(recipe.step_count for recipe in analysis.recipe_stats)
    print(f"Analyzed {len(analysis.recipe_stats)} recipes and {total_steps} steps.")
    print(f"Findings: HIGH={high}, MEDIUM={medium}, LOW={low}")
    if analysis.recipe_stats:
        top = analysis.recipe_stats[0]
        print(
            "Highest complexity: "
            f"{top.name} ({top.step_count} steps, depth {top.max_depth}, score {top.complexity_score})"
        )
    if analysis.findings:
        print("Top findings:")
        for finding in analysis.findings[:8]:
            location = f" [{finding.recipe}]" if finding.recipe else ""
            print(f"- {finding.severity} {finding.category}: {finding.title}{location}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and safely optimize Workato recipe export packages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="Workato export .zip or extracted package directory")
    parser.add_argument(
        "--markdown",
        "-m",
        type=Path,
        help="Write a Markdown optimization report",
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="Write a standalone HTML optimization report",
    )
    parser.add_argument(
        "--json-report",
        "-j",
        type=Path,
        help="Write a machine-readable JSON report",
    )
    parser.add_argument(
        "--optimized-zip",
        "-o",
        type=Path,
        help="Write a compacted copy of the package without changing recipe logic",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=80,
        help="Maximum findings to include in Markdown or HTML output",
    )
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="Exit with status 1 when HIGH findings are present",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    source = Path(args.source).expanduser().resolve()

    try:
        package = WorkatoPackage(source)
        files = package.files()
        analysis = analyze_package(files, str(source))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print_console_summary(analysis)

    if args.markdown:
        args.markdown.write_text(
            render_markdown(analysis, max_findings=args.max_findings),
            encoding="utf-8",
        )
        print(f"Markdown report: {args.markdown}")

    if args.html:
        args.html.write_text(
            render_html(analysis, max_findings=args.max_findings),
            encoding="utf-8",
        )
        print(f"HTML report: {args.html}")

    if args.json_report:
        args.json_report.write_text(
            json.dumps(render_json(analysis), indent=2, ensure_ascii=False, default=dict),
            encoding="utf-8",
        )
        print(f"JSON report: {args.json_report}")

    if args.optimized_zip:
        write_compact_zip(files, args.optimized_zip)
        original_size = sum(len(file.content) for file in files)
        compact_size = args.optimized_zip.stat().st_size
        print(
            f"Optimized zip: {args.optimized_zip} "
            f"({format_bytes(original_size)} source content -> {format_bytes(compact_size)} zip)"
        )

    if not args.markdown and not args.html and not args.json_report:
        print("")
        print(render_markdown(analysis, max_findings=min(args.max_findings, 20)))

    has_high_findings = any(finding.severity == "HIGH" for finding in analysis.findings)
    return 1 if args.fail_on_high and has_high_findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
