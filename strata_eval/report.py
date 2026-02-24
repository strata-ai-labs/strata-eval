"""Report generator — reads result JSONs and produces Markdown / LaTeX tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=["markdown", "latex"], default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--bench", nargs="*", default=None,
        help="Filter to specific benchmark suites",
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(ROOT / "results"),
        help="Directory containing result JSON files (default: results/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write output to file instead of stdout",
    )


def run_report(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        return

    reports = _load_reports(results_dir)
    if args.bench:
        reports = [r for r in reports if _report_category(r) in args.bench]

    if not reports:
        print("No result files found.")
        return

    if args.format == "latex":
        output = _generate_latex(reports)
    else:
        output = _generate_markdown(reports)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


# -------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------

def _load_reports(results_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            # Basic validation: must be a dict with recognized structure
            if not isinstance(data, dict):
                continue
            if "results" in data or ("dataset" in data and "mode" in data):
                reports.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return reports


def _report_category(report: dict) -> str:
    """Extract the benchmark category from a report."""
    # New schema: results[].category
    results = report.get("results", [])
    if results:
        return results[0].get("category", "unknown")
    # Legacy BEIR schema: has "dataset" and "mode" at top level
    if "dataset" in report and "mode" in report:
        return "beir"
    return "unknown"


def _format_metric(value: object) -> str:
    """Format a single metric value for display."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        # Use fewer decimals for large numbers, more for small
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 1:
            return f"{value:.4f}"
        return f"{value:.6f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _collect_columns(results: list[dict]) -> list[str]:
    """Collect metric columns from ALL results, not just the first."""
    all_cols: set[str] = set()
    for res in results:
        all_cols.update(res.get("metrics", {}).keys())
    return sorted(all_cols)


def _generate_markdown(reports: list[dict]) -> str:
    lines = ["# Benchmark Results\n"]

    by_category: dict[str, list[dict]] = {}
    for r in reports:
        cat = _report_category(r)
        by_category.setdefault(cat, []).append(r)

    for category, cat_reports in sorted(by_category.items()):
        lines.append(f"## {category.upper()}\n")
        for report in cat_reports:
            results = report.get("results", [])
            if not results:
                # Legacy BEIR format
                lines.append(_format_legacy_beir_md(report))
                continue

            meta = report.get("metadata", {})
            ts = meta.get("timestamp", "unknown")
            sdk_ver = meta.get("sdk_version", "?")
            lines.append(f"*Run: {ts} | SDK: {sdk_ver}*\n")

            # Collect columns from all results
            cols = _collect_columns(results)
            if not cols:
                continue

            header = "| Benchmark | " + " | ".join(cols) + " |"
            sep = "|---|" + "|".join("---:" for _ in cols) + "|"
            lines.append(header)
            lines.append(sep)
            for res in results:
                name = res.get("benchmark", "?")
                m = res.get("metrics", {})
                vals = " | ".join(_format_metric(m.get(c, "")) for c in cols)
                lines.append(f"| {name} | {vals} |")
            lines.append("")
    return "\n".join(lines)


def _format_legacy_beir_md(report: dict) -> str:
    """Format a legacy BEIR result (old flat JSON schema)."""
    ds = report.get("dataset", "?")
    mode = report.get("mode", "?")
    metrics = report.get("metrics", {})
    ndcg10 = metrics.get("ndcg", {}).get("NDCG@10", 0)
    recall100 = metrics.get("recall", {}).get("Recall@100", 0)
    lines = [
        "| Dataset | NDCG@10 | Recall@100 |",
        "|---|---:|---:|",
        f"| {ds} ({mode}) | {ndcg10:.4f} | {recall100:.4f} |",
        "",
    ]
    return "\n".join(lines)


def _escape_latex(s: str) -> str:
    """Escape LaTeX special characters."""
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}", "~", "^"):
        s = s.replace(char, f"\\{char}")
    return s


def _generate_latex(reports: list[dict]) -> str:
    lines = []

    by_category: dict[str, list[dict]] = {}
    for r in reports:
        cat = _report_category(r)
        by_category.setdefault(cat, []).append(r)

    for category, cat_reports in sorted(by_category.items()):
        lines.append(f"% ---- {category.upper()} ----")

        for report in cat_reports:
            results = report.get("results", [])
            if not results:
                continue

            cols = _collect_columns(results)
            if not cols:
                continue

            col_spec = "l" + "c" * len(cols)
            lines.append(r"\begin{table}[t]")
            lines.append(r"\centering")
            lines.append(f"\\caption{{{_escape_latex(category.upper())} Benchmark Results}}")
            lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            lines.append(r"\toprule")
            header = "Benchmark & " + " & ".join(_escape_latex(c) for c in cols) + r" \\"
            lines.append(header)
            lines.append(r"\midrule")
            for res in results:
                name = _escape_latex(res.get("benchmark", "?"))
                m = res.get("metrics", {})
                vals = " & ".join(_format_metric(m.get(c, "")) for c in cols)
                lines.append(f"{name} & {vals} \\\\")
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

    return "\n".join(lines)
