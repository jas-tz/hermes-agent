#!/usr/bin/env python3
"""Generate a PR comment from evaluation results.

This script reads the evaluation output and generates a formatted
comment suitable for posting on a GitHub PR.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional


def parse_eval_output(filepath: str) -> Dict[str, Any]:
    """Parse the evaluation output file to extract key metrics."""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath) as f:
        content = f.read()
    
    # Try to find JSON report in the output
    json_start = content.find('{')
    json_end = content.rfind('}')
    
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            return json.loads(content[json_start:json_end + 1])
        except json.JSONDecodeError:
            pass
    
    return {}


def generate_comment(
    report: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a PR comment from evaluation results."""
    lines: List[str] = []
    
    # Header
    lines.append("## JARVIS Evaluation Results")
    lines.append("")
    
    # Summary
    summary = report.get("summary", {})
    lines.append("### Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Cases | {summary.get('total_cases', 'N/A')} |")
    lines.append(f"| Passed | {summary.get('total_passed', 'N/A')} |")
    lines.append(f"| Partial | {summary.get('total_partial', 'N/A')} |")
    lines.append(f"| Failed | {summary.get('total_failed', 'N/A')} |")
    lines.append(f"| Pass Rate | {summary.get('overall_pass_rate', 0):.1%} |")
    lines.append(f"| Quality Score | {summary.get('overall_quality_score', 0):.1f}/100 |")
    lines.append(f"| **Verdict** | **{summary.get('overall_verdict', 'N/A')}** |")
    lines.append("")
    
    # Quality Gates
    gates = report.get("quality_gates", {})
    if gates:
        lines.append("### Quality Gates")
        lines.append("")
        lines.append("| Gate | Status |")
        lines.append("|------|--------")
        
        gate_names = [
            ("safety_passed", "Safety"),
            ("correctness_passed", "Correctness"),
            ("grounding_passed", "Grounding"),
            ("tool_behavior_passed", "Tool Behavior"),
            ("trajectory_passed", "Trajectory"),
            ("overall_quality_passed", "Overall Quality"),
        ]
        
        for key, name in gate_names:
            passed = gates.get(key, False)
            emoji = "✅" if passed else "❌"
            status = "PASS" if passed else "FAIL"
            lines.append(f"| {name} | {emoji} {status} |")
        
        lines.append("")
        
        # Gate failures
        failures = gates.get("failures", [])
        if failures:
            lines.append("**Gate Failures:**")
            for failure in failures:
                lines.append(f"- {failure}")
            lines.append("")
    
    # Hard Blockers
    hard_blockers = report.get("hard_blockers", [])
    if hard_blockers:
        lines.append("### 🚨 Hard Blockers")
        lines.append("")
        for blocker in hard_blockers:
            lines.append(f"- `{blocker}`")
        lines.append("")
    
    # Regressions
    regressions = report.get("regressions", [])
    if regressions:
        lines.append("### ⚠️ Regressions")
        lines.append("")
        for regression in regressions:
            lines.append(f"- `{regression}`")
        lines.append("")
    
    # Baseline Comparison
    if comparison:
        lines.append("### Baseline Comparison")
        lines.append("")
        if comparison.get("compatible", True):
            lines.append(f"- Baseline: `{comparison.get('baseline_id', 'N/A')}`")
            lines.append(f"- Pass Rate Delta: {comparison.get('pass_rate_delta', 0):+.1%}")
            lines.append(f"- Regressions: {comparison.get('regressed_count', 0)}")
            lines.append(f"- Improvements: {comparison.get('improved_count', 0)}")
        else:
            lines.append(f"⚠️ Incompatible with baseline: {comparison.get('incompatibility_reason', 'Unknown')}")
        lines.append("")
    
    # Categories
    categories = report.get("categories", [])
    if categories:
        lines.append("### Category Breakdown")
        lines.append("")
        lines.append("| Category | Total | Pass | Partial | Fail | Pass Rate |")
        lines.append("|----------|-------|------|---------|------|-----------|")
        
        for cat in categories:
            pass_rate = cat.get('pass_rate', 0)
            lines.append(
                f"| {cat.get('category', 'N/A')} | {cat.get('total_cases', 0)} | "
                f"{cat.get('passed', 0)} | {cat.get('partial', 0)} | "
                f"{cat.get('failed', 0)} | {pass_rate:.1%} |"
            )
        
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("*Generated by JARVIS Evaluation CI*")
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    # Get environment variables
    pr_number = os.environ.get("PR_NUMBER", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    
    # Parse evaluation output
    eval_output = parse_eval_output("eval_output.txt")
    
    # Parse baseline comparison
    comparison = None
    if os.path.exists("baseline_comparison.json"):
        try:
            with open("baseline_comparison.json") as f:
                comparison = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Generate comment
    comment = generate_comment(eval_output, comparison)
    
    # Write comment to file
    with open("pr_comment.md", "w") as f:
        f.write(comment)
    
    # Print comment
    print(comment)
    
    # Post comment on PR if credentials available
    if pr_number and github_token:
        try:
            import urllib.request
            import urllib.error
            
            # Read comment content
            with open("pr_comment.md") as f:
                comment_body = f.read()
            
            # Post comment via GitHub API
            api_url = f"https://api.github.com/repos/{os.environ.get('GITHUB_REPOSITORY', '')}/issues/{pr_number}/comments"
            
            data = json.dumps({"body": comment_body}).encode("utf-8")
            
            req = urllib.request.Request(
                api_url,
                data=data,
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    print(f"Comment posted: {response.status}")
            except urllib.error.HTTPError as e:
                print(f"Failed to post comment: {e.code} {e.reason}", file=sys.stderr)
                return 1
                
        except Exception as e:
            print(f"Error posting comment: {e}", file=sys.stderr)
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
