"""
Evaluation utilities and metrics for hierarchical medical responses.

This module provides:
- HierarchicalStructureEvaluator: parses markdown lists into a tree and
  computes structural statistics.
- check_compliance: checks multi-level (≥3) hierarchy with at most N children
  per parent node.
- Text-level metrics: string similarity, ROUGE-L, BLEU.
- Structure-level metrics and a combined eval_pair helper.
"""

from __future__ import annotations

from collections import defaultdict
from math import exp, log
from typing import Dict, List, Tuple, Optional
import re


class HierarchicalStructureEvaluator:
    """Analyze hierarchical structures based on markdown list indentation."""

    def __init__(self, min_depth: int = 3, max_points_per_level: int = 5) -> None:
        """
        Args:
            min_depth: Minimum depth of the hierarchy (e.g. 3 means ≥3 levels).
            max_points_per_level: Maximum number of points per level when using
                the legacy “per-level total count” rule (kept for statistics
                and backward compatibility).
        """
        self.min_depth = min_depth
        self.max_points_per_level = max_points_per_level

    def parse_markdown_list(self, text: str) -> List[Tuple[int, str]]:
        """
        Parse a markdown list and return a list of (level, content) tuples.

        Conventions:
        - Use "-" or "*" as bullet markers.
        - Indentation is interpreted in steps of 2 spaces:
          0 spaces -> level 1, 2 spaces -> level 2, 4 spaces -> level 3, etc.

        Args:
            text: Markdown-formatted list text.

        Returns:
            List of (level, content) tuples, where `level` is 1-based.
        """
        lines = text.strip().split("\n")
        parsed: List[Tuple[int, str]] = []

        for original_line in lines:
            # Keep original line for indentation; skip empty lines
            stripped_line = original_line.strip()
            if not stripped_line:
                continue

            # Match format: [leading spaces][- or *][space][content]
            match = re.match(r"^(\s*)([-*])\s+(.+)$", original_line)
            if match:
                leading_spaces = len(match.group(1))
                level = (leading_spaces // 2) + 1
                content = match.group(3).strip()
                parsed.append((level, content))

        return parsed

    def analyze_structure(self, text: str) -> Dict:
        """
        Analyze structure using the legacy “per-level total count” rule.

        Returns:
            dict with:
              - max_depth: maximum depth
              - points_per_level: global node count per level
              - total_points: total node count
              - structure_valid: whether depth and per-level limits are satisfied
              - violations: list of violation strings
        """
        parsed = self.parse_markdown_list(text)

        if not parsed:
            return {
                "max_depth": 0,
                "points_per_level": {},
                "total_points": 0,
                "structure_valid": False,
                "violations": ["无有效的markdown列表结构"],
            }

        points_per_level: Dict[int, int] = defaultdict(int)
        for level, _ in parsed:
            points_per_level[level] += 1

        max_depth = max(level for level, _ in parsed)
        total_points = len(parsed)

        violations: List[str] = []
        structure_valid = True

        if max_depth < self.min_depth:
            violations.append(f"深度不足: {max_depth} < {self.min_depth}")
            structure_valid = False

        for level, count in points_per_level.items():
            if count > self.max_points_per_level:
                violations.append(
                    f"第{level}层点数过多: {count} > {self.max_points_per_level}"
                )
                structure_valid = False

        return {
            "max_depth": max_depth,
            "points_per_level": dict(points_per_level),
            "total_points": total_points,
            "structure_valid": structure_valid,
            "violations": violations,
        }


def check_compliance(
    text: str,
    min_depth: int = 3,
    max_points_per_level: int = 5,
    verbose: bool = False,
) -> Dict:
    """Check structural compliance using the main project rule.

    Rule:
    - Minimum depth: max_depth ≥ min_depth.
    - For every node at any level, the number of its direct children must be
      ≤ max_points_per_level.

    Args:
        text: Markdown list text.
        min_depth: Required minimum depth.
        max_points_per_level: Maximum allowed children per parent node.
        verbose: Whether to print a human-readable report.

    Returns:
        dict with:
          - is_compliant: whether depth and per-parent constraints both pass
          - max_depth: maximum depth
          - total_points: total number of nodes
          - has_structure: whether any markdown list structure was detected
          - violations: list of violation strings
          - node_child_counts: list of per-node summaries
    """
    evaluator = HierarchicalStructureEvaluator(
        min_depth=min_depth, max_points_per_level=max_points_per_level
    )

    parsed = evaluator.parse_markdown_list(text)

    if not parsed:
        result = {
            "is_compliant": False,
            "max_depth": 0,
            "total_points": 0,
            "has_structure": False,
            "violations": ["No valid markdown list structure detected"],
            "node_child_counts": [],
        }
    else:
        # Build an implicit tree using a stack and count children per node
        nodes: List[Dict] = []
        # stack holds indices into `nodes` for the current ancestor chain
        stack: List[int] = []

        for level, content in parsed:
            # Pop nodes with level >= current level to find the parent
            while stack and nodes[stack[-1]]["level"] >= level:
                stack.pop()

            parent_idx: Optional[int] = stack[-1] if stack else None
            node = {
                "level": level,
                "content": content,
                "parent": parent_idx,
                "child_count": 0,
            }
            idx = len(nodes)
            nodes.append(node)

            # Update parent child count
            if parent_idx is not None:
                nodes[parent_idx]["child_count"] += 1

            stack.append(idx)

        max_depth = max(n["level"] for n in nodes)
        total_points = len(nodes)

        violations: List[str] = []
        if max_depth < min_depth:
            violations.append(f"Insufficient depth: {max_depth} < {min_depth}")

        for n in nodes:
            if n["child_count"] > max_points_per_level:
                violations.append(
                    "Node(level="
                    f"{n['level']}, content='{n['content'][:30]}...') has "
                    f"{n['child_count']} children > {max_points_per_level}"
                )

        is_compliant = max_depth >= min_depth and not violations

        result = {
            "is_compliant": is_compliant,
            "max_depth": max_depth,
            "total_points": total_points,
            "has_structure": True,
            "violations": violations,
            "node_child_counts": [
                {
                    "level": n["level"],
                    "content_preview": n["content"][:50],
                    "child_count": n["child_count"],
                }
                for n in nodes
            ],
        }

    if verbose:
        print("=" * 80)
        print("Compliance check (per-parent children ≤ max_points_per_level)")
        print("=" * 80)
        print(f"\nInput length (chars): {len(text)}")
        print(f"Has structure: {result['has_structure']}")
        print(f"Max depth: {result['max_depth']} (required ≥ {min_depth})")
        print(f"Total nodes (bullets): {result['total_points']}")
        print(f"\nCompliant: {result['is_compliant']}")

        print("\nPer-node children counts (first 15 nodes):")
        for i, info in enumerate(result["node_child_counts"][:15]):
            level = info["level"]
            cc = info["child_count"]
            preview = info["content_preview"]
            status = "✅" if cc <= max_points_per_level else "❌"
            print(
                f"  Node {i}: level={level}, children={cc} {status}, "
                f"content: {preview}..."
            )
        if len(result["node_child_counts"]) > 15:
            print(
                f"  ... {len(result['node_child_counts'])} nodes in total, "
                f"showing first 15"
            )

        if result["violations"]:
            print("\nViolations:")
            for v in result["violations"]:
                print(f"  ❌ {v}")
        else:
            print("\nNo violations detected")

        print("=" * 80)

    return result


# ------------------------ 文本相似度指标 ------------------------ #


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: split on whitespace and lowercase."""
    return text.lower().strip().split()


def string_similarity(pred: str, ref: str) -> float:
    """Token-level normalized Levenshtein similarity in [0, 1]."""
    a = _tokenize(pred)
    b = _tokenize(ref)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # 删除
                dp[i][j - 1] + 1,  # 插入
                dp[i - 1][j - 1] + cost,  # 替换
            )

    dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - dist / max_len


def rouge_l(pred: str, ref: str) -> float:
    """Token-level ROUGE-L F score."""
    hyp = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    m, n = len(ref_tokens), len(hyp)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec == 0 or rec == 0:
        return 0.0
    beta2 = 1.0
    return (1 + beta2) * prec * rec / (rec + beta2 * prec)


def bleu_score(pred: str, ref: str, max_n: int = 4) -> float:
    """Simple BLEU implementation (default BLEU-4) with +1 smoothing."""
    hyp = _tokenize(pred)
    ref_tokens = _tokenize(ref)
    if not hyp or not ref_tokens:
        return 0.0

    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_ngrams = ngrams(hyp, n)
        ref_ngrams = ngrams(ref_tokens, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        ref_counts: Dict[Tuple[str, ...], int] = {}
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1

        match = 0
        for g in hyp_ngrams:
            if ref_counts.get(g, 0) > 0:
                match += 1
                ref_counts[g] -= 1

        # +1 平滑，避免 0
        precisions.append((match + 1) / (len(hyp_ngrams) + 1))

    hyp_len = len(hyp)
    ref_len = len(ref_tokens)
    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = exp(1 - ref_len / hyp_len)

    avg_log_p = sum(log(p) if p > 0 else -999999999.0 for p in precisions) / max_n
    if any(p == 0.0 for p in precisions):
        bleu = 0.0
    else:
        bleu = bp * exp(avg_log_p)
    return float(bleu)


def structure_metrics(
    pred: str,
    min_depth: int = 3,
    max_points_per_parent: int = 5,
    verbose: bool = False,
) -> Dict:
    """Structure-related metrics (Structure Score 2.0 - Pyramid Principle).

    Metrics:
    1. Constraint Score (40%): Adherence to max_points_per_parent (<= 5).
       - Binary check per parent: 1.0 if <= 5, 0.0 if > 5.
       - Averaged over all parent nodes.
    2. Grouping Score (40%): Pyramid grouping quality (avoiding single children).
       - 3-5 children: 1.0 (Optimal)
       - 2 children: 0.9 (Good)
       - 1 child: 0.5 (Poor grouping)
       - > 5 children: 0.0 (Violates constraint)
    3. Depth Score (20%): Structure depth.
       - >= min_depth: 1.0
       - < min_depth: depth / min_depth

    Returns:
        - depth_ok, max_depth, depth_score
        - per_parent_constraint_ok, constraint_score
        - grouping_score
        - mece_compliant (binary)
        - mece_score (weighted sum)
    """
    comp_pred = check_compliance(
        pred,
        min_depth=min_depth,
        max_points_per_level=max_points_per_parent,
        verbose=False,
    )

    max_depth = comp_pred["max_depth"]
    total_points = comp_pred["total_points"]
    
    # 1. Depth Score (20%)
    depth_ok = 1.0 if max_depth >= min_depth else 0.0
    depth_score = min(1.0, max_depth / min_depth) if min_depth > 0 else 1.0
    
    # 2. Constraint & Grouping Scores
    parent_nodes = [
        n for n in comp_pred["node_child_counts"] if n["child_count"] > 0
    ]
    
    if len(parent_nodes) == 0:
        # No structure or only leaves?
        # If total_points > 0 but no parents, it's a flat list (depth 1)
        # A flat list has NO hierarchy, so Constraint Score (hierarchy quality) should be 0.
        constraint_score = 0.0
        grouping_score = 0.0
        per_parent_constraint_ok = 1.0  # Binary check still technically passes (no violations)
    else:
        # Constraint Score
        compliant_count = sum(
            1 for n in parent_nodes if n["child_count"] <= max_points_per_parent
        )
        constraint_score = compliant_count / len(parent_nodes)
        per_parent_constraint_ok = 1.0 if constraint_score == 1.0 else 0.0
        
        # Grouping Score
        grouping_sum = 0.0
        for n in parent_nodes:
            cc = n["child_count"]
            if cc > max_points_per_parent:
                score = 0.0
            elif 3 <= cc <= max_points_per_parent:
                score = 1.0
            elif cc == 2:
                score = 0.9  # Good but not optimal
            elif cc == 1:
                score = 0.5  # Poor grouping
            else:
                score = 0.0
            grouping_sum += score
            
        grouping_score = grouping_sum / len(parent_nodes)
    
    # MECE compliance: All constraints must pass + depth ok
    mece_compliant = 1.0 if (depth_ok == 1.0 and per_parent_constraint_ok == 1.0) else 0.0
    
    # MECE Score 2.0 (Weighted)
    # Weights: Constraint 0.4, Grouping 0.4, Depth 0.2
    mece_score = (constraint_score * 0.4) + (grouping_score * 0.4) + (depth_score * 0.2)

    if verbose:
        print("Structure metrics (Pyramid Quality 2.0):")
        print(f"\n  Max depth: {max_depth} (required: >= {min_depth})")
        print(f"  Depth score: {depth_score:.4f} (weight 0.2)")
        print(f"  Constraint score: {constraint_score:.4f} (weight 0.4)")
        print(f"  Grouping score: {grouping_score:.4f} (weight 0.4)")
        print(f"  MECE compliant: {bool(mece_compliant)}")
        print(f"  MECE score: {mece_score:.4f}")
        if comp_pred["violations"]:
            print(f"\n  Violations: {comp_pred['violations']}")

    return {
        "depth_ok": depth_ok,
        "max_depth": float(max_depth),
        "depth_score": depth_score,
        "per_parent_constraint_ok": per_parent_constraint_ok,
        "constraint_score": constraint_score,
        "grouping_score": grouping_score,
        "total_points": float(total_points),
        "mece_compliant": mece_compliant,
        "mece_score": mece_score,
    }


def eval_pair(
    pred: str,
    ref: str,
    min_depth: int = 3,
    max_points_per_parent: int = 5,
    verbose: bool = True,
) -> Dict:
    """Evaluate a single (prediction, reference) pair with multiple metrics.

    Metrics:
    - string_similarity
    - rouge_l
    - bleu
    - structure_level_accuracy
    - depth_match
    - per_parent_constraint_ok
    """
    s_sim = string_similarity(pred, ref)
    r_l = rouge_l(pred, ref)
    b = bleu_score(pred, ref)
    s_metrics = structure_metrics(
        pred,
        min_depth=min_depth,
        max_points_per_parent=max_points_per_parent,
        verbose=verbose,
    )

    metrics: Dict[str, float] = {
        "string_similarity": s_sim,
        "rouge_l": r_l,
        "bleu": b,
        **s_metrics,
    }

    if verbose:
        print("=" * 80)
        print("Evaluation results (single sample)")
        print("=" * 80)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("=" * 80)

    return metrics


__all__ = [
    "HierarchicalStructureEvaluator",
    "check_compliance",
    "string_similarity",
    "rouge_l",
    "bleu_score",
    "structure_metrics",
    "eval_pair",
]



if __name__ == "__main__":
    print("Running structure metric tests...")
    
    test_cases = [
        {
            "name": "1. Perfect Pyramid (完美金字塔)",
            "desc": "层级 >= 3，每个父节点下有 3-4 个子节点。预期高分 (>0.9)。",
            "text": """* **1. Level 1**
  * 1.1 Sub A
    * Detail 1
    * Detail 2
    * Detail 3
  * 1.2 Sub B
    * Detail 1
    * Detail 2
    * Detail 3"""
        },
        {
            "name": "2. Poor Grouping (单传结构)",
            "desc": "层级够深，但每个父节点只有一个子节点。预期分数受 Grouping 惩罚 (~0.6-0.7)。",
            "text": """* **1. Level 1**
  * 1.1 Sub A
    * Detail 1
* **2. Level 1**
  * 2.1 Sub B
    * Detail 1"""
        },
        {
            "name": "3. Constraint Violation (子节点过多)",
            "desc": "某个父节点下有 8 个子节点 (>5)。Constraint Score 应惩罚。",
            "text": """* **1. Level 1**
  * 1.1 Sub A
    * Item 1
    * Item 2
    * Item 3
    * Item 4
    * Item 5
    * Item 6
    * Item 7
    * Item 8"""
        },
        {
            "name": "4. Shallow Structure (深度不足)",
            "desc": "只有 2 层。Depth Score 应惩罚。",
            "text": """* **1. Level 1**
* **2. Level 1**
* **3. Level 1**"""
        },
        {
            "name": "5. No Structure (无结构/拒绝回答)",
            "desc": "普通文本，无列表。预期 0 分。",
            "text": """I don't know. This is just a sentence."""
        },
        {
            "name": "6. Borderline (每层2个)",
            "desc": "每层正好 2 个子节点。预期分数不错，但不如完美金字塔 (~0.85)。",
            "text": """* **1. A**
  * 1.1 AA
    * AAA
    * AAB
  * 1.2 AB
    * ABA
    * ABB"""
        }
    ]

    print(f"{'Test Case Name':<35} | {'Score':<6} | {'Depth':<5} | {'Constr':<6} | {'Group':<6} | {'Status'}")
    print("-" * 95)

    for case in test_cases:
        metrics = structure_metrics(case['text'], verbose=False)
        score = metrics['mece_score']
        
        if score > 0.9: status = "🌟 Excellent"
        elif score > 0.8: status = "✅ Good"
        elif score > 0.6: status = "⚠️ Fair"
        else: status = "❌ Poor"
        
        print(f"{case['name']:<35} | {score:.4f} | {metrics['depth_score']:.2f}  | {metrics['constraint_score']:.2f}   | {metrics['grouping_score']:.2f}   | {status}")

    print("-" * 95)