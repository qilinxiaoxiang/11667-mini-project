#!/usr/bin/env python3
"""
评估 ICL 结果的脚本

从 ICL 评估结果 JSON 文件中加载数据，计算各种评估指标：
- 文本相似度指标：string_similarity, ROUGE-L, BLEU
- 结构指标：深度、MECE合规性、结构分数等
- 生成统计报告和详细结果

Usage:
    python scripts/eval_icl_results.py evaluation/icl_results/icl_eval_train_20251123_203244/icl_results_checkpoint_20.json
    
    或评估所有检查点文件：
    python scripts/eval_icl_results.py evaluation/icl_results/icl_eval_train_20251123_203244/ --all
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import eval_pair, structure_metrics, check_compliance


def load_icl_results(json_file: str) -> List[Dict]:
    """加载ICL结果JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def evaluate_single_result(result: Dict, verbose: bool = False) -> Dict:
    """
    评估单个ICL结果
    
    Args:
        result: 包含 generated_answer 和 reference_answer 的结果字典
        verbose: 是否打印详细信息
        
    Returns:
        包含所有评估指标的字典
    """
    generated = result.get('generated_answer', '')
    reference = result.get('reference_answer', '')
    
    # 如果生成为 "FAILED"，跳过评估
    if not generated or generated == "FAILED":
        return {
            "success": False,
            "error": "Generation failed"
        }
    
    # 使用 eval_pair 进行综合评估
    metrics = eval_pair(
        pred=generated,
        ref=reference,
        min_depth=3,
        max_points_per_parent=5,
        verbose=verbose
    )
    
    # 添加额外的元信息
    metrics.update({
        "index": result.get("index", -1),
        "description": result.get("description", ""),
        "status": result.get("status", ""),
        "generation_time": result.get("generation_time", 0.0),
        "success": True,
        "generated_length": len(generated),
        "reference_length": len(reference),
    })
    
    return metrics


def evaluate_all_results(results: List[Dict], verbose: bool = False) -> Dict:
    """
    评估所有ICL结果并生成统计报告
    
    Args:
        results: ICL结果列表
        verbose: 是否打印每个样本的详细信息
        
    Returns:
        包含所有评估结果的字典
    """
    all_metrics = []
    failed_count = 0
    
    print(f"\n开始评估 {len(results)} 个结果...")
    print("=" * 80)
    
    for i, result in enumerate(results):
        if verbose:
            print(f"\n[{i+1}/{len(results)}] 评估样本 {result.get('index', i)}...")
        
        metrics = evaluate_single_result(result, verbose=verbose)
        
        if metrics.get("success", False):
            all_metrics.append(metrics)
        else:
            failed_count += 1
            if verbose:
                print(f"  ✗ 跳过失败的生成")
    
    if not all_metrics:
        print("\n⚠ 警告：没有成功的结果可以评估！")
        return {
            "total_samples": len(results),
            "successful_samples": 0,
            "failed_samples": failed_count,
        }
    
    # 计算统计指标
    stats = compute_statistics(all_metrics)
    stats.update({
        "total_samples": len(results),
        "successful_samples": len(all_metrics),
        "failed_samples": failed_count,
        "detailed_results": all_metrics,
    })
    
    return stats


def compute_statistics(metrics_list: List[Dict]) -> Dict:
    """计算评估指标的统计信息"""
    if not metrics_list:
        return {}
    
    # 提取所有指标
    metric_names = [
        "string_similarity",
        "rouge_l",
        "bleu",
        "mece_score",
        "mece_compliant",
        "depth_score",
        "constraint_score",
        "grouping_score",
        "max_depth",
        "total_points",
        "generation_time",
        "generated_length",
        "reference_length",
    ]
    
    stats = {}
    
    for metric_name in metric_names:
        values = [m.get(metric_name, 0) for m in metrics_list if metric_name in m]
        if values:
            stats[f"{metric_name}_mean"] = float(np.mean(values))
            stats[f"{metric_name}_std"] = float(np.std(values))
            stats[f"{metric_name}_min"] = float(np.min(values))
            stats[f"{metric_name}_max"] = float(np.max(values))
            stats[f"{metric_name}_median"] = float(np.median(values))
            
            # 对于合规性指标，计算百分比
            if metric_name in ["mece_compliant"]:
                stats[f"{metric_name}_rate"] = float(np.mean(values))
    
    # 计算合规率
    compliant_count = sum(1 for m in metrics_list if m.get("mece_compliant", 0) == 1.0)
    stats["mece_compliant_rate"] = compliant_count / len(metrics_list) if metrics_list else 0.0
    
    # 按严重程度分组统计
    severity_stats = {}
    for severity in ["low severity", "medium severity", "high severity"]:
        severity_metrics = [m for m in metrics_list if m.get("status", "").lower() == severity.lower()]
        if severity_metrics:
            severity_stats[severity] = {
                "count": len(severity_metrics),
                "mece_score_mean": float(np.mean([m.get("mece_score", 0) for m in severity_metrics])),
                "rouge_l_mean": float(np.mean([m.get("rouge_l", 0) for m in severity_metrics])),
                "mece_compliant_rate": float(np.mean([m.get("mece_compliant", 0) for m in severity_metrics])),
            }
    stats["by_severity"] = severity_stats
    
    return stats


def print_summary_report(stats: Dict, output_file: Optional[str] = None):
    """打印评估摘要报告"""
    report_lines = []
    
    def add_line(text: str = ""):
        report_lines.append(text)
        print(text)
    
    add_line("\n" + "=" * 80)
    add_line("ICL 评估结果摘要")
    add_line("=" * 80)
    
    # 基本统计
    add_line(f"\n样本统计:")
    add_line(f"  总样本数: {stats.get('total_samples', 0)}")
    add_line(f"  成功评估: {stats.get('successful_samples', 0)}")
    add_line(f"  生成失败: {stats.get('failed_samples', 0)}")
    success_rate = stats.get('successful_samples', 0) / stats.get('total_samples', 1) * 100
    add_line(f"  成功率: {success_rate:.1f}%")
    
    # 文本相似度指标
    add_line(f"\n文本相似度指标:")
    if 'string_similarity_mean' in stats:
        add_line(f"  String Similarity: {stats['string_similarity_mean']:.4f} ± {stats['string_similarity_std']:.4f}")
    if 'rouge_l_mean' in stats:
        add_line(f"  ROUGE-L:          {stats['rouge_l_mean']:.4f} ± {stats['rouge_l_std']:.4f}")
    if 'bleu_mean' in stats:
        add_line(f"  BLEU:             {stats['bleu_mean']:.4f} ± {stats['bleu_std']:.4f}")
    
    # 结构指标
    add_line(f"\n结构指标:")
    if 'mece_score_mean' in stats:
        add_line(f"  MECE Score:       {stats['mece_score_mean']:.4f} ± {stats['mece_score_std']:.4f}")
    if 'mece_compliant_rate' in stats:
        compliant_rate = stats['mece_compliant_rate'] * 100
        add_line(f"  MECE 合规率:      {compliant_rate:.1f}%")
    if 'depth_score_mean' in stats:
        add_line(f"  Depth Score:      {stats['depth_score_mean']:.4f} ± {stats['depth_score_std']:.4f}")
    if 'constraint_score_mean' in stats:
        add_line(f"  Constraint Score: {stats['constraint_score_mean']:.4f} ± {stats['constraint_score_std']:.4f}")
    if 'grouping_score_mean' in stats:
        add_line(f"  Grouping Score:   {stats['grouping_score_mean']:.4f} ± {stats['grouping_score_std']:.4f}")
    
    # 结构深度
    if 'max_depth_mean' in stats:
        add_line(f"\n结构深度:")
        add_line(f"  平均深度:         {stats['max_depth_mean']:.2f} ± {stats['max_depth_std']:.2f}")
        add_line(f"  最小深度:         {stats['max_depth_min']:.0f}")
        add_line(f"  最大深度:         {stats['max_depth_max']:.0f}")
    
    # 生成时间和长度
    if 'generation_time_mean' in stats:
        add_line(f"\n生成性能:")
        add_line(f"  平均生成时间:     {stats['generation_time_mean']:.2f}s ± {stats['generation_time_std']:.2f}s")
    if 'generated_length_mean' in stats:
        add_line(f"  平均生成长度:     {stats['generated_length_mean']:.0f} ± {stats['generated_length_std']:.0f} 字符")
    
    # 按严重程度分组
    if 'by_severity' in stats and stats['by_severity']:
        add_line(f"\n按严重程度分组统计:")
        for severity, sev_stats in stats['by_severity'].items():
            add_line(f"\n  {severity}:")
            add_line(f"    样本数: {sev_stats['count']}")
            add_line(f"    MECE Score: {sev_stats['mece_score_mean']:.4f}")
            add_line(f"    ROUGE-L: {sev_stats['rouge_l_mean']:.4f}")
            add_line(f"    MECE 合规率: {sev_stats['mece_compliant_rate']*100:.1f}%")
    
    add_line("\n" + "=" * 80)
    
    # 保存报告到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        add_line(f"\n报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="评估ICL结果")
    parser.add_argument(
        "input_path",
        type=str,
        help="ICL结果JSON文件路径或包含多个JSON文件的目录"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="如果输入是目录，评估所有JSON文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告文件路径（可选）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个样本的详细评估信息"
    )
    parser.add_argument(
        "--save-details",
        type=str,
        default=None,
        help="保存详细评估结果到JSON文件（可选）"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # 确定要评估的文件
    json_files = []
    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        if args.all:
            # 评估目录下所有JSON文件
            json_files = sorted(input_path.glob("*.json"))
        else:
            # 只评估最新的checkpoint文件
            checkpoint_files = sorted(input_path.glob("icl_results_checkpoint_*.json"))
            if checkpoint_files:
                json_files = [checkpoint_files[-1]]  # 最新的检查点
            else:
                # 如果没有checkpoint，尝试找最终结果文件
                result_files = sorted(input_path.glob("icl_results_*.json"))
                if result_files:
                    json_files = [result_files[-1]]
                else:
                    print(f"错误：目录 {input_path} 中未找到JSON文件")
                    sys.exit(1)
    else:
        print(f"错误：路径不存在: {input_path}")
        sys.exit(1)
    
    if not json_files:
        print(f"错误：未找到要评估的JSON文件")
        sys.exit(1)
    
    print(f"找到 {len(json_files)} 个JSON文件进行评估")
    
    # 评估每个文件
    all_results = {}
    for json_file in json_files:
        print(f"\n{'='*80}")
        print(f"评估文件: {json_file}")
        print(f"{'='*80}")
        
        results = load_icl_results(str(json_file))
        stats = evaluate_all_results(results, verbose=args.verbose)
        
        file_key = json_file.stem
        all_results[file_key] = stats
        
        # 打印摘要
        print_summary_report(stats)
    
    # 如果只有一个文件，设置输出路径
    if len(json_files) == 1:
        json_file = json_files[0]
        output_dir = json_file.parent
        
        # 生成默认输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output is None:
            args.output = output_dir / f"eval_report_{timestamp}.txt"
        if args.save_details is None:
            args.save_details = output_dir / f"eval_details_{timestamp}.json"
        
        # 保存详细结果（包含每个样本的评估分数）
        if args.save_details:
            # 保存完整结果，包括每个样本的详细评估分数
            full_results = all_results[list(all_results.keys())[0]].copy()
            with open(args.save_details, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存到: {args.save_details}")
            detailed_count = len(full_results.get('detailed_results', []))
            print(f"  - 包含 {detailed_count} 个样本的完整评估分数")
            
            # 同时保存为CSV格式，方便查看每个样本的分数
            csv_file = str(args.save_details).replace('.json', '.csv')
            detailed_results = full_results.get('detailed_results', [])
            if detailed_results:
                # 提取所有可能的字段名
                all_fields = set()
                for result in detailed_results:
                    all_fields.update(result.keys())
                
                # 定义字段顺序（重要指标在前）
                priority_fields = [
                    'index', 'status', 'success',
                    'string_similarity', 'rouge_l', 'bleu',
                    'mece_score', 'mece_compliant', 'depth_score', 
                    'constraint_score', 'grouping_score',
                    'max_depth', 'total_points',
                    'generation_time', 'generated_length', 'reference_length'
                ]
                
                # 构建字段列表：优先字段 + 其他字段
                fieldnames = []
                for field in priority_fields:
                    if field in all_fields:
                        fieldnames.append(field)
                        all_fields.remove(field)
                fieldnames.extend(sorted(all_fields))  # 其余字段按字母顺序
                
                # 写入CSV
                with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for result in detailed_results:
                        # 处理列表和字典类型的值，转换为字符串
                        row = {}
                        for key in fieldnames:
                            value = result.get(key, '')
                            if isinstance(value, (list, dict)):
                                value = json.dumps(value, ensure_ascii=False)
                            row[key] = value
                        writer.writerow(row)
                print(f"  - CSV格式已保存到: {csv_file}")
        
        # 另外保存一个只包含统计摘要的文件（可选）
        summary_file = output_dir / f"eval_summary_{timestamp}.json"
        summary_only = {k: v for k, v in all_results[list(all_results.keys())[0]].items() 
                       if k != 'detailed_results'}
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_only, f, indent=2, ensure_ascii=False)
        print(f"统计摘要已保存到: {summary_file}")
    
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
