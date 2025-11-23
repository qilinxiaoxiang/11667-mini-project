#!/usr/bin/env python3
"""
查看训练集和测试集的划分情况
使用与训练脚本相同的划分方式（seed=42, test_size=0.1）
显示训练集和测试集的索引，并验证它们不重合
"""

import sys
import os
from pathlib import Path
from datasets import load_from_disk

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    # 数据集路径
    dataset_path = project_root / "data" / "processed" / "hierarchical_dataset_clean"
    
    print("=" * 60)
    print("训练集/测试集划分查看工具")
    print("=" * 60)
    print(f"\n加载数据集: {dataset_path}")
    
    # 加载数据集
    dataset = load_from_disk(str(dataset_path))
    print(f"✓ 数据集加载成功，共 {len(dataset)} 个样本\n")
    
    # 使用与训练脚本相同的划分方式
    # seed=42, test_size=0.1 (10% 测试集，90% 训练集)
    print("=" * 60)
    print("数据集划分 (使用 seed=42, test_size=0.1)")
    print("=" * 60)
    
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    print(f"\n训练集大小: {len(train_dataset)} 个样本 ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"测试集大小: {len(test_dataset)} 个样本 ({len(test_dataset)/len(dataset)*100:.1f}%)")
    print(f"总计:       {len(dataset)} 个样本")
    
    # 获取训练集和测试集的原始索引
    # train_test_split 会重新索引，我们需要找到原始索引映射
    print("\n" + "=" * 60)
    print("查找训练集和测试集的原始索引")
    print("=" * 60)
    
    # 通过比较原始数据集和 split 后的数据集来找到原始索引
    # 创建一个签名来唯一标识每个样本 (Description + Patient 的前100个字符)
    original_signatures = {}
    for idx, sample in enumerate(dataset):
        signature = f"{sample['Description']}|||{sample['Patient'][:200]}"
        if signature not in original_signatures:
            original_signatures[signature] = idx
    
    # 查找训练集的原始索引
    train_original_indices = []
    for train_sample in train_dataset:
        signature = f"{train_sample['Description']}|||{train_sample['Patient'][:200]}"
        if signature in original_signatures:
            train_original_indices.append(original_signatures[signature])
        else:
            print(f"⚠ 警告: 训练集样本未找到原始索引")
    
    # 查找测试集的原始索引
    test_original_indices = []
    for test_sample in test_dataset:
        signature = f"{test_sample['Description']}|||{test_sample['Patient'][:200]}"
        if signature in original_signatures:
            test_original_indices.append(original_signatures[signature])
        else:
            print(f"⚠ 警告: 测试集样本未找到原始索引")
    
    train_original_indices.sort()
    test_original_indices.sort()
    
    print(f"\n训练集原始索引范围: {min(train_original_indices)} 到 {max(train_original_indices)}")
    print(f"训练集样本数: {len(train_original_indices)}")
    print(f"\n测试集原始索引范围: {min(test_original_indices)} 到 {max(test_original_indices)}")
    print(f"测试集样本数: {len(test_original_indices)}")
    
    # 显示前20个和后20个索引
    print(f"\n训练集原始索引 (前20个): {train_original_indices[:20]}")
    if len(train_original_indices) > 20:
        print(f"训练集原始索引 (后20个): {train_original_indices[-20:]}")
    
    print(f"\n测试集原始索引 (前20个): {test_original_indices[:20]}")
    if len(test_original_indices) > 20:
        print(f"测试集原始索引 (后20个): {test_original_indices[-20:]}")
    
    # 验证训练集和测试集不重合
    train_set = set(train_original_indices)
    test_set = set(test_original_indices)
    overlap = train_set & test_set
    
    print("\n" + "=" * 60)
    print("验证训练集和测试集不重合")
    print("=" * 60)
    
    if len(overlap) == 0:
        print("\n✓ 验证通过：训练集和测试集完全不重合")
        print("✓ 使用 seed=42 保证划分的可重现性")
    else:
        print(f"\n✗ 警告：发现 {len(overlap)} 个重叠的索引: {list(overlap)[:10]}")
    
    # 显示测试集的样本
    print("\n" + "=" * 60)
    print("测试集样本预览 (前 5 个)")
    print("=" * 60)
    
    for i in range(min(5, len(test_dataset))):
        sample = test_dataset[i]
        print(f"\n--- 测试集样本 #{i} ---")
        print(f"Description: {sample['Description']}")
        print(f"Patient: {sample['Patient'][:150]}..." if len(sample['Patient']) > 150 else f"Patient: {sample['Patient']}")
        print(f"Status: {sample['Status']}")
        print(f"Doctor Response (前300字符):")
        print(sample['Doctor'][:300] + "..." if len(sample['Doctor']) > 300 else sample['Doctor'])
    
    # 保存测试集索引到文件（用于后续评估）
    print("\n" + "=" * 60)
    print("保存测试集信息")
    print("=" * 60)
    
    output_dir = project_root / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    # 保存测试集的数据（用于后续评估）
    test_dataset_path = output_dir / "test_dataset"
    test_dataset.save_to_disk(str(test_dataset_path))
    print(f"\n✓ 测试集已保存到: {test_dataset_path}")
    
    # 保存索引信息到 JSON 文件
    import json
    indices_info = {
        "train_indices": train_original_indices,
        "test_indices": test_original_indices,
        "train_size": len(train_original_indices),
        "test_size": len(test_original_indices),
        "total_size": len(dataset),
        "seed": 42,
        "test_size_ratio": 0.1
    }
    
    indices_file = output_dir / "train_test_indices.json"
    with open(indices_file, 'w', encoding='utf-8') as f:
        json.dump(indices_info, f, indent=2, ensure_ascii=False)
    print(f"✓ 索引信息已保存到: {indices_file}")
    
    # 保存训练集的数据（用于参考）
    train_dataset_path = output_dir / "train_dataset"
    train_dataset.save_to_disk(str(train_dataset_path))
    print(f"✓ 训练集已保存到: {train_dataset_path}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    
    # 按 Status 统计
    train_status = {}
    test_status = {}
    
    for sample in train_dataset:
        status = sample.get('Status', 'unknown')
        train_status[status] = train_status.get(status, 0) + 1
    
    for sample in test_dataset:
        status = sample.get('Status', 'unknown')
        test_status[status] = test_status.get(status, 0) + 1
    
    print("\n训练集 Status 分布:")
    for status, count in sorted(train_status.items()):
        print(f"  {status}: {count} ({count/len(train_dataset)*100:.1f}%)")
    
    print("\n测试集 Status 分布:")
    for status, count in sorted(test_status.items()):
        print(f"  {status}: {count} ({count/len(test_dataset)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n提示：")
    print("  - 使用 seed=42 和 test_size=0.1 保证与训练脚本一致")
    print("  - 测试集和训练集已自动保存到 evaluation/ 目录")
    print("  - 可以使用这些数据集进行后续评估")
    print()


if __name__ == "__main__":
    main()

