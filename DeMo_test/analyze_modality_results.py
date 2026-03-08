#!/usr/bin/env python3
"""
模态消融实验结果分析与可视化

功能：
1. 读取训练过程中保存的模态消融实验结果
2. 生成详细的分析报告
3. 绘制性能对比图表
4. 识别主导模态

用法：
python analyze_modality_results.py --result_dir logs/RGBNT201/demo/
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from collections import defaultdict

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_results(result_file):
    """加载模态消融实验结果"""
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"结果文件不存在: {result_file}")

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results


def plot_performance_curves(results, output_dir):
    """绘制性能变化曲线"""
    print("绘制性能曲线...")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('模态消融实验 - 性能对比曲线', fontsize=16, fontweight='bold')

    # 定义颜色和标记
    colors = {
        'all': '#2E86DE',
        'missing_RGB': '#EE5A24',
        'missing_NI': '#009432',
        'missing_TI': '#8E44AD'
    }

    labels = {
        'all': '所有模态（基准）',
        'missing_RGB': '缺失RGB',
        'missing_NI': '缺失近红外',
        'missing_TI': '缺失热红外'
    }

    markers = {
        'all': 'o',
        'missing_RGB': 's',
        'missing_NI': '^',
        'missing_TI': 'D'
    }

    # 绘制mAP曲线
    ax1 = axes[0, 0]
    for pattern_name in ['all', 'missing_RGB', 'missing_NI', 'missing_TI']:
        if pattern_name in results and 'mAP' in results[pattern_name]:
            epochs = results[pattern_name].get('epoch', list(range(1, len(results[pattern_name]['mAP']) + 1)))
            mAP_values = [v * 100 for v in results[pattern_name]['mAP']]

            ax1.plot(epochs, mAP_values,
                    label=labels[pattern_name],
                    color=colors[pattern_name],
                    marker=markers[pattern_name],
                    markersize=6,
                    linewidth=2,
                    alpha=0.8)

    ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax1.set_ylabel('mAP (%)', fontsize=12)
    ax1.set_title('mAP性能对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 绘制Rank-1曲线
    ax2 = axes[0, 1]
    for pattern_name in ['all', 'missing_RGB', 'missing_NI', 'missing_TI']:
        if pattern_name in results and 'Rank-1' in results[pattern_name]:
            epochs = results[pattern_name].get('epoch', list(range(1, len(results[pattern_name]['Rank-1']) + 1)))
            rank1_values = [v * 100 for v in results[pattern_name]['Rank-1']]

            ax2.plot(epochs, rank1_values,
                    label=labels[pattern_name],
                    color=colors[pattern_name],
                    marker=markers[pattern_name],
                    markersize=6,
                    linewidth=2,
                    alpha=0.8)

    ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax2.set_ylabel('Rank-1 准确率 (%)', fontsize=12)
    ax2.set_title('Rank-1性能对比', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 绘制性能差异（相对基准）
    ax3 = axes[1, 0]
    if 'all' in results and 'mAP' in results['all']:
        baseline_mAP = np.array(results['all']['mAP'])

        for pattern_name in ['missing_RGB', 'missing_NI', 'missing_TI']:
            if pattern_name in results and 'mAP' in results[pattern_name]:
                epochs = results[pattern_name].get('epoch', list(range(1, len(results[pattern_name]['mAP']) + 1)))
                pattern_mAP = np.array(results[pattern_name]['mAP'])

                # 确保长度一致
                min_len = min(len(baseline_mAP), len(pattern_mAP))
                mAP_diff = (baseline_mAP[:min_len] - pattern_mAP[:min_len]) * 100

                ax3.plot(epochs[:min_len], mAP_diff,
                        label=labels[pattern_name],
                        color=colors[pattern_name],
                        marker=markers[pattern_name],
                        markersize=6,
                        linewidth=2,
                        alpha=0.8)

    ax3.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax3.set_ylabel('mAP下降 (%)', fontsize=12)
    ax3.set_title('相对基准的mAP下降幅度', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # 绘制平均性能对比柱状图
    ax4 = axes[1, 1]
    pattern_names_list = ['all', 'missing_RGB', 'missing_NI', 'missing_TI']
    avg_mAP = []
    avg_rank1 = []

    for pattern_name in pattern_names_list:
        if pattern_name in results and 'mAP' in results[pattern_name]:
            avg_mAP.append(np.mean(results[pattern_name]['mAP']) * 100)
            avg_rank1.append(np.mean(results[pattern_name]['Rank-1']) * 100)
        else:
            avg_mAP.append(0)
            avg_rank1.append(0)

    x = np.arange(len(pattern_names_list))
    width = 0.35

    bars1 = ax4.bar(x - width/2, avg_mAP, width, label='平均 mAP',
                    color=[colors[p] for p in pattern_names_list], alpha=0.8)
    bars2 = ax4.bar(x + width/2, avg_rank1, width, label='平均 Rank-1',
                    color=[colors[p] for p in pattern_names_list], alpha=0.5)

    ax4.set_xlabel('模态配置', fontsize=12)
    ax4.set_ylabel('准确率 (%)', fontsize=12)
    ax4.set_title('平均性能对比', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([labels[p] for p in pattern_names_list], rotation=15, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 保存图表
    output_file = os.path.join(output_dir, 'modality_performance_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"性能曲线图已保存: {output_file}")

    plt.close()


def plot_modality_importance(results, output_dir):
    """绘制模态重要性分析图"""
    print("绘制模态重要性分析图...")

    if 'all' not in results or 'mAP' not in results['all']:
        print("缺少基准数据，无法绘制模态重要性图")
        return

    baseline_mAP = np.mean(results['all']['mAP'])
    baseline_rank1 = np.mean(results['all']['Rank-1'])

    modality_impacts = []

    for pattern_name in ['missing_RGB', 'missing_NI', 'missing_TI']:
        if pattern_name in results and 'mAP' in results[pattern_name]:
            pattern_mAP = np.mean(results[pattern_name]['mAP'])
            pattern_rank1 = np.mean(results[pattern_name]['Rank-1'])

            mAP_drop = (baseline_mAP - pattern_mAP) * 100
            rank1_drop = (baseline_rank1 - pattern_rank1) * 100

            modality_name = pattern_name.replace('missing_', '')

            modality_impacts.append({
                'modality': modality_name,
                'mAP_drop': mAP_drop,
                'rank1_drop': rank1_drop
            })

    # 按mAP下降幅度排序
    modality_impacts.sort(key=lambda x: x['mAP_drop'], reverse=True)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('模态重要性分析', fontsize=16, fontweight='bold')

    # 模态重要性排名（mAP）
    modalities = [item['modality'] for item in modality_impacts]
    mAP_drops = [item['mAP_drop'] for item in modality_impacts]

    colors_importance = ['#EE5A24', '#F79F1F', '#FFC312']

    bars = ax1.barh(modalities, mAP_drops, color=colors_importance, alpha=0.8)
    ax1.set_xlabel('mAP下降 (%)', fontsize=12)
    ax1.set_title('模态重要性排名（按mAP下降）', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, mAP_drops)):
        ax1.text(value + 0.1, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}%',
                va='center', fontsize=10, fontweight='bold')

        # 标注排名
        ax1.text(-0.5, bar.get_y() + bar.get_height()/2,
                f'#{i+1}',
                va='center', ha='right', fontsize=12, fontweight='bold',
                color=colors_importance[i])

    # mAP vs Rank-1 下降对比
    rank1_drops = [item['rank1_drop'] for item in modality_impacts]

    x = np.arange(len(modalities))
    width = 0.35

    bars1 = ax2.bar(x - width/2, mAP_drops, width, label='mAP下降',
                    color='#EE5A24', alpha=0.8)
    bars2 = ax2.bar(x + width/2, rank1_drops, width, label='Rank-1下降',
                    color='#8E44AD', alpha=0.8)

    ax2.set_xlabel('模态', fontsize=12)
    ax2.set_ylabel('性能下降 (%)', fontsize=12)
    ax2.set_title('mAP vs Rank-1 性能下降对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(modalities)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存图表
    output_file = os.path.join(output_dir, 'modality_importance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"模态重要性图已保存: {output_file}")

    plt.close()

    return modality_impacts


def generate_detailed_report(results, output_dir):
    """生成详细的文本报告"""
    print("生成详细分析报告...")

    report_file = os.path.join(output_dir, 'detailed_analysis_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("多模态ReID - 模态消融实验详细分析报告\n")
        f.write("="*80 + "\n\n")

        # 1. 实验概览
        f.write("1. 实验概览\n")
        f.write("-"*80 + "\n")
        f.write(f"实验模式数量: {len(results)}\n")
        f.write(f"训练轮数: {len(results.get('all', {}).get('epoch', []))}\n\n")

        # 2. 各模式平均性能
        f.write("2. 各模式平均性能\n")
        f.write("-"*80 + "\n")

        pattern_stats = {}
        for pattern_name in ['all', 'missing_RGB', 'missing_NI', 'missing_TI']:
            if pattern_name in results and 'mAP' in results[pattern_name]:
                mAP_values = results[pattern_name]['mAP']
                rank1_values = results[pattern_name]['Rank-1']

                avg_mAP = np.mean(mAP_values)
                std_mAP = np.std(mAP_values)
                avg_rank1 = np.mean(rank1_values)
                std_rank1 = np.std(rank1_values)

                pattern_stats[pattern_name] = {
                    'avg_mAP': avg_mAP,
                    'std_mAP': std_mAP,
                    'avg_rank1': avg_rank1,
                    'std_rank1': std_rank1
                }

                label_map = {
                    'all': '所有模态（基准）',
                    'missing_RGB': '缺失RGB',
                    'missing_NI': '缺失近红外',
                    'missing_TI': '缺失热红外'
                }

                f.write(f"\n{label_map[pattern_name]}:\n")
                f.write(f"  平均 mAP: {avg_mAP:.1%} ± {std_mAP:.1%}\n")
                f.write(f"  平均 Rank-1: {avg_rank1:.1%} ± {std_rank1:.1%}\n")

        # 3. 模态重要性分析
        f.write("\n3. 模态重要性分析\n")
        f.write("-"*80 + "\n")

        if 'all' in pattern_stats:
            baseline = pattern_stats['all']

            modality_impacts = []
            for pattern_name in ['missing_RGB', 'missing_NI', 'missing_TI']:
                if pattern_name in pattern_stats:
                    mAP_drop = baseline['avg_mAP'] - pattern_stats[pattern_name]['avg_mAP']
                    rank1_drop = baseline['avg_rank1'] - pattern_stats[pattern_name]['avg_rank1']

                    mAP_drop_pct = (mAP_drop / baseline['avg_mAP'] * 100) if baseline['avg_mAP'] > 0 else 0
                    rank1_drop_pct = (rank1_drop / baseline['avg_rank1'] * 100) if baseline['avg_rank1'] > 0 else 0

                    modality_name = pattern_name.replace('missing_', '')

                    modality_impacts.append({
                        'modality': modality_name,
                        'mAP_drop': mAP_drop,
                        'mAP_drop_pct': mAP_drop_pct,
                        'rank1_drop': rank1_drop,
                        'rank1_drop_pct': rank1_drop_pct
                    })

            modality_impacts.sort(key=lambda x: x['mAP_drop'], reverse=True)

            f.write("\n模态重要性排名（按平均mAP下降幅度）:\n\n")
            for i, impact in enumerate(modality_impacts, 1):
                f.write(f"{i}. {impact['modality']}:\n")
                f.write(f"   mAP下降: {impact['mAP_drop']:.1%} ({impact['mAP_drop_pct']:.1f}%)\n")
                f.write(f"   Rank-1下降: {impact['rank1_drop']:.1%} ({impact['rank1_drop_pct']:.1f}%)\n\n")

            # 4. 结论
            f.write("4. 结论\n")
            f.write("-"*80 + "\n")
            f.write(f"\n主导模态: {modality_impacts[0]['modality']}\n")
            f.write(f"  - 该模态缺失后，mAP平均下降 {modality_impacts[0]['mAP_drop']:.1%} ({modality_impacts[0]['mAP_drop_pct']:.1f}%)\n")
            f.write(f"  - 该模态缺失后，Rank-1平均下降 {modality_impacts[0]['rank1_drop']:.1%} ({modality_impacts[0]['rank1_drop_pct']:.1f}%)\n")
            f.write(f"  - 这表明{modality_impacts[0]['modality']}模态对ReID性能的贡献最大\n\n")

            f.write(f"次要模态: {modality_impacts[1]['modality']}\n")
            f.write(f"  - 该模态缺失后，mAP平均下降 {modality_impacts[1]['mAP_drop']:.1%} ({modality_impacts[1]['mAP_drop_pct']:.1f}%)\n\n")

            f.write(f"最不重要模态: {modality_impacts[2]['modality']}\n")
            f.write(f"  - 该模态缺失后，mAP平均下降 {modality_impacts[2]['mAP_drop']:.1%} ({modality_impacts[2]['mAP_drop_pct']:.1f}%)\n\n")

        # 5. 建议
        f.write("5. 建议\n")
        f.write("-"*80 + "\n")
        if modality_impacts:
            f.write(f"\n基于实验结果，建议:\n")
            f.write(f"1. 在资源受限的情况下，应优先保留 {modality_impacts[0]['modality']} 模态\n")
            f.write(f"2. {modality_impacts[0]['modality']} 和 {modality_impacts[1]['modality']} 模态的组合可以提供较好的性能\n")
            f.write(f"3. {modality_impacts[2]['modality']} 模态的贡献相对较小，可考虑在轻量化模型中移除\n")
            f.write(f"4. 三个模态结合能获得最佳性能，建议在精度要求高的场景使用全模态\n")

        f.write("\n" + "="*80 + "\n")
        f.write("报告生成完成\n")
        f.write("="*80 + "\n")

    print(f"详细报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="分析模态消融实验结果")
    parser.add_argument('--result_dir', type=str, required=True,
                       help='实验结果目录（包含modality_ablation_results.json的目录）')

    args = parser.parse_args()

    result_file = os.path.join(args.result_dir, 'modality_ablation_results.json')

    print("="*80)
    print("模态消融实验结果分析")
    print("="*80)
    print(f"结果文件: {result_file}")

    # 加载结果
    results = load_results(result_file)

    print(f"加载了 {len(results)} 种模态配置的结果")

    # 绘制性能曲线
    plot_performance_curves(results, args.result_dir)

    # 绘制模态重要性
    modality_impacts = plot_modality_importance(results, args.result_dir)

    # 生成详细报告
    generate_detailed_report(results, args.result_dir)

    print("\n"+"="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"1. {os.path.join(args.result_dir, 'modality_performance_curves.png')}")
    print(f"2. {os.path.join(args.result_dir, 'modality_importance.png')}")
    print(f"3. {os.path.join(args.result_dir, 'detailed_analysis_report.txt')}")
    print("="*80)


if __name__ == '__main__':
    main()
