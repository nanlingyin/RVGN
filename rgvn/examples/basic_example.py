import os
import sys
import json
import matplotlib.pyplot as plt
import networkx as nx

# 确保能导入rgvn模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rgvn.llm.api_client import LLMAPIClient
from rgvn.core.graph_builder import ReasoningGraphBuilder
from rgvn.core.error_detector import GraphErrorDetector
from rgvn.validators.math_validator import MathValidator

def visualize_graph(graph, title="推理图结构"):
    """可视化推理图"""
    plt.figure(figsize=(12, 8))
    
    # 获取节点位置
    pos = {}
    for node, data in graph.nodes(data=True):
        # 水平布局，按步骤顺序排列
        pos[node] = (data['position'], 0)
    
    # 设置节点颜色，基于置信度
    node_colors = [data['confidence'] for _, data in graph.nodes(data=True)]
    
    # 设置边宽度，基于权重
    edge_widths = [data['weight'] * 3 for _, _, data in graph.edges(data=True)]
    
    # 设置边颜色，区分类型
    edge_colors = ['blue' if data['type']=='sequential' else 'green' 
                   for _, _, data in graph.edges(data=True)]
    
    # 绘制节点
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        cmap=plt.cm.Reds,
        node_size=800,
        alpha=0.8
    )
    
    # 绘制边
    nx.draw_networkx_edges(
        graph, pos,
        width=edge_widths,
        alpha=0.7,
        edge_color=edge_colors,
        arrowsize=15
    )
    
    # 添加节点标签
    labels = {node: f"步骤 {data['position']}" for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels, font_size=12)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # 配置API密钥
    api_key = input("请输入您的OpenAI API密钥: ")
    if not api_key:
        print("错误：需要API密钥才能继续")
        return
    
    # 初始化组件
    llm_client = LLMAPIClient(provider="openai", model="gpt-4o", api_key=api_key)
    graph_builder = ReasoningGraphBuilder()
    error_detector = GraphErrorDetector()
    math_validator = MathValidator()
    
    # 获取用户问题
    print("\n--- RGVN推理验证系统 ---\n")
    problem = input("请输入您想解决的问题 (默认是积分问题): ")
    if not problem:
        problem = "计算积分 ∫(x²+3x+2)dx"
    
    print(f"\n问题: {problem}\n")
    
    # 生成推理
    print("正在使用LLM生成推理过程...")
    reasoning = llm_client.generate_reasoning(problem)
    print("\n原始推理内容:")
    print("-" * 70)
    print(reasoning)
    print("-" * 70)
    
    # 构建推理图
    print("\n构建推理图结构...")
    graph = graph_builder.build_graph(reasoning)
    print(f"图创建完成: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
      # 验证数学表达式
    print("\n验证数学表达式...")
    math_validation = math_validator.validate_expressions(reasoning)
    print(f"提取了 {len(math_validation)} 个数学表达式")
    
    # 显示验证结果
    for i, expr in enumerate(math_validation):
        validity = "✓ 有效" if expr.get('is_valid', False) else "✗ 无效"
        print(f"{i+1}. {expr['equation']} - {validity}")
        if 'note' in expr:
            print(f"   备注: {expr['note']}")
    
    # 检测错误
    print("\n检测推理错误...")
    errors = error_detector.detect_errors(graph)
    
    # 显示错误
    if errors["errors"]:
        print("\n发现以下错误:")
        for error in errors["errors"]:
            print(f"  - {error['type']} 在 {error['location']}: {error['description']}")
    else:
        print("\n未发现明显推理错误")
    
    if errors["low_confidence"]:
        print("\n低置信度节点:")
        for node, score, reason in errors["low_confidence"]:
            print(f"  - {node}: 置信度 {score:.2f} - {reason}")
    
    # 可视化推理图
    visualize = input("\n是否可视化推理图? (y/n): ")
    if visualize.lower() == 'y':
        visualize_graph(graph)
    
    # 如果发现错误，生成改进建议
    if errors["errors"] or errors["low_confidence"]:
        generate_feedback = input("\n是否生成改进建议? (y/n): ")
        if generate_feedback.lower() == 'y':
            print("\n生成改进建议...")
            improved_reasoning = llm_client.critique_reasoning(reasoning, errors)
            
            print("\n改进后的推理:")
            print("-" * 70)
            print(improved_reasoning)
            print("-" * 70)
    
    print("\nRGVN分析完成!")

if __name__ == "__main__":
    main()