import networkx as nx
from typing import Dict, List, Any, Optional

class GraphErrorDetector:
    """图结构错误检测器 - 分析推理图中的潜在错误"""
    
    def __init__(self, confidence_threshold=0.6):
        """
        初始化错误检测器
        
        参数:
            confidence_threshold: 置信度阈值，低于此值的节点会被标记
        """
        self.confidence_threshold = confidence_threshold
    
    def detect_errors(self, graph: nx.DiGraph) -> Dict[str, List]:
        """
        检测推理图中的潜在错误
        
        参数:
            graph: 推理图结构
            
        返回:
            包含错误、低置信度节点和建议的字典
        """
        results = {
            "errors": [],
            "low_confidence": [],
            "suggestions": []
        }
        
        # 检测低置信度节点
        for node, data in graph.nodes(data=True):
            if data['confidence'] < self.confidence_threshold:
                location = f"步骤 {data['position']}" if 'position' in data else node
                results["low_confidence"].append((
                    node, 
                    data['confidence'],
                    f"置信度低于阈值 ({self.confidence_threshold})"
                ))
        
        # 检测可能的断点 - 推理链中的逻辑跳跃
        self._detect_reasoning_gaps(graph, results)
        
        # 检测可能的循环依赖
        self._detect_circular_reasoning(graph, results)
        
        # 检测独立子图 - 表示可能的脱节推理
        self._detect_isolated_components(graph, results)
        
        # 生成改进建议
        self._generate_suggestions(results)
        
        return results
    
    def _detect_reasoning_gaps(self, graph: nx.DiGraph, results: Dict) -> None:
        """检测推理中的逻辑跳跃"""
        nodes = sorted([(n, data) for n, data in graph.nodes(data=True)], 
                       key=lambda x: x[1]['position'])
        
        for i in range(len(nodes)-1):
            current_node, current_data = nodes[i]
            next_node, next_data = nodes[i+1]
            
            # 检查两个相邻节点之间是否存在边
            if not graph.has_edge(current_node, next_node):
                results["errors"].append({
                    "type": "reasoning_gap",
                    "location": f"步骤 {current_data['position']} 和 {next_data['position']} 之间",
                    "description": "推理步骤间缺少明显连接，可能存在逻辑跳跃"
                })
                
            # 检查相邻节点间的边是否具有低置信度
            elif graph[current_node][next_node]['confidence'] < self.confidence_threshold:
                results["errors"].append({
                    "type": "weak_connection",
                    "location": f"步骤 {current_data['position']} 到 {next_data['position']}",
                    "description": f"步骤间连接不强 (置信度: {graph[current_node][next_node]['confidence']:.2f})"
                })
    
    def _detect_circular_reasoning(self, graph: nx.DiGraph, results: Dict) -> None:
        """检测循环依赖"""
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                cycle_steps = [f"步骤 {graph.nodes[n]['position']}" for n in cycle]
                results["errors"].append({
                    "type": "circular_reasoning",
                    "location": " → ".join(cycle_steps),
                    "description": "检测到循环推理，这可能表明逻辑错误"
                })
        except nx.NetworkXNoCycle:
            pass  # 没有循环，这是好事
        except Exception as e:
            # 如果因为其他原因导致循环检测失败，记录错误但继续执行
            results["errors"].append({
                "type": "analysis_error",
                "location": "循环检测过程",
                "description": f"循环分析出错: {str(e)}"
            })
    
    def _detect_isolated_components(self, graph: nx.DiGraph, results: Dict) -> None:
        """检测孤立的子图组件"""
        if not graph.nodes:
            return
            
        components = list(nx.weakly_connected_components(graph))
        if len(components) > 1:
            for i, component in enumerate(components):
                nodes = sorted([n for n in component], 
                              key=lambda x: graph.nodes[x].get('position', 0))
                steps = [f"步骤 {graph.nodes[n]['position']}" for n in nodes]
                results["errors"].append({
                    "type": "isolated_component",
                    "location": ", ".join(steps),
                    "description": f"检测到独立的推理组件 {i+1}，缺少与其他步骤的连接"
                })
    
    def _generate_suggestions(self, results: Dict) -> None:
        """基于检测结果生成改进建议"""
        for error in results.get("errors", []):
            if error["type"] == "reasoning_gap":
                results["suggestions"].append({
                    "for": error["location"],
                    "suggestion": "添加明确的中间步骤或解释这两步之间的逻辑关系"
                })
            elif error["type"] == "weak_connection":
                results["suggestions"].append({
                    "for": error["location"],
                    "suggestion": "加强步骤间的逻辑关联，明确说明如何从前一步推导出当前结论"
                })
            elif error["type"] == "circular_reasoning":
                results["suggestions"].append({
                    "for": error["location"],
                    "suggestion": "重构推理路径，避免循环论证，确保每一步都基于已知信息"
                })
            elif error["type"] == "isolated_component":
                results["suggestions"].append({
                    "for": error["location"],
                    "suggestion": "将独立的推理部分与主要推理链连接起来，或解释它们之间的关系"
                })
        
        for node, score, reason in results.get("low_confidence", []):
            results["suggestions"].append({
                "for": node,
                "suggestion": f"增强此步骤的解释，确保它与其他步骤有明确的逻辑联系 (当前置信度: {score:.2f})"
            })