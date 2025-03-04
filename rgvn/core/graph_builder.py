import networkx as nx
from typing import Dict, List, Any, Optional
try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

from rgvn.utils.text_processing import segment_reasoning, extract_key_concepts

class ReasoningGraphBuilder:
    """推理图构建器 - 将文本推理转换为图结构"""
    
    def __init__(self):
        """初始化推理图构建器"""
        self.nlp = None
        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                pass
        
    def build_graph(self, reasoning_text: str) -> nx.DiGraph:
        """
        从推理文本构建推理图
        
        参数:
            reasoning_text: 包含推理步骤的文本
            
        返回:
            表示推理结构的有向图
        """
        # 创建有向图
        graph = nx.DiGraph()
        
        # 分段推理文本
        segments = segment_reasoning(reasoning_text)
        
        # 为每个段落创建节点
        for segment in segments:
            node_id = f"step_{segment['id']}"
            concepts = extract_key_concepts(segment['content'])
            
            # 添加节点及其属性
            graph.add_node(
                node_id,
                content=segment['content'],
                type=segment['type'],
                concepts=concepts,
                confidence=1.0,  # 初始置信度
                position=segment['id']
            )
        
        # 创建边连接相邻步骤
        for i in range(len(segments)-1):
            current_id = f"step_{segments[i]['id']}"
            next_id = f"step_{segments[i+1]['id']}"
            
            # 计算概念重叠度作为边权重
            current_concepts = set(graph.nodes[current_id]['concepts'])
            next_concepts = set(graph.nodes[next_id]['concepts'])
            overlap = len(current_concepts.intersection(next_concepts))
            total = len(current_concepts.union(next_concepts)) if current_concepts or next_concepts else 1
            weight = overlap / total if total > 0 else 0
            
            graph.add_edge(
                current_id, 
                next_id, 
                weight=weight,
                type="sequential",
                confidence=weight  # 边的初始置信度基于概念重叠
            )
        
        # 推断节点间的非顺序依赖关系
        self._infer_dependencies(graph)
        
        # 计算节点和边的置信度分数
        self._compute_confidence_scores(graph)
        
        return graph
    
    def _infer_dependencies(self, graph: nx.DiGraph) -> None:
        """推断非顺序步骤间的依赖关系"""
        nodes = list(graph.nodes(data=True))
        
        for i, (node_i, data_i) in enumerate(nodes):
            for j, (node_j, data_j) in enumerate(nodes):
                if i >= j:  # 避免自环和重复
                    continue
                
                # 检查是否存在共享概念但没有直接连接
                concepts_i = set(data_i['concepts'])
                concepts_j = set(data_j['concepts'])
                
                # 如果有显著概念重叠但没有直接连接，添加推断边
                overlap = len(concepts_i.intersection(concepts_j))
                total = len(concepts_i.union(concepts_j)) if concepts_i or concepts_j else 1
                
                if overlap > 0 and not graph.has_edge(node_i, node_j) and not graph.has_edge(node_j, node_i):
                    # 检查哪个节点更可能是依赖源
                    if data_i['position'] < data_j['position']:  # 假设较早的步骤是依赖源
                        similarity = overlap / total if total > 0 else 0
                        if similarity > 0.3:  # 只在相似度较高时添加边
                            graph.add_edge(
                                node_i, node_j,
                                weight=similarity,
                                type="inferred",
                                confidence=0.6 * similarity  # 推断边的置信度较低
                            )
    
    def _compute_confidence_scores(self, graph: nx.DiGraph) -> None:
        """计算节点和边的置信度分数"""
        # 基于连接模式计算节点置信度
        for node in graph.nodes:
            # 计算入度和出度
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            # 孤立节点降低置信度
            if in_degree == 0 and out_degree == 0:
                graph.nodes[node]['confidence'] *= 0.5
            
            # 对于中间节点，确保有合理的连接
            position = graph.nodes[node]['position']
            max_position = max(data['position'] for _, data in graph.nodes(data=True))
            
            if 1 < position < max_position:  # 中间节点
                if in_degree == 0:  # 中间节点没有入边
                    graph.nodes[node]['confidence'] *= 0.7
                if out_degree == 0:  # 中间节点没有出边
                    graph.nodes[node]['confidence'] *= 0.7