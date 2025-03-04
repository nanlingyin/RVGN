import re
from typing import List, Dict, Any

def segment_reasoning(text: str) -> List[Dict[str, Any]]:
    """
    将推理文本分割为独立的推理步骤段落
    
    参数:
        text: 完整推理文本
        
    返回:
        包含各段落信息的字典列表
    """
    # 匹配形如"步骤1"、"第1步"等格式
    step_pattern = r"(?:步骤|第)\s*(\d+)(?:步)?[：:\.]\s*(.*?)(?=(?:步骤|第)\s*\d+(?:步)?[：:\.] | $)"
    matches = re.finditer(step_pattern, text, re.DOTALL)
    
    segments = []
    for match in matches:
        step_num = int(match.group(1))
        content = match.group(2).strip()
        segments.append({
            "id": step_num,
            "type": "step",
            "content": content,
            "start": match.start(),
            "end": match.end()
        })
    
    # 如果没有明确的步骤标记，尝试按段落分割
    if not segments:
        paragraphs = text.split("\n\n")
        segments = [
            {
                "id": i+1,
                "type": "paragraph",
                "content": p.strip(),
                "start": text.find(p),
                "end": text.find(p) + len(p)
            }
            for i, p in enumerate(paragraphs) if p.strip()
        ]
    
    return segments

def extract_key_concepts(text: str) -> List[str]:
    """
    从文本中提取关键概念
    
    参数:
        text: 输入文本
        
    返回:
        关键概念列表
    """
    # 提取数学符号
    math_symbols = re.findall(r'[=><+\-*/∫∂∑∏√]', text)
    
    # 提取数字
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    
    # 提取可能的变量和函数
    variables = re.findall(r'\b[a-zA-Z](?:\([^)]*\))?\b', text)
    
    # 提取可能的重要术语 (通常是2-3个词组成的专业术语)
    terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', text)
    
    # 合并所有概念并去除重复
    concepts = list(set(math_symbols + numbers + variables + terms))
    return [c for c in concepts if c]