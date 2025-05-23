"""
数据集处理和生成模块的主接口.
为了简化数据集管理，此模块提供了统一的接口来处理不同的数据集.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import tqdm
from pathlib import Path

# 导入各个数据集处理器
from rgvn.data_processing.dataset_logiqa import LogiQAProcessor
from rgvn.data_processing.dataset_reclor import ReClorProcessor
from rgvn.data_processing.dataset_adversarial import AdversarialLogicProcessor
from rgvn.llm.api_client import LLMAPIClient

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    """
    数据管理器，负责处理和生成各种数据集.
    """
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据管理器.
        
        Args:
            data_dir: 数据根目录. 如果为None，默认使用项目根目录下的'data'.
        """
        if data_dir is None:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算项目根目录
            project_root = os.path.abspath(os.path.join(current_dir, '../..'))
            self.data_dir = os.path.join(project_root, 'data')
        else:
            self.data_dir = data_dir
            
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化各数据集处理器
        self.logiqa_processor = LogiQAProcessor(os.path.join(self.data_dir, 'logiqa'))
        self.reclor_processor = ReClorProcessor(os.path.join(self.data_dir, 'reclor'))
        self.adversarial_processor = AdversarialLogicProcessor(os.path.join(self.data_dir, 'adversarial'))
        
    def download_datasets(self):
        """
        下载所有数据集.
        注意：某些数据集可能需要手动下载.
        """
        logger.info("开始下载数据集...")
        
        # 下载LogiQA数据集
        logger.info("下载LogiQA数据集...")
        if not self.logiqa_processor.download():
            logger.warning("LogiQA数据集需要手动下载")
            
        # 下载ReClor数据集
        logger.info("下载ReClor数据集...")
        if not self.reclor_processor.download():
            logger.warning("ReClor数据集需要手动下载")
            
        logger.info("数据集下载/指导完成")
        
    def prepare_datasets(self):
        """
        预处理所有数据集.
        """
        logger.info("开始预处理数据集...")
        
        # 处理LogiQA
        logger.info("处理LogiQA数据集...")
        self.logiqa_processor.extract()
        self.logiqa_processor.preprocess()
        
        # 处理ReClor
        logger.info("处理ReClor数据集...")
        self.reclor_processor.extract()
        self.reclor_processor.preprocess()
        
        logger.info("数据集预处理完成")
        
    def generate_reasoning(self, dataset: str, split: str, llm_client: LLMAPIClient, 
                          sample_size: int = None, prompt_template: str = None):
        """
        使用LLM为数据集生成推理.
        
        Args:
            dataset: 数据集名称，可选 'logiqa', 'reclor'.
            split: 数据分割，可选 'train', 'dev'/'val', 'test'.
            llm_client: LLMAPIClient实例，用于生成推理.
            sample_size: 可选，要生成的样本数量。如果为None，则处理整个数据集.
            prompt_template: 可选，用于生成推理的自定义模板.
        """
        logger.info(f"为 {dataset} 数据集 ({split} 分割) 生成推理...")
        
        # 加载数据
        if dataset.lower() == 'logiqa':
            data = self.logiqa_processor.load_data(split)
        elif dataset.lower() == 'reclor':
            # ReClor命名约定: 'val' 而不是 'dev'
            if split == 'dev':
                split = 'val'
            data = self.reclor_processor.load_data(split)
        else:
            logger.error(f"不支持的数据集: {dataset}")
            return
            
        if not data:
            logger.error(f"未加载到 {dataset} 的 {split} 数据")
            return
            
        # 如果指定了样本大小，随机选择子集
        if sample_size is not None and sample_size < len(data):
            import random
            data = random.sample(data, sample_size)
            
        # 默认提示模板
        if prompt_template is None:
            prompt_template = (
                "以下是一个逻辑推理问题。首先，请理解问题和选项。然后，一步一步详细地解释推理过程，说明为什么正确答案是正确的，"
                "以及为什么其他选项是错误的。请清晰地标记每一步推理。\n\n"
                "上下文: {context}\n"
                "问题: {question}\n"
                "选项:\n{options_text}\n\n"
                "请提供详细的推理过程:"
            )
            
        # 处理每个样本
        for i, item in enumerate(tqdm.tqdm(data, desc="生成推理")):
            # 准备选项文本
            options_text = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(item['options'])])
            
            # 格式化提示
            problem_description = prompt_template.format(
                context=item['context'],
                question=item['question'],
                options_text=options_text
            )
            
            # 生成推理
            reasoning = llm_client.generate_reasoning(problem_description)
            
            # 将推理添加到数据项
            data[i]['reasoning'] = reasoning
            
            # 每生成10个样本，保存一次
            if (i + 1) % 10 == 0:
                logger.info(f"已完成 {i+1}/{len(data)} 个样本")
                
                # 保存当前进度
                if dataset.lower() == 'logiqa':
                    self.logiqa_processor.save_reasoning(data, split)
                else:
                    self.reclor_processor.save_reasoning(data, split)
        
        # 最终保存
        if dataset.lower() == 'logiqa':
            self.logiqa_processor.save_reasoning(data, split)
        else:
            self.reclor_processor.save_reasoning(data, split)
            
        logger.info(f"为 {len(data)} 个样本生成推理完成")
        
    def generate_adversarial_samples(self, source_dataset: str, source_split: str, 
                                   perturbation_type: str = 'all', ratio: float = 0.3):
        """
        生成对抗性样本.
        
        Args:
            source_dataset: 源数据集名称，可选 'logiqa', 'reclor'.
            source_split: 源数据分割，可选 'train', 'dev'/'val', 'test'.
            perturbation_type: 扰动类型，可选 'irrelevant_info', 'negation', 'reordering', 'all'.
            ratio: 生成对抗性样本的比例，相对于原始数据集.
        """
        logger.info(f"基于 {source_dataset} 数据集 ({source_split}) 生成对抗性样本...")
        
        # 加载源数据
        if source_dataset.lower() == 'logiqa':
            source_data = self.logiqa_processor.load_data(source_split)
        elif source_dataset.lower() == 'reclor':
            # ReClor命名约定: 'val' 而不是 'dev'
            if source_split == 'dev':
                source_split = 'val'
            source_data = self.reclor_processor.load_data(source_split)
        else:
            logger.error(f"不支持的数据集: {source_dataset}")
            return
            
        if not source_data:
            logger.error(f"未加载到 {source_dataset} 的 {source_split} 数据")
            return
        
        # 生成对抗性样本
        adversarial_samples = self.adversarial_processor.generate_adversarial_samples(
            source_data, 
            perturbation_type=perturbation_type, 
            ratio=ratio
        )
        
        # 保存对抗性样本
        if adversarial_samples:
            filename = f"{source_dataset}_{source_split}_{perturbation_type}"
            self.adversarial_processor.save_adversarial_data(adversarial_samples, filename)
            
        logger.info(f"生成了 {len(adversarial_samples)} 个对抗性样本")
        
    def load_dataset_with_reasoning(self, dataset: str, split: str) -> List[Dict]:
        """
        加载带有推理的数据集.
        
        Args:
            dataset: 数据集名称，可选 'logiqa', 'reclor'
            split: 数据分割，可选 'train', 'dev'/'val', 'test'
            
        Returns:
            带有推理的数据列表
        """
        # 构建文件路径
        if dataset.lower() == 'logiqa':
            if split not in ['train', 'dev', 'test']:
                logger.error(f"LogiQA不支持的分割: {split}")
                return []
                
            path = os.path.join(self.data_dir, 'logiqa', 'processed', f"{split}_with_reasoning.json")
            
        elif dataset.lower() == 'reclor':
            if split == 'dev':
                split = 'val'
                
            if split not in ['train', 'val', 'test']:
                logger.error(f"ReClor不支持的分割: {split}")
                return []
                
            path = os.path.join(self.data_dir, 'reclor', 'processed', f"{split}_with_reasoning.json")
            
        else:
            logger.error(f"不支持的数据集: {dataset}")
            return []
        
        # 检查文件是否存在
        if not os.path.exists(path):
            logger.error(f"带推理的数据文件不存在: {path}")
            return []
        
        # 加载数据
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"从 {path} 加载了 {len(data)} 个带推理的样本")
            return data
        except Exception as e:
            logger.error(f"加载带推理的数据时出错: {e}")
            return []
            
    def load_adversarial_dataset(self, base_dataset: str, split: str, 
                               perturbation_type: str = 'all') -> List[Dict]:
        """
        加载对抗性数据集.
        
        Args:
            base_dataset: 基础数据集名称，可选 'logiqa', 'reclor'
            split: 数据分割，可选 'train', 'dev'/'val', 'test'
            perturbation_type: 扰动类型，可选 'irrelevant_info', 'negation', 'reordering', 'all'
            
        Returns:
            对抗性样本列表
        """
        # 格式化分割名
        if base_dataset.lower() == 'reclor' and split == 'dev':
            split = 'val'
            
        filename = f"{base_dataset}_{split}_{perturbation_type}"
        return self.adversarial_processor.load_adversarial_data(filename)

# 示例用法
if __name__ == "__main__":
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 下载和准备数据集
    # 注意：这些操作可能需要根据命令行参数控制
    data_manager.download_datasets()
    data_manager.prepare_datasets()
    
    # 示例：生成推理 (需要LLM API访问)
    import os
    api_key = os.getenv("OPENAI_API_KEY")  # 假设已设置环境变量
    
    if api_key:
        from rgvn.llm.api_client import LLMAPIClient
        llm_client = LLMAPIClient(api_key=api_key, model="gpt-4o-mini")
        
        # 为LogiQA开发集生成推理 (示例只生成5个样本)
        data_manager.generate_reasoning('logiqa', 'dev', llm_client, sample_size=5)
        
    # 示例：生成对抗性样本
    data_manager.generate_adversarial_samples('logiqa', 'dev', perturbation_type='all', ratio=0.2)
    
    # 示例：加载带推理的数据
    data_with_reasoning = data_manager.load_dataset_with_reasoning('logiqa', 'dev')
    if data_with_reasoning:
        print(f"加载了 {len(data_with_reasoning)} 个带推理的样本")
        
    # 示例：加载对抗性数据
    adversarial_data = data_manager.load_adversarial_dataset('logiqa', 'dev', 'all')
    if adversarial_data:
        print(f"加载了 {len(adversarial_data)} 个对抗性样本")
