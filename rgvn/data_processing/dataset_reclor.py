"""
ReClor数据集处理模块.
该模块处理ReClor逻辑推理理解数据集的下载、解析和预处理.
"""

import os
import json
import requests
import pandas as pd
import zipfile
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReClorProcessor:
    """
    ReClor数据集处理器.
    负责数据集的下载、解析和预处理.
    """
    
    # ReClor数据集的GitHub页面
    RECLOR_URL = "https://github.com/yuweihao/reclor"
    
    def __init__(self, data_dir: str = None):
        """
        初始化ReClor处理器.
        
        Args:
            data_dir: 数据目录路径. 如果为None，默认使用项目根目录下的'data/reclor'.
        """
        if data_dir is None:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算项目根目录 (假设项目结构是rgvn/data_processing/dataset_reclor.py)
            project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
            self.data_dir = os.path.join(project_root, 'data', 'reclor')
        else:
            self.data_dir = data_dir
            
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 设置原始和处理后数据的路径
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # 存放训练集、验证集和测试集的路径
        self.train_path = os.path.join(self.processed_dir, 'train.json')
        self.val_path = os.path.join(self.processed_dir, 'val.json')
        self.test_path = os.path.join(self.processed_dir, 'test.json')
        
    def download(self, force: bool = False) -> bool:
        """
        从GitHub指导下载ReClor数据集.
        
        Args:
            force: 如果为True，即使文件已存在也重新下载.
            
        Returns:
            下载是否成功.
        """
        # 检查是否已经下载过数据集
        if os.path.exists(os.path.join(self.raw_dir, 'reclor_data')) and not force:
            logger.info("ReClor数据集已存在，跳过下载。")
            return True
        
        logger.info(f"请从 {self.RECLOR_URL} 下载ReClor数据集...")
        logger.info("由于ReClor数据集需要申请访问权限，请按照以下步骤操作:")
        logger.info("1. 访问ReClor GitHub页面")
        logger.info("2. 按照指示填写数据访问表单")
        logger.info("3. 下载数据后，将其解压到: %s", self.raw_dir)
        logger.info("4. 确保下载的文件包含train.json, val.json和test.json")
        
        # ReClor需要手动申请和下载，无法自动下载
        return False
            
    def extract(self) -> bool:
        """
        检查ReClor数据集文件.
        
        Returns:
            检查是否成功.
        """
        # ReClor通常已经是JSON格式，不需要特别解压
        required_files = [
            os.path.join(self.raw_dir, 'train.json'),
            os.path.join(self.raw_dir, 'val.json'),
            os.path.join(self.raw_dir, 'test.json')
        ]
        
        if not all(os.path.exists(f) for f in required_files):
            logger.error(f"未找到所有必需的ReClor数据文件")
            logger.error(f"请确保以下文件存在: {required_files}")
            return False
        
        logger.info("ReClor数据文件检查完成.")
        return True
            
    def preprocess(self) -> bool:
        """
        预处理ReClor数据集，将其转换为标准格式.
        
        Returns:
            预处理是否成功.
        """
        # 检查原始数据文件是否存在
        train_file = os.path.join(self.raw_dir, 'train.json')
        val_file = os.path.join(self.raw_dir, 'val.json')
        test_file = os.path.join(self.raw_dir, 'test.json')
        
        if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            logger.error("原始数据文件不存在，请确保已正确下载数据集。")
            return False
        
        # 处理每个数据集
        try:
            logger.info("处理训练集...")
            train_data = self._parse_dataset_file(train_file)
            
            logger.info("处理验证集...")
            val_data = self._parse_dataset_file(val_file)
            
            logger.info("处理测试集...")
            test_data = self._parse_dataset_file(test_file)
            
            # 保存处理后的数据
            with open(self.train_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
                
            with open(self.val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)
                
            with open(self.test_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
                
            logger.info("ReClor数据集预处理完成.")
            logger.info(f"训练集: {len(train_data)} 个样本")
            logger.info(f"验证集: {len(val_data)} 个样本")
            logger.info(f"测试集: {len(test_data)} 个样本")
            
            return True
            
        except Exception as e:
            logger.error(f"预处理ReClor数据集时出错: {e}")
            return False
    
    def _parse_dataset_file(self, file_path: str) -> List[Dict]:
        """
        解析ReClor数据集文件.
        
        Args:
            file_path: 数据文件路径.
            
        Returns:
            解析后的数据列表.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        data = []
        
        # ReClor数据格式转换
        for i, item in enumerate(raw_data):
            processed_item = {
                'id': item.get('id_string', f"reclor_{i}"),
                'context': item.get('context', ''),
                'question': item.get('question', ''),
                'options': item.get('answers', []),
                'answer': item.get('label', -1),  # ReClor中label是答案索引
                'reasoning': None  # 初始化为None，后续由LLM生成
            }
            
            data.append(processed_item)
        
        return data
    
    def load_data(self, split: str = 'train') -> List[Dict]:
        """
        加载预处理后的ReClor数据.
        
        Args:
            split: 数据集分割，可选 'train', 'val', 或 'test'.
            
        Returns:
            数据列表.
        """
        if split == 'train':
            path = self.train_path
        elif split == 'val':
            path = self.val_path
        elif split == 'test':
            path = self.test_path
        else:
            raise ValueError(f"无效的数据集分割: {split}. 可选 'train', 'val', 或 'test'.")
        
        if not os.path.exists(path):
            logger.error(f"数据文件不存在: {path}. 请先下载并预处理数据集.")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def save_reasoning(self, data: List[Dict], split: str = 'train') -> bool:
        """
        保存带有LLM生成的推理的数据.
        
        Args:
            data: 包含推理的数据列表.
            split: 数据集分割，可选 'train', 'val', 或 'test'.
            
        Returns:
            保存是否成功.
        """
        if split == 'train':
            path = self.train_path.replace('.json', '_with_reasoning.json')
        elif split == 'val':
            path = self.val_path.replace('.json', '_with_reasoning.json')
        elif split == 'test':
            path = self.test_path.replace('.json', '_with_reasoning.json')
        else:
            raise ValueError(f"无效的数据集分割: {split}. 可选 'train', 'val', 或 'test'.")
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"带推理的数据已保存至: {path}")
            return True
        except Exception as e:
            logger.error(f"保存推理数据时出错: {e}")
            return False

# 示例用法
if __name__ == "__main__":
    processor = ReClorProcessor()
    
    # 提示用户手动下载数据集
    processor.download()
    
    # 检查数据文件
    if processor.extract():
        # 预处理数据集
        processor.preprocess()
        
        # 加载训练集
        train_data = processor.load_data('train')
        print(f"加载了 {len(train_data)} 个训练样本")
        if train_data:
            print("样本示例:")
            print(json.dumps(train_data[0], ensure_ascii=False, indent=2))
    else:
        print("请按照提示下载ReClor数据集并放入指定目录后继续")
