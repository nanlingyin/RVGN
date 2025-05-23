"""
LogiQA数据集处理模块.
该模块处理LogiQA逻辑谜题数据集的下载、解析和预处理.
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

class LogiQAProcessor:
    """
    LogiQA数据集处理器.
    负责数据集的下载、解析和预处理.
    """
    
    # LogiQA数据集的默认下载URL
    LOGIQA_URL = "https://github.com/lgw863/LogiQA-dataset"
    
    def __init__(self, data_dir: str = None):
        """
        初始化LogiQA处理器.
        
        Args:
            data_dir: 数据目录路径. 如果为None，默认使用项目根目录下的'data/logiqa'.
        """
        if data_dir is None:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算项目根目录 (假设项目结构是rgvn/data_processing/dataset_logiqa.py)
            project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
            self.data_dir = os.path.join(project_root, 'data', 'logiqa')
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
        self.dev_path = os.path.join(self.processed_dir, 'dev.json')
        self.test_path = os.path.join(self.processed_dir, 'test.json')
        
    def download(self, force: bool = False) -> bool:
        """
        从GitHub下载LogiQA数据集.
        
        Args:
            force: 如果为True，即使文件已存在也重新下载.
            
        Returns:
            下载是否成功.
        """
        # 检查是否已经下载过数据集
        if os.path.exists(os.path.join(self.raw_dir, 'logiqa_data.zip')) and not force:
            logger.info("LogiQA数据集已存在，跳过下载。")
            return True
        
        logger.info(f"开始从 {self.LOGIQA_URL} 下载LogiQA数据集...")
        logger.info("注意: LogiQA数据集需要手动下载，请访问项目页面获取数据。")
        logger.info("下载后，请将数据放在 %s 目录下", self.raw_dir)
        
        # 由于LogiQA数据集需要手动下载，我们这里只提供指导
        return False
            
    def extract(self) -> bool:
        """
        解压LogiQA数据集.
        
        Returns:
            解压是否成功.
        """
        zip_path = os.path.join(self.raw_dir, 'logiqa_data.zip')
        
        if not os.path.exists(zip_path):
            logger.error(f"未找到LogiQA数据集压缩文件: {zip_path}")
            return False
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                logger.info("解压LogiQA数据集...")
                zip_ref.extractall(self.raw_dir)
            logger.info("解压完成.")
            return True
        except Exception as e:
            logger.error(f"解压LogiQA数据集时出错: {e}")
            return False
            
    def preprocess(self) -> bool:
        """
        预处理LogiQA数据集，将其转换为标准JSON格式.
        
        Returns:
            预处理是否成功.
        """
        # 检查原始数据文件是否存在
        train_file = os.path.join(self.raw_dir, 'Train.txt')
        dev_file = os.path.join(self.raw_dir, 'Eval.txt')
        test_file = os.path.join(self.raw_dir, 'Test.txt')
        
        if not all(os.path.exists(f) for f in [train_file, dev_file, test_file]):
            logger.error("原始数据文件不存在, 请确保已正确解压数据集.")
            return False
        
        # 处理每个数据集
        try:
            logger.info("处理训练集...")
            train_data = self._parse_dataset_file(train_file)
            
            logger.info("处理验证集...")
            dev_data = self._parse_dataset_file(dev_file)
            
            logger.info("处理测试集...")
            test_data = self._parse_dataset_file(test_file)
            
            # 保存处理后的数据
            with open(self.train_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
                
            with open(self.dev_path, 'w', encoding='utf-8') as f:
                json.dump(dev_data, f, ensure_ascii=False, indent=2)
                
            with open(self.test_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
                
            logger.info("LogiQA数据集预处理完成.")
            logger.info(f"训练集: {len(train_data)} 个样本")
            logger.info(f"验证集: {len(dev_data)} 个样本")
            logger.info(f"测试集: {len(test_data)} 个样本")
            
            return True
            
        except Exception as e:
            logger.error(f"预处理LogiQA数据集时出错: {e}")
            return False
    
    def _parse_dataset_file(self, file_path: str) -> List[Dict]:
        """
        解析LogiQA数据集文件.
        
        Args:
            file_path: 数据文件路径.
            
        Returns:
            解析后的数据列表.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        i = 0
        
        # LogiQA数据格式解析
        while i < len(lines):
            try:
                # 每组数据解析
                qid = int(lines[i].strip())
                i += 1
                
                # 上下文
                context = lines[i].strip()
                i += 1
                
                # 问题
                question = lines[i].strip()
                i += 1
                
                # 选项
                options = []
                for _ in range(4):  # LogiQA有4个选项
                    options.append(lines[i].strip())
                    i += 1
                
                # 正确答案
                answer = int(lines[i].strip())
                i += 1
                
                # 构建数据项
                item = {
                    'id': qid,
                    'context': context,
                    'question': question,
                    'options': options,
                    'answer': answer,
                    'reasoning': None  # 初始化为None，后续由LLM生成
                }
                
                data.append(item)
                
            except Exception as e:
                logger.warning(f"解析第{len(data)}个样本时出错: {e}. 跳过该样本。")
                # 尝试恢复到下一个样本
                i += 1
        
        return data
    
    def load_data(self, split: str = 'train') -> List[Dict]:
        """
        加载预处理后的LogiQA数据.
        
        Args:
            split: 数据集分割，可选 'train', 'dev', 或 'test'.
            
        Returns:
            数据列表.
        """
        if split == 'train':
            path = self.train_path
        elif split == 'dev':
            path = self.dev_path
        elif split == 'test':
            path = self.test_path
        else:
            raise ValueError(f"无效的数据集分割: {split}. 可选 'train', 'dev', 或 'test'.")
        
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
            split: 数据集分割，可选 'train', 'dev', 或 'test'.
            
        Returns:
            保存是否成功.
        """
        if split == 'train':
            path = self.train_path.replace('.json', '_with_reasoning.json')
        elif split == 'dev':
            path = self.dev_path.replace('.json', '_with_reasoning.json')
        elif split == 'test':
            path = self.test_path.replace('.json', '_with_reasoning.json')
        else:
            raise ValueError(f"无效的数据集分割: {split}. 可选 'train', 'dev', 或 'test'.")
        
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
    processor = LogiQAProcessor()
    
    # 尝试下载数据集
    if not processor.download():
        print("请手动下载LogiQA数据集，放入指定目录后继续")
    
    # 解压数据集
    processor.extract()
    
    # 预处理数据集
    processor.preprocess()
    
    # 加载训练集
    train_data = processor.load_data('train')
    print(f"加载了 {len(train_data)} 个训练样本")
    if train_data:
        print("样本示例:")
        print(json.dumps(train_data[0], ensure_ascii=False, indent=2))
