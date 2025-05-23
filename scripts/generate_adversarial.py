#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成对抗性样本的脚本。
这些样本用于测试推理模型对各种扰动的鲁棒性。
"""

import os
import json
import random
import logging
import argparse
import sys
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(ROOT_DIR)

class AdversarialSampleGenerator:
    """生成各种类型的对抗性样本"""
    
    def __init__(self, output_dir: str):
        """
        初始化生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_samples(self, source_data: List[Dict], 
                        perturbation_type: str,
                        sample_size: int = None) -> List[Dict]:
        """
        生成对抗性样本
        
        Args:
            source_data: 源数据列表
            perturbation_type: 扰动类型 ('irrelevant_info', 'negation', 'reordering', 'all')
            sample_size: 要生成的样本数量
            
        Returns:
            生成的对抗性样本列表
        """
        if sample_size is None:
            sample_size = len(source_data)
        else:
            sample_size = min(sample_size, len(source_data))
        
        # 随机选择样本
        selected_samples = random.sample(source_data, sample_size)
        
        # 根据扰动类型选择生成函数
        if perturbation_type == 'irrelevant_info':
            return self._add_irrelevant_information(selected_samples)
        elif perturbation_type == 'negation':
            return self._apply_negation(selected_samples)
        elif perturbation_type == 'reordering':
            return self._reorder_context(selected_samples)
        elif perturbation_type == 'all':
            # 对每个样本随机应用一种扰动
            adversarial_samples = []
            for sample in selected_samples:
                method = random.choice(['irrelevant_info', 'negation', 'reordering'])
                if method == 'irrelevant_info':
                    adversarial_samples.extend(self._add_irrelevant_information([sample]))
                elif method == 'negation':
                    adversarial_samples.extend(self._apply_negation([sample]))
                else:
                    adversarial_samples.extend(self._reorder_context([sample]))
            return adversarial_samples
        else:
            logger.error(f"未知的扰动类型: {perturbation_type}")
            return []
    
    def _add_irrelevant_information(self, samples: List[Dict]) -> List[Dict]:
        """添加不相关信息"""
        irrelevant_statements = [
            "苹果是红色的水果。",
            "太阳从东方升起。",
            "水在100摄氏度沸腾。",
            "地球绕太阳运行。",
            "猫是哺乳动物。",
            "书籍通常由纸张制成。",
            "大多数鸟类会飞行。",
            "人类有206块骨头。",
            "1加1等于2。",
            "一年有12个月。",
            "Apples are red fruits.",
            "The sun rises in the east.",
            "Water boils at 100 degrees Celsius.",
            "The Earth orbits around the sun.",
            "Cats are mammals.",
            "Books are usually made of paper.",
            "Most birds can fly.",
            "Humans have 206 bones.",
            "One plus one equals two.",
            "There are 12 months in a year."
        ]
        
        adversarial_samples = []
        for sample in samples:
            # 创建样本的深拷贝
            new_sample = sample.copy()
            
            # 选择1-3个不相关陈述
            statements = random.sample(irrelevant_statements, random.randint(1, 3))
            
            # 将不相关信息添加到上下文中
            context = new_sample.get('context', '')
            new_context = context + " " + " ".join(statements)
            new_sample['context'] = new_context
            
            # 添加扰动标记
            new_sample['perturbation'] = 'irrelevant_info'
            new_sample['original_id'] = sample.get('id', '')
            new_sample['id'] = f"adv_{new_sample['original_id']}_irr"
            
            adversarial_samples.append(new_sample)
            
        return adversarial_samples
    
    def _apply_negation(self, samples: List[Dict]) -> List[Dict]:
        """应用逻辑否定"""
        adversarial_samples = []
        
        # 常见的否定词对
        negation_pairs = [
            ('所有', '并非所有'), ('都', '不都'), ('总是', '并不总是'),
            ('必须', '不必'), ('肯定', '不一定'), ('可能', '不可能'),
            ('每个', '不是每个'), ('全部', '不全部'), ('没有', '有一些'),
            ('all', 'not all'), ('always', 'not always'), ('must', 'need not'),
            ('definitely', 'not necessarily'), ('possible', 'impossible'),
            ('every', 'not every'), ('none', 'some'), ('no', 'some')
        ]
        
        for sample in samples:
            # 创建样本的深拷贝
            new_sample = sample.copy()
            
            # 获取上下文
            context = new_sample.get('context', '')
            
            # 尝试应用否定
            applied = False
            for pos, neg in negation_pairs:
                if pos in context and random.random() > 0.5:
                    context = context.replace(pos, neg, 1)  # 只替换一次
                    applied = True
                    break
            
            # 如果没有找到可替换的词，随机选一个陈述添加否定前缀
            if not applied:
                sentences = context.split('.')
                if len(sentences) > 1:
                    idx = random.randint(0, len(sentences) - 2)
                    if sentences[idx].strip():
                        neg_prefix = random.choice(['事实上，并非', '实际上，不是', 'In fact, it is not true that'])
                        sentences[idx] = neg_prefix + sentences[idx].strip().lower()
                        context = '.'.join(sentences)
                        applied = True
            
            new_sample['context'] = context
            
            # 添加扰动标记
            new_sample['perturbation'] = 'negation'
            new_sample['original_id'] = sample.get('id', '')
            new_sample['id'] = f"adv_{new_sample['original_id']}_neg"
            new_sample['applied'] = applied
            
            adversarial_samples.append(new_sample)
            
        return adversarial_samples
    
    def _reorder_context(self, samples: List[Dict]) -> List[Dict]:
        """重新排序上下文中的句子"""
        adversarial_samples = []
        
        for sample in samples:
            # 创建样本的深拷贝
            new_sample = sample.copy()
            
            # 获取上下文
            context = new_sample.get('context', '')
            
            # 按句子分割
            sentences = [s.strip() + '.' for s in context.split('.') if s.strip()]
            
            if len(sentences) <= 1:
                # 句子太少，无法重排序
                new_sample['perturbation'] = 'reordering_failed'
                new_sample['original_id'] = sample.get('id', '')
                new_sample['id'] = f"adv_{new_sample['original_id']}_reord"
                adversarial_samples.append(new_sample)
                continue
                
            # 随机打乱句子顺序
            random.shuffle(sentences)
            new_context = ' '.join(sentences)
            new_sample['context'] = new_context
            
            # 添加扰动标记
            new_sample['perturbation'] = 'reordering'
            new_sample['original_id'] = sample.get('id', '')
            new_sample['id'] = f"adv_{new_sample['original_id']}_reord"
            
            adversarial_samples.append(new_sample)
            
        return adversarial_samples
    
    def save_samples(self, samples: List[Dict], output_file: str):
        """保存生成的样本到文件"""
        output_path = os.path.join(self.output_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'data': samples}, f, ensure_ascii=False, indent=4)
            
        logger.info(f"保存了 {len(samples)} 个对抗性样本到 {output_path}")

def load_dataset(dataset_name, split):
    """加载数据集"""
    dataset_path = os.path.join(ROOT_DIR, 'data', dataset_name, f"{split}.json")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('data', [])
    except FileNotFoundError:
        logger.error(f"数据集文件不存在: {dataset_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"数据集文件格式错误: {dataset_path}")
        return []

def main():
    parser = argparse.ArgumentParser(description='生成对抗性样本')
    parser.add_argument('--dataset', type=str, default='logiqa', 
                        choices=['logiqa', 'reclor'], 
                        help='源数据集名称')
    parser.add_argument('--split', type=str, default='dev',
                        help='数据集分割 (train, dev/val, test)')
    parser.add_argument('--perturbation', type=str, default='all', 
                        choices=['irrelevant_info', 'negation', 'reordering', 'all'],
                        help='扰动类型')
    parser.add_argument('--samples', type=int, default=30,
                        help='要生成的样本数量')
    args = parser.parse_args()
    
    # 加载数据集
    logger.info(f"加载 {args.dataset} 数据集的 {args.split} 分割...")
    
    # 调整ReClor的split名称
    if args.dataset == 'reclor' and args.split == 'dev':
        split = 'val'  # ReClor使用'val'而非'dev'
    else:
        split = args.split
        
    source_data = load_dataset(args.dataset, split)
    
    if not source_data:
        logger.error("未能加载数据集，请确保数据集已下载并放置在正确位置")
        return
        
    logger.info(f"加载了 {len(source_data)} 个样本")
    
    # 初始化生成器
    output_dir = os.path.join(ROOT_DIR, 'data', 'adversarial')
    generator = AdversarialSampleGenerator(output_dir)
    
    # 生成对抗性样本
    logger.info(f"使用 '{args.perturbation}' 扰动生成对抗性样本...")
    adversarial_samples = generator.generate_samples(
        source_data, 
        args.perturbation,
        args.samples
    )
    
    # 保存样本
    output_file = f"{args.dataset}_{split}_{args.perturbation}.json"
    generator.save_samples(adversarial_samples, output_file)
    
    logger.info("完成!")

if __name__ == '__main__':
    main()
