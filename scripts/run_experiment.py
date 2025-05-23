#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RVGN主实验运行脚本。
此脚本执行完整的实验流程，包括数据处理、模型推理、RVGN分析和结果评估。
"""

import os
import sys
import json
import yaml
import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(ROOT_DIR)

from rgvn.evaluation.evaluator import RVGNExperimentManager
from rgvn.llm.api_client import LLMAPIClient
from rgvn.data_processing.data_manager import DataManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """加载实验配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件出错: {e}")
        return {}

def prepare_data(data_manager: DataManager, config: Dict) -> bool:
    """
    准备实验所需的数据
    
    Args:
        data_manager: 数据管理器实例
        config: 实验配置
        
    Returns:
        成功标志
    """
    logger.info("开始准备数据...")
    
    try:
        # 下载数据集
        data_manager.download_datasets()
        
        # 预处理数据集
        data_manager.prepare_datasets()
        
        # 生成对抗性样本
        if config.get('experiments', {}).get('run_adversarial', False):
            adv_config = config.get('datasets', {}).get('adversarial', {})
            source_dataset = adv_config.get('source_dataset', 'logiqa')
            source_split = adv_config.get('source_split', 'dev')
            perturbation_types = adv_config.get('perturbation_types', ['irrelevant_info', 'negation', 'reordering'])
            
            for pert_type in perturbation_types:
                data_manager.adversarial_processor.generate(
                    source_dataset=source_dataset,
                    source_split=source_split,
                    perturbation_type=pert_type,
                    sample_size=adv_config.get('sample_size', 30)
                )
                
        return True
    except Exception as e:
        logger.error(f"准备数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def setup_llm_client(config: Dict) -> Optional[LLMAPIClient]:
    """
    设置LLM客户端
    
    Args:
        config: 实验配置
        
    Returns:
        LLM客户端实例
    """
    llm_config = config.get('llm', {})
    model = llm_config.get('model', 'gpt-4o-mini')
    
    # 获取API密钥，首先从环境变量，然后尝试配置
    api_key_env = llm_config.get('api_key_env', 'OPENAI_API_KEY')
    api_key = os.environ.get(api_key_env)
    
    if not api_key:
        logger.error(f"未找到API密钥。请设置 {api_key_env} 环境变量")
        return None
        
    try:
        return LLMAPIClient(
            api_key=api_key,
            model=model
        )
    except Exception as e:
        logger.error(f"初始化LLM客户端出错: {e}")
        return None

def run_experiment(config_path: str, api_key: str = None):
    """
    运行实验
    
    Args:
        config_path: 配置文件路径
        api_key: OpenAI API密钥（可选，如果未设置环境变量）
    """
    # 加载配置
    config = load_config(config_path)
    if not config:
        logger.error("加载配置失败，退出")
        return
    
    # 如果提供了API密钥，设置环境变量
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    
    # 创建实验时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = config.get('experiment_name', 'rvgn_experiment')
    experiment_id = f"{experiment_name}_{timestamp}"
    
    # 设置输出目录
    output_dir = os.path.join(ROOT_DIR, config.get('output', {}).get('results_dir', 'results'), experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置副本
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    # 初始化数据管理器
    data_manager = DataManager()
    
    # 准备数据
    if not prepare_data(data_manager, config):
        logger.error("数据准备失败，退出")
        return
    
    # 设置LLM客户端
    llm_client = setup_llm_client(config)
    if not llm_client:
        logger.error("LLM客户端设置失败，退出")
        return
    
    # 初始化实验管理器
    experiment_manager = RVGNExperimentManager(config=config, output_dir=output_dir)
    
    # 运行实验
    logger.info(f"开始执行实验: {experiment_id}")
    
    # 设置数据集和分割
    datasets_config = config.get('datasets', {})
    
    # 运行基线实验
    if config.get('experiments', {}).get('run_baseline', True):
        logger.info("运行基线实验...")
        for dataset_name, dataset_config in datasets_config.items():
            # 跳过对抗性数据集配置
            if dataset_name == 'adversarial':
                continue
                
            splits = dataset_config.get('splits', ['dev'])
            for split in splits:
                sample_size = dataset_config.get('sample_size', {}).get(split, 50)
                
                logger.info(f"运行 {dataset_name} 数据集 {split} 分割的基线实验 (样本数: {sample_size})...")
                experiment_manager.run_baseline_experiment(
                    dataset=dataset_name,
                    split=split,
                    sample_size=sample_size,
                    llm_client=llm_client
                )
    
    # 运行RVGN实验
    if config.get('experiments', {}).get('run_rvgn', True):
        logger.info("运行RVGN实验...")
        for dataset_name, dataset_config in datasets_config.items():
            # 跳过对抗性数据集配置
            if dataset_name == 'adversarial':
                continue
                
            splits = dataset_config.get('splits', ['dev'])
            for split in splits:
                sample_size = dataset_config.get('sample_size', {}).get(split, 50)
                
                logger.info(f"运行 {dataset_name} 数据集 {split} 分割的RVGN实验 (样本数: {sample_size})...")
                experiment_manager.run_rvgn_experiment(
                    dataset=dataset_name,
                    split=split,
                    sample_size=sample_size,
                    llm_client=llm_client
                )
    
    # 运行对抗性实验
    if config.get('experiments', {}).get('run_adversarial', True):
        logger.info("运行对抗性实验...")
        
        adv_config = datasets_config.get('adversarial', {})
        source_dataset = adv_config.get('source_dataset', 'logiqa')
        source_split = adv_config.get('source_split', 'dev')
        perturbation_types = adv_config.get('perturbation_types', ['irrelevant_info', 'negation', 'reordering'])
        
        for pert_type in perturbation_types:
            logger.info(f"运行扰动类型 '{pert_type}' 的对抗性实验...")
            
            # 基线对抗性实验
            experiment_manager.run_adversarial_baseline(
                source_dataset=source_dataset,
                perturbation_type=pert_type,
                llm_client=llm_client
            )
            
            # RVGN对抗性实验
            experiment_manager.run_adversarial_rvgn(
                source_dataset=source_dataset,
                perturbation_type=pert_type,
                llm_client=llm_client
            )
    
    # 生成比较报告
    logger.info("生成比较报告...")
    experiment_manager.generate_comparison_report()
    
    logger.info(f"实验 {experiment_id} 完成! 结果保存在 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='运行RVGN实验')
    parser.add_argument('--config', type=str, default='experiments/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API密钥（可选，如果未设置环境变量）')
    args = parser.parse_args()
    
    # 获取配置文件的绝对路径
    config_path = os.path.join(ROOT_DIR, args.config) if not os.path.isabs(args.config) else args.config
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        return
        
    run_experiment(config_path, args.api_key)

if __name__ == '__main__':
    main()
