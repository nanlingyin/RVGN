#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载和准备 LogiQA 和 ReClor 数据集的脚本。
"""

import os
import requests
import zipfile
import tarfile
import argparse
import logging
import shutil
from pathlib import Path
import json
import sys
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(ROOT_DIR)

def download_file(url, dest_path, chunk_size=8192):
    """
    下载文件，显示进度条
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest_path

def download_logiqa():
    """
    下载 LogiQA 数据集
    
    LogiQA 数据集来源：https://github.com/lgw863/LogiQA-dataset
    """
    logger.info("开始下载 LogiQA 数据集...")
    
    logiqa_dir = os.path.join(ROOT_DIR, 'data', 'logiqa')
    os.makedirs(logiqa_dir, exist_ok=True)
    
    # LogiQA GitHub 存储库URL
    logiqa_url = "https://github.com/lgw863/LogiQA-dataset/archive/master.zip"
    zip_path = os.path.join(logiqa_dir, 'logiqa.zip')
    
    try:
        # 下载数据集
        logger.info(f"从 {logiqa_url} 下载数据...")
        download_file(logiqa_url, zip_path)
        
        # 解压数据集
        logger.info("解压数据集...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(logiqa_dir)
            
        # 移动文件到合适的位置
        extracted_dir = os.path.join(logiqa_dir, 'LogiQA-dataset-master')
        for filename in os.listdir(extracted_dir):
            src_path = os.path.join(extracted_dir, filename)
            dst_path = os.path.join(logiqa_dir, filename)
            shutil.move(src_path, dst_path)
            
        # 清理临时文件
        os.remove(zip_path)
        shutil.rmtree(extracted_dir, ignore_errors=True)
        
        logger.info(f"LogiQA 数据集已成功下载到 {logiqa_dir}")
        return True
        
    except Exception as e:
        logger.error(f"下载 LogiQA 数据集出错: {e}")
        logger.info("请参考以下指南手动下载 LogiQA 数据集:")
        logger.info("1. 访问 https://github.com/lgw863/LogiQA-dataset")
        logger.info("2. 下载 ZIP 文件并解压到 'data/logiqa' 目录")
        return False

def download_reclor():
    """
    下载 ReClor 数据集
    
    ReClor 数据集需要在官方网站注册后才能下载：
    https://whyu.me/reclor/
    """
    logger.info("开始准备 ReClor 数据集...")
    
    reclor_dir = os.path.join(ROOT_DIR, 'data', 'reclor')
    os.makedirs(reclor_dir, exist_ok=True)
    
    # 由于 ReClor 需要注册，我们提供说明而不是直接下载
    logger.info("ReClor 数据集需要手动下载，请按照以下步骤操作：")
    logger.info("1. 访问 https://whyu.me/reclor/ 并注册账号")
    logger.info("2. 登录后下载数据集")
    logger.info(f"3. 解压下载的文件，并将内容放置到 {reclor_dir} 目录")
    
    # 创建样例数据，用于开发目的
    logger.info("为了开发目的，创建一个小型样例数据集...")
    create_reclor_samples(reclor_dir)
    
    return False

def create_reclor_samples(reclor_dir):
    """
    创建 ReClor 样例数据，用于开发目的
    """
    sample_data = {
        "train": [
            {
                "id": "train_0",
                "context": "All residents of green county speak both English and French. No residents in blue county speak French.",
                "question": "Which of the following statements must be true?",
                "answers": [0],
                "options": [
                    "No residents of blue county live in green county.",
                    "All residents in blue county speak English only.",
                    "Some residents of green county live in blue county.",
                    "Some residents of green county do not speak English."
                ]
            },
            {
                "id": "train_1",
                "context": "All birds have feathers. All birds lay eggs. No mammals have feathers.",
                "question": "Which of the following conclusions can be drawn?",
                "answers": [2],
                "options": [
                    "No mammals lay eggs.",
                    "Some mammals lay eggs.",
                    "No mammals are birds.",
                    "Some birds are mammals."
                ]
            }
        ],
        "val": [
            {
                "id": "val_0",
                "context": "All employees who work on weekends receive overtime pay. Some employees who receive overtime pay are managers.",
                "question": "Which of the following can be logically concluded?",
                "answers": [1],
                "options": [
                    "All managers work on weekends.",
                    "Some managers work on weekends.",
                    "No managers work on weekends.",
                    "Most managers do not work on weekends."
                ]
            }
        ],
        "test": [
            {
                "id": "test_0",
                "context": "All doctors are professionals. Some professionals are wealthy. No doctors are politicians.",
                "question": "Which of the following conclusions is valid?",
                "answers": [3],
                "options": [
                    "All professionals are doctors.",
                    "All wealthy people are doctors.",
                    "No professionals are politicians.",
                    "Some professionals are not politicians."
                ]
            }
        ]
    }
    
    # 保存样例数据
    for split, samples in sample_data.items():
        out_path = os.path.join(reclor_dir, f"{split}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({"data": samples}, f, ensure_ascii=False, indent=4)
            
    logger.info(f"已创建 ReClor 样例数据: {len(sample_data['train'])} 训练, {len(sample_data['val'])} 验证, {len(sample_data['test'])} 测试")

def main():
    parser = argparse.ArgumentParser(description='下载并准备 LogiQA 和 ReClor 数据集')
    parser.add_argument('--dataset', type=str, choices=['logiqa', 'reclor', 'all'], 
                        default='all', help='指定要下载的数据集')
    args = parser.parse_args()
    
    if args.dataset == 'logiqa' or args.dataset == 'all':
        download_logiqa()
        
    if args.dataset == 'reclor' or args.dataset == 'all':
        download_reclor()
    
    logger.info("数据集准备完成.")

if __name__ == '__main__':
    main()
