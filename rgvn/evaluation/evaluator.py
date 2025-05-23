"""
RVGN评估器模块。
该模块负责协调评估过程、运行实验、收集结果并生成报告。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import datetime
from tqdm import tqdm
import yaml
from pathlib import Path

from rgvn.data_processing.data_manager import DataManager
from rgvn.llm.api_client import LLMAPIClient
from rgvn.core.graph_builder import ReasoningGraphBuilder
from rgvn.core.error_detector import GraphErrorDetector
from rgvn.evaluation.metrics import RVGNEvaluator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RVGNExperimentManager:
    """
    RVGN实验管理器，负责运行实验和评估。
    """
    
    def __init__(self, config: Dict = None, output_dir: str = None):
        """
        初始化实验管理器。
        
        Args:
            config: 实验配置字典
            output_dir: 输出目录。如果为None，默认为项目根目录下的'results'
        """
        if output_dir is None:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算项目根目录
            project_root = os.path.abspath(os.path.join(current_dir, '../..'))
            self.output_dir = os.path.join(project_root, 'results')
        else:
            self.output_dir = output_dir
            
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载或设置默认配置
        self.config = config or {}
        self._set_default_config()
        
        # 初始化组件
        self.data_manager = DataManager()
        self.evaluator = RVGNEvaluator()
        
        # 初始化LLM客户端、图构建器和错误检测器
        self._initialize_components()
        
    def _set_default_config(self):
        """设置默认配置。"""
        # 基本配置
        if 'experiment_name' not in self.config:
            self.config['experiment_name'] = f"rvgn_experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # 数据集配置
        if 'datasets' not in self.config:
            self.config['datasets'] = {
                'logiqa': {
                    'splits': ['dev'],  # 使用哪些数据分割
                    'sample_size': 100  # 每个分割使用的样本数量
                }
            }
            
        # LLM配置
        if 'llm' not in self.config:
            self.config['llm'] = {
                'model': 'gpt-4o-mini',
                'max_tokens': 1500,
                'temperature': 0.7
            }
            
        # RVGN配置
        if 'rvgn' not in self.config:
            self.config['rvgn'] = {
                'confidence_threshold': 0.6,  # 错误检测器的置信度阈值
                'enable_critique': True  # 是否使用LLM对RVGN反馈进行优化
            }
            
        # 实验配置
        if 'experiments' not in self.config:
            self.config['experiments'] = {
                'run_baseline': True,  # 运行基线实验
                'run_rvgn': True,  # 运行RVGN实验
                'run_adversarial': True,  # 运行对抗性实验
                'adversarial_types': ['reordering', 'irrelevant_info']  # 对抗性样本类型
            }
        
    def _initialize_components(self):
        """初始化RVGN组件。"""
        # 初始化LLM API客户端
        llm_config = self.config.get('llm', {})
        api_key = llm_config.get('api_key') or os.getenv("OPENAI_API_KEY")
        base_url = llm_config.get('base_url') or os.getenv("OPENAI_API_BASE")
        
        if not api_key:
            logger.warning("未设置OpenAI API密钥，LLM功能将不可用。请通过config或环境变量设置OPENAI_API_KEY")
        
        self.llm_client = LLMAPIClient(
            api_key=api_key,
            base_url=base_url,
            model=llm_config.get('model', 'gpt-4o-mini')
        ) if api_key else None
        
        # 初始化图构建器
        self.graph_builder = ReasoningGraphBuilder()
        
        # 初始化错误检测器
        rvgn_config = self.config.get('rvgn', {})
        self.error_detector = GraphErrorDetector(
            confidence_threshold=rvgn_config.get('confidence_threshold', 0.6)
        )
        
    def run_experiments(self):
        """
        运行所有配置的实验。
        """
        logger.info(f"开始运行实验: {self.config['experiment_name']}")
        
        # 创建实验特定的输出目录
        experiment_dir = os.path.join(self.output_dir, self.config['experiment_name'])
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
        results = {}
        
        # 运行标准评估
        for dataset_name, dataset_config in self.config['datasets'].items():
            logger.info(f"处理数据集: {dataset_name}")
            
            for split in dataset_config.get('splits', ['dev']):
                logger.info(f"处理分割: {split}")
                
                # 基线评估
                if self.config['experiments'].get('run_baseline', True):
                    logger.info("运行基线评估...")
                    baseline_results = self.run_baseline_evaluation(
                        dataset_name, 
                        split, 
                        sample_size=dataset_config.get('sample_size')
                    )
                    
                    results[f"{dataset_name}_{split}_baseline"] = baseline_results
                
                # RVGN评估
                if self.config['experiments'].get('run_rvgn', True):
                    logger.info("运行RVGN评估...")
                    rvgn_results = self.run_rvgn_evaluation(
                        dataset_name,
                        split,
                        sample_size=dataset_config.get('sample_size')
                    )
                    
                    results[f"{dataset_name}_{split}_rvgn"] = rvgn_results
                    
                    # 计算基线与RVGN之间的比较
                    if self.config['experiments'].get('run_baseline', True):
                        comparisons = self.evaluator.compare_methods(
                            baseline_results['task_accuracy'],
                            rvgn_results['task_accuracy']
                        )
                        
                        results[f"{dataset_name}_{split}_comparison"] = comparisons
                
                # 对抗性评估
                if self.config['experiments'].get('run_adversarial', True):
                    logger.info("运行对抗性评估...")
                    for adv_type in self.config['experiments'].get('adversarial_types', ['reordering']):
                        logger.info(f"对抗性类型: {adv_type}")
                        
                        adv_results = self.run_adversarial_evaluation(
                            dataset_name,
                            split,
                            adv_type,
                            sample_size=dataset_config.get('sample_size')
                        )
                        
                        results[f"{dataset_name}_{split}_{adv_type}"] = adv_results
        
        # 生成报告
        self.generate_report(results, experiment_dir)
        
        # 保存完整结果
        with open(os.path.join(experiment_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"实验完成，结果已保存至: {experiment_dir}")
        
        return results
        
    def run_baseline_evaluation(self, dataset: str, split: str, sample_size: int = None) -> Dict[str, Any]:
        """
        运行基线评估，即直接使用LLM输出的推理和答案。
        
        Args:
            dataset: 数据集名称
            split: 数据分割
            sample_size: 要评估的样本数量
            
        Returns:
            评估结果字典
        """
        # 加载带推理的数据集
        data = self.data_manager.load_dataset_with_reasoning(dataset, split)
        
        if not data:
            logger.error(f"无法加载{dataset}数据集的{split}分割（带推理）")
            return {"error": "数据加载失败"}
        
        # 限制样本数量
        if sample_size is not None and sample_size < len(data):
            import random
            random.seed(42)  # 固定随机数，确保实验可重现
            data = random.sample(data, sample_size)
            
        # 提取答案和推理
        ground_truth = [item['answer'] for item in data]
        reasoning_texts = [item['reasoning'] for item in data]
        
        # 从推理文本中提取预测答案
        predictions = []
        for reasoning in reasoning_texts:
            pred = self.evaluator.extract_answer_from_reasoning(reasoning)
            if pred == -1:  # 无法从文本中提取答案
                # 随机选择一个答案，或者更保守地选择常见答案
                pred = 0  # 默认选第一个选项
            predictions.append(pred)
            
        # 评估任务准确率
        task_accuracy = self.evaluator.evaluate_task_accuracy(predictions, ground_truth)
        
        # 评估推理质量
        reasoning_quality = []
        for reasoning in reasoning_texts:
            quality = self.evaluator.evaluate_reasoning_quality(reasoning)
            reasoning_quality.append(quality)
            
        avg_quality = {metric: np.mean([q[metric] for q in reasoning_quality]) 
                      for metric in reasoning_quality[0].keys()}
        
        return {
            "task_accuracy": task_accuracy,
            "reasoning_quality": avg_quality,
            "predictions": predictions,
            "reasoning_texts": reasoning_texts,
            "ground_truth": ground_truth,
            "detailed_quality": reasoning_quality
        }
        
    def run_rvgn_evaluation(self, dataset: str, split: str, sample_size: int = None) -> Dict[str, Any]:
        """
        运行RVGN评估，使用RVGN优化推理并评估结果。
        
        Args:
            dataset: 数据集名称
            split: 数据分割
            sample_size: 要评估的样本数量
            
        Returns:
            评估结果字典
        """
        # 加载带推理的数据集
        data = self.data_manager.load_dataset_with_reasoning(dataset, split)
        
        if not data:
            logger.error(f"无法加载{dataset}数据集的{split}分割（带推理）")
            return {"error": "数据加载失败"}
        
        # 限制样本数量
        if sample_size is not None and sample_size < len(data):
            import random
            random.seed(42)  # 固定随机数，确保实验可重现
            data = random.sample(data, sample_size)
        
        # 提取答案和原始推理
        original_reasoning_texts = [item['reasoning'] for item in data]
        ground_truth = [item['answer'] for item in data]
        
        improved_reasoning_texts = []
        errors_detected = []
        
        # RVGN处理每个推理
        for i, item in enumerate(tqdm(data, desc="RVGN处理")):
            # 构建推理图
            graph = self.graph_builder.build_graph(item['reasoning'])
            
            # 检测错误
            detected_errors = self.error_detector.detect_errors(graph)
            errors_detected.append(detected_errors)
            
            if not self.config['rvgn'].get('enable_critique', True) or not self.llm_client:
                # 如果不启用批判或没有LLM客户端，保留原始推理
                improved_reasoning_texts.append(item['reasoning'])
                continue
            
            # 如果检测到错误，请求LLM改进推理
            if detected_errors['errors'] or detected_errors['low_confidence']:
                problem_description = item['context'] + "\n" + item['question'] + "\n"
                options_text = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(item['options'])])
                problem_description += options_text
                
                improved_reasoning = self.llm_client.critique_and_improve_reasoning(
                    item['reasoning'],
                    problem_description,
                    detected_errors
                )
                improved_reasoning_texts.append(improved_reasoning)
            else:
                # 如果没有检测到错误，保留原始推理
                improved_reasoning_texts.append(item['reasoning'])
                
        # 从原始和改进的推理文本中提取预测答案
        original_predictions = []
        improved_predictions = []
        
        for original, improved in zip(original_reasoning_texts, improved_reasoning_texts):
            # 从原始推理中提取
            orig_pred = self.evaluator.extract_answer_from_reasoning(original)
            if orig_pred == -1:
                orig_pred = 0  # 默认选第一个选项
            original_predictions.append(orig_pred)
            
            # 从改进的推理中提取
            imp_pred = self.evaluator.extract_answer_from_reasoning(improved)
            if imp_pred == -1:
                imp_pred = orig_pred  # 如果无法提取，使用原始预测
            improved_predictions.append(imp_pred)
                
        # 评估任务准确率
        original_accuracy = self.evaluator.evaluate_task_accuracy(original_predictions, ground_truth)
        improved_accuracy = self.evaluator.evaluate_task_accuracy(improved_predictions, ground_truth)
        
        # 评估推理质量
        original_quality = []
        improved_quality = []
        
        for original, improved in zip(original_reasoning_texts, improved_reasoning_texts):
            orig_quality = self.evaluator.evaluate_reasoning_quality(original)
            original_quality.append(orig_quality)
            
            imp_quality = self.evaluator.evaluate_reasoning_quality(improved)
            improved_quality.append(imp_quality)
            
        avg_original_quality = {metric: np.mean([q[metric] for q in original_quality]) 
                               for metric in original_quality[0].keys()}
        avg_improved_quality = {metric: np.mean([q[metric] for q in improved_quality]) 
                               for metric in improved_quality[0].keys()}
        
        # 计算RVGN的错误检测统计信息
        error_stats = {
            "total_samples": len(data),
            "samples_with_errors": sum(1 for e in errors_detected if e['errors'] or e['low_confidence']),
            "error_types": {},
            "low_confidence_nodes_avg": np.mean([len(e.get('low_confidence', [])) for e in errors_detected])
        }
        
        # 统计错误类型
        for error_list in errors_detected:
            for error in error_list.get('errors', []):
                error_type = error['type']
                if error_type in error_stats['error_types']:
                    error_stats['error_types'][error_type] += 1
                else:
                    error_stats['error_types'][error_type] = 1
        
        return {
            "task_accuracy": improved_accuracy,
            "original_accuracy": original_accuracy,
            "reasoning_quality": avg_improved_quality,
            "original_quality": avg_original_quality,
            "predictions": improved_predictions,
            "original_predictions": original_predictions,
            "reasoning_texts": improved_reasoning_texts,
            "original_reasoning_texts": original_reasoning_texts,
            "ground_truth": ground_truth,
            "error_detection_stats": error_stats,
            "detected_errors": errors_detected
        }
        
    def run_adversarial_evaluation(self, dataset: str, split: str, adv_type: str = 'all', sample_size: int = None) -> Dict[str, Any]:
        """
        运行对抗性评估，测试模型在干扰样本上的鲁棒性。
        
        Args:
            dataset: 数据集名称
            split: 数据分割
            adv_type: 对抗性样本类型
            sample_size: 要评估的样本数量
            
        Returns:
            评估结果字典
        """
        # 加载原始数据集
        original_data = self.data_manager.load_dataset_with_reasoning(dataset, split)
        
        if not original_data:
            logger.error(f"无法加载{dataset}数据集的{split}分割（带推理）")
            return {"error": "数据加载失败"}
            
        # 限制样本数量
        if sample_size is not None and sample_size < len(original_data):
            import random
            random.seed(42)  # 固定随机数，确保实验可重现
            original_data = random.sample(original_data, sample_size)
            
        # 生成对抗性样本
        self.data_manager.generate_adversarial_samples(dataset, split, adv_type, ratio=1.0)
        
        # 加载对抗性样本
        adversarial_data = self.data_manager.load_adversarial_dataset(dataset, split, adv_type)
        
        if not adversarial_data:
            logger.error(f"无法加载或生成对抗性样本")
            return {"error": "对抗性样本生成失败"}
            
        # 提取原始答案和推理
        original_ground_truth = [item['answer'] for item in original_data]
        original_reasoning_texts = [item['reasoning'] for item in original_data]
        
        # 从原始推理文本中提取预测答案
        original_predictions = []
        for reasoning in original_reasoning_texts:
            pred = self.evaluator.extract_answer_from_reasoning(reasoning)
            if pred == -1:
                pred = 0  # 默认选第一个选项
            original_predictions.append(pred)
            
        # 使用RVGN处理对抗性样本
        # 为对抗性样本生成推理 (如果它们没有推理)
        adversarial_reasoning_texts = []
        
        if self.llm_client and all('reasoning' not in item for item in adversarial_data[:5]):
            logger.info("为对抗性样本生成推理...")
            
            for i, item in enumerate(tqdm(adversarial_data, desc="生成对抗性推理")):
                problem_description = item['context'] + "\n" + item['question'] + "\n"
                options_text = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(item['options'])])
                problem_description += options_text
                
                reasoning = self.llm_client.generate_reasoning(problem_description)
                adversarial_reasoning_texts.append(reasoning)
                adversarial_data[i]['reasoning'] = reasoning
        else:
            for item in adversarial_data:
                adversarial_reasoning_texts.append(item.get('reasoning', ''))
        
        # 提取对抗性样本的预测
        adversarial_predictions = []
        for reasoning in adversarial_reasoning_texts:
            pred = self.evaluator.extract_answer_from_reasoning(reasoning)
            if pred == -1:
                pred = 0
            adversarial_predictions.append(pred)
        
        # 应用RVGN改进对抗性推理
        improved_adversarial_reasoning = []
        
        if self.config['rvgn'].get('enable_critique', True) and self.llm_client:
            logger.info("使用RVGN改进对抗性推理...")
            
            for i, item in enumerate(tqdm(adversarial_data, desc="RVGN处理对抗性样本")):
                # 构建推理图
                graph = self.graph_builder.build_graph(item['reasoning'])
                
                # 检测错误
                detected_errors = self.error_detector.detect_errors(graph)
                
                # 如果检测到错误，请求LLM改进推理
                if detected_errors['errors'] or detected_errors['low_confidence']:
                    problem_description = item['context'] + "\n" + item['question'] + "\n"
                    options_text = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(item['options'])])
                    problem_description += options_text
                    
                    improved_reasoning = self.llm_client.critique_and_improve_reasoning(
                        item['reasoning'],
                        problem_description,
                        detected_errors
                    )
                    improved_adversarial_reasoning.append(improved_reasoning)
                else:
                    # 如果没有检测到错误，保留原始推理
                    improved_adversarial_reasoning.append(item['reasoning'])
        else:
            improved_adversarial_reasoning = adversarial_reasoning_texts
            
        # 从改进的对抗性推理中提取预测
        improved_adversarial_predictions = []
        for reasoning in improved_adversarial_reasoning:
            pred = self.evaluator.extract_answer_from_reasoning(reasoning)
            if pred == -1:
                pred = 0
            improved_adversarial_predictions.append(pred)
            
        # 评估鲁棒性 (原始 vs 对抗性)
        robustness = self.evaluator.evaluate_robustness(
            original_predictions,
            adversarial_predictions,
            original_reasoning_texts,
            adversarial_reasoning_texts
        )
        
        # 评估RVGN改进后的鲁棒性 (原始 vs 改进的对抗性)
        improved_robustness = self.evaluator.evaluate_robustness(
            original_predictions,
            improved_adversarial_predictions,
            original_reasoning_texts,
            improved_adversarial_reasoning
        )
        
        return {
            "original_predictions": original_predictions,
            "adversarial_predictions": adversarial_predictions,
            "improved_adversarial_predictions": improved_adversarial_predictions,
            "ground_truth": original_ground_truth,
            "robustness": robustness,
            "improved_robustness": improved_robustness,
            "adversarial_type": adv_type,
            "original_reasoning": original_reasoning_texts[:10],  # 只保存前10个样本的推理，避免结果文件过大
            "adversarial_reasoning": adversarial_reasoning_texts[:10],
            "improved_adversarial_reasoning": improved_adversarial_reasoning[:10]
        }
        
    def generate_report(self, results: Dict[str, Any], output_dir: str):
        """
        生成实验报告，包括表格和可视化。
        
        Args:
            results: 评估结果字典
            output_dir: 输出目录
        """
        logger.info("生成实验报告...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 任务准确率比较表格
        accuracy_data = []
        
        for key, value in results.items():
            if 'task_accuracy' in value:
                dataset_split = key.rsplit('_', 1)[0]  # 去掉最后一个下划线之后的内容
                method = key.rsplit('_', 1)[1]  # 获取最后一个下划线之后的内容
                
                accuracy_data.append({
                    'Dataset_Split': dataset_split,
                    'Method': method,
                    'Accuracy': value['task_accuracy'].get('accuracy', 0),
                    'Macro_F1': value['task_accuracy'].get('macro_f1', 0),
                    'Micro_F1': value['task_accuracy'].get('micro_f1', 0)
                })
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            accuracy_table_path = os.path.join(output_dir, 'accuracy_comparison.csv')
            accuracy_df.to_csv(accuracy_table_path, index=False)
            
            # 绘制准确率比较图
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Dataset_Split', y='Accuracy', hue='Method', data=accuracy_df)
            plt.title('Task Accuracy Comparison')
            plt.xlabel('Dataset and Split')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
            plt.close()
            
        # 2. 推理质量比较表格
        quality_data = []
        
        for key, value in results.items():
            if 'reasoning_quality' in value:
                dataset_split = key.rsplit('_', 1)[0]
                method = key.rsplit('_', 1)[1]
                
                quality_data.append({
                    'Dataset_Split': dataset_split,
                    'Method': method,
                    'Overall_Quality': value['reasoning_quality'].get('overall', 0),
                    'Structure': value['reasoning_quality'].get('structure', 0),
                    'Coherence': value['reasoning_quality'].get('coherence', 0),
                    'Completeness': value['reasoning_quality'].get('completeness', 0),
                    'Precision': value['reasoning_quality'].get('precision', 0),
                    'Consistency': value['reasoning_quality'].get('consistency', 0)
                })
                
                # 如果有原始质量数据，也添加
                if 'original_quality' in value:
                    quality_data.append({
                        'Dataset_Split': dataset_split,
                        'Method': method + '_original',
                        'Overall_Quality': value['original_quality'].get('overall', 0),
                        'Structure': value['original_quality'].get('structure', 0),
                        'Coherence': value['original_quality'].get('coherence', 0),
                        'Completeness': value['original_quality'].get('completeness', 0),
                        'Precision': value['original_quality'].get('precision', 0),
                        'Consistency': value['original_quality'].get('consistency', 0)
                    })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            quality_table_path = os.path.join(output_dir, 'quality_comparison.csv')
            quality_df.to_csv(quality_table_path, index=False)
            
            # 绘制质量比较图
            plt.figure(figsize=(12, 8))
            
            # 绘制综合质量评分
            plt.subplot(2, 1, 1)
            sns.barplot(x='Dataset_Split', y='Overall_Quality', hue='Method', data=quality_df)
            plt.title('Overall Reasoning Quality Comparison')
            plt.xlabel('Dataset and Split')
            plt.ylabel('Overall Quality Score')
            plt.xticks(rotation=45)
            
            # 绘制子维度质量
            plt.subplot(2, 1, 2)
            quality_metrics = quality_df.melt(
                id_vars=['Dataset_Split', 'Method'], 
                value_vars=['Structure', 'Coherence', 'Completeness', 'Precision', 'Consistency'],
                var_name='Quality_Metric', value_name='Score'
            )
            
            # 只使用一个数据集作为示例
            example_dataset = quality_metrics['Dataset_Split'].unique()[0]
            example_data = quality_metrics[quality_metrics['Dataset_Split'] == example_dataset]
            
            sns.barplot(x='Quality_Metric', y='Score', hue='Method', data=example_data)
            plt.title(f'Quality Metrics Breakdown for {example_dataset}')
            plt.xlabel('Quality Dimension')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quality_comparison.png'), dpi=300)
            plt.close()
            
        # 3. 鲁棒性比较表格
        robustness_data = []
        
        for key, value in results.items():
            if 'robustness' in value:
                dataset_split_adv = key
                
                robustness_data.append({
                    'Dataset_Config': dataset_split_adv,
                    'Method': 'baseline',
                    'Prediction_Consistency': value['robustness'].get('prediction_consistency', 0),
                    'Reasoning_Similarity': value['robustness'].get('avg_reasoning_similarity', 0),
                    'Contradictory_Rate': value['robustness'].get('contradictory_reasoning_rate', 0),
                    'Robustness_Score': value['robustness'].get('robustness_score', 0)
                })
                
                if 'improved_robustness' in value:
                    robustness_data.append({
                        'Dataset_Config': dataset_split_adv,
                        'Method': 'rvgn',
                        'Prediction_Consistency': value['improved_robustness'].get('prediction_consistency', 0),
                        'Reasoning_Similarity': value['improved_robustness'].get('avg_reasoning_similarity', 0),
                        'Contradictory_Rate': value['improved_robustness'].get('contradictory_reasoning_rate', 0),
                        'Robustness_Score': value['improved_robustness'].get('robustness_score', 0)
                    })
        
        if robustness_data:
            robustness_df = pd.DataFrame(robustness_data)
            robustness_table_path = os.path.join(output_dir, 'robustness_comparison.csv')
            robustness_df.to_csv(robustness_table_path, index=False)
            
            # 绘制鲁棒性比较图
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Dataset_Config', y='Robustness_Score', hue='Method', data=robustness_df)
            plt.title('Robustness Score Comparison')
            plt.xlabel('Dataset and Configuration')
            plt.ylabel('Robustness Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_comparison.png'), dpi=300)
            plt.close()
        
        # 4. 错误检测统计表格
        error_stats_data = []
        
        for key, value in results.items():
            if 'error_detection_stats' in value:
                dataset_split = key.rsplit('_', 1)[0]
                
                stats = value['error_detection_stats']
                error_stats_data.append({
                    'Dataset_Split': dataset_split,
                    'Total_Samples': stats.get('total_samples', 0),
                    'Samples_With_Errors': stats.get('samples_with_errors', 0),
                    'Error_Rate': stats.get('samples_with_errors', 0) / stats.get('total_samples', 1),
                    'Low_Confidence_Nodes_Avg': stats.get('low_confidence_nodes_avg', 0)
                })
                
                # 添加各种错误类型的统计
                for error_type, count in stats.get('error_types', {}).items():
                    if not any(d.get('Error_Type') == error_type for d in error_stats_data):
                        error_stats_data.append({
                            'Dataset_Split': 'Error_Type_Count',
                            'Error_Type': error_type,
                            'Count': count
                        })
        
        if error_stats_data:
            # 分成两个表格
            error_stats_general = [d for d in error_stats_data if 'Error_Type' not in d]
            error_stats_types = [d for d in error_stats_data if 'Error_Type' in d]
            
            if error_stats_general:
                error_stats_general_df = pd.DataFrame(error_stats_general)
                error_stats_general_path = os.path.join(output_dir, 'error_detection_stats.csv')
                error_stats_general_df.to_csv(error_stats_general_path, index=False)
            
            if error_stats_types:
                error_stats_types_df = pd.DataFrame(error_stats_types)
                error_stats_types_path = os.path.join(output_dir, 'error_types_stats.csv')
                error_stats_types_df.to_csv(error_stats_types_path, index=False)
                
                # 绘制错误类型统计图
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Error_Type', y='Count', data=error_stats_types_df)
                plt.title('Error Type Distribution')
                plt.xlabel('Error Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'error_types.png'), dpi=300)
                plt.close()
                
        # 生成摘要报告
        summary_path = os.path.join(output_dir, 'summary_report.md')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# RVGN Experiment Summary Report\n\n")
            f.write(f"Experiment: {self.config['experiment_name']}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Experiment Configuration\n\n")
            f.write(f"- Datasets: {', '.join(self.config['datasets'].keys())}\n")
            f.write(f"- LLM Model: {self.config['llm'].get('model', 'N/A')}\n")
            f.write(f"- RVGN Confidence Threshold: {self.config['rvgn'].get('confidence_threshold', 0.6)}\n\n")
            
            if accuracy_data:
                f.write(f"## Task Accuracy Results\n\n")
                f.write("| Dataset_Split | Method | Accuracy | Macro_F1 | Micro_F1 |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                
                for row in accuracy_data:
                    f.write(f"| {row['Dataset_Split']} | {row['Method']} | {row['Accuracy']:.4f} | {row['Macro_F1']:.4f} | {row['Micro_F1']:.4f} |\n")
                
                f.write("\n![Accuracy Comparison](accuracy_comparison.png)\n\n")
                
            if quality_data:
                f.write(f"## Reasoning Quality Results\n\n")
                f.write("| Dataset_Split | Method | Overall_Quality |\n")
                f.write("| --- | --- | --- |\n")
                
                for row in quality_data:
                    f.write(f"| {row['Dataset_Split']} | {row['Method']} | {row['Overall_Quality']:.4f} |\n")
                
                f.write("\n![Quality Comparison](quality_comparison.png)\n\n")
                
            if robustness_data:
                f.write(f"## Robustness Results\n\n")
                f.write("| Dataset_Config | Method | Prediction_Consistency | Robustness_Score |\n")
                f.write("| --- | --- | --- | --- |\n")
                
                for row in robustness_data:
                    f.write(f"| {row['Dataset_Config']} | {row['Method']} | {row['Prediction_Consistency']:.4f} | {row['Robustness_Score']:.4f} |\n")
                
                f.write("\n![Robustness Comparison](robustness_comparison.png)\n\n")
            
            f.write(f"## Conclusion\n\n")
            f.write("Comprehensive analysis results can be found in the CSV files and visualizations in this directory.\n")
            
        logger.info(f"实验报告已保存至: {summary_path}")
        return summary_path

# 用于从配置文件加载实验配置
def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载实验配置。
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

# 示例用法
if __name__ == "__main__":
    # 示例配置
    example_config = {
        'experiment_name': 'logic_puzzle_experiment',
        'datasets': {
            'logiqa': {
                'splits': ['dev'],
                'sample_size': 20  # 小样本用于示例
            }
        },
        'llm': {
            'model': 'gpt-4o-mini',
            'max_tokens': 1500,
            'temperature': 0.7
            # 注意: 需要设置 API 密钥环境变量 OPENAI_API_KEY
        },
        'rvgn': {
            'confidence_threshold': 0.6,
            'enable_critique': True
        },
        'experiments': {
            'run_baseline': True,
            'run_rvgn': True,
            'run_adversarial': True,
            'adversarial_types': ['reordering']
        }
    }
    
    # 通过环境变量或参数设置API密钥
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        # 初始化并运行实验
        experiment_manager = RVGNExperimentManager(example_config)
        results = experiment_manager.run_experiments()
        print("实验完成。")
    else:
        print("警告：未设置 OPENAI_API_KEY 环境变量，无法运行完整实验。")
        print("请设置环境变量后重试。")
