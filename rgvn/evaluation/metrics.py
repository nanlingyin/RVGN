"""
RVGN的评估指标模块.
此模块实现了三种主要评估指标:
1. 任务准确率 (Task Accuracy)
2. 推理过程质量评分 (Reasoning Process Quality Score)
3. 鲁棒性与一致性 (Robustness and Consistency)

这些指标可以用于评估RVGN在改善LLM推理能力方面的效果.
"""

import os
import json
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, f1_score
import logging
import re
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import string
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查并下载NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class RVGNEvaluator:
    """
    RVGN评估器类，用于计算和比较各种指标.
    """
    
    def __init__(self):
        """初始化评估器."""
        # 初始化Rouge评分器
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # BLEU评分的平滑函数
        self.bleu_smoothing = SmoothingFunction().method1
        
    def evaluate_task_accuracy(self, predictions: List[int], ground_truth: List[int]) -> Dict[str, float]:
        """
        计算任务准确率，即模型在逻辑谜题上给出正确答案的百分比.
        
        Args:
            predictions: 预测的答案索引列表
            ground_truth: 正确答案索引列表
            
        Returns:
            包含准确率和F1分数的字典
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(f"预测列表长度 ({len(predictions)}) 与真实答案列表长度 ({len(ground_truth)}) 不匹配")
            
        accuracy = accuracy_score(ground_truth, predictions)
        
        # 计算宏平均和微平均F1分数
        # 对于多类分类问题，这提供了更全面的评估
        macro_f1 = f1_score(ground_truth, predictions, average='macro')
        micro_f1 = f1_score(ground_truth, predictions, average='micro')
        
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1
        }
    
    def extract_answer_from_reasoning(self, reasoning: str) -> int:
        """
        从推理文本中提取答案索引.
        
        Args:
            reasoning: LLM生成的推理文本
            
        Returns:
            提取的答案索引 (0-3) 或 -1 (如果无法提取)
        """
        # 寻找答案模式
        patterns = [
            r"答案[是为]:?\s*?([1234])",  # 中文模式: 答案是: 2
            r"答案[是为][选项]?\s*?([ABCD])",  # 中文模式: 答案是选项B
            r"正确答案[是为]:?\s*?([1234])",  # 中文模式: 正确答案是: 3
            r"正确答案[是为][选项]?\s*?([ABCD])",  # 中文模式: 正确答案是选项C
            r"选项\s*?([1234])\s*?是正确的",  # 中文模式: 选项2是正确的
            r"选项\s*?([ABCD])\s*?是正确的",  # 中文模式: 选项B是正确的
            r"answer\s*?is\s*?([1234])",  # 英文模式: The answer is 2
            r"answer\s*?is\s*?([ABCD])",  # 英文模式: The answer is B
            r"correct\s*?answer\s*?is\s*?([1234])",  # 英文模式: The correct answer is 3
            r"correct\s*?answer\s*?is\s*?([ABCD])",  # 英文模式: The correct answer is C
            r"option\s*?([1234])\s*?is\s*?correct",  # 英文模式: Option 4 is correct
            r"option\s*?([ABCD])\s*?is\s*?correct",  # 英文模式: Option D is correct
            r"([1234])\s*?is\s*?the\s*?(correct)?\s*?answer",  # 英文模式: 3 is the correct answer
            r"([ABCD])\s*?is\s*?the\s*?(correct)?\s*?answer",  # 英文模式: C is the answer
        ]
        
        # 尝试各种模式
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            if matches:
                answer = matches[-1]  # 使用最后一次出现的答案，假设它是最终结论
                if isinstance(answer, tuple):  # 某些正则表达式会返回元组
                    answer = answer[0]
                    
                # 将字母转换为数字 (A->0, B->1, ...)
                if answer in "ABCD":
                    answer_idx = ord(answer) - ord('A')
                else:
                    answer_idx = int(answer) - 1
                    
                return answer_idx
                
        # 如果没有找到明确的答案，尝试通过频率分析
        option_counts = {
            0: len(re.findall(r"选项1|选项A|option 1|option A|第一个选项|first option", reasoning, re.IGNORECASE)),
            1: len(re.findall(r"选项2|选项B|option 2|option B|第二个选项|second option", reasoning, re.IGNORECASE)),
            2: len(re.findall(r"选项3|选项C|option 3|option C|第三个选项|third option", reasoning, re.IGNORECASE)),
            3: len(re.findall(r"选项4|选项D|option 4|option D|第四个选项|fourth option", reasoning, re.IGNORECASE)),
        }
        
        # 寻找正确/错误关键词与选项的共现
        for idx in range(4):
            option_patterns = [
                f"选项{idx+1}.*?(正确|是对的|是真的)",
                f"选项[ABCD][{idx}].*?(正确|是对的|是真的)",
                f"option {idx+1}.*?(correct|true|right)",
                f"option [ABCD][{idx}].*?(correct|true|right)"
            ]
            
            for pat in option_patterns:
                option_counts[idx] += 3 * len(re.findall(pat, reasoning, re.IGNORECASE))
                
            # 负面模式应减少可能性
            negative_patterns = [
                f"选项{idx+1}.*?(错误|不对|不正确|是假的)",
                f"选项[ABCD][{idx}].*?(错误|不对|不正确|是假的)",
                f"option {idx+1}.*?(incorrect|false|wrong)",
                f"option [ABCD][{idx}].*?(incorrect|false|wrong)"
            ]
            
            for pat in negative_patterns:
                option_counts[idx] -= 2 * len(re.findall(pat, reasoning, re.IGNORECASE))
        
        # 如果有明显的最高分选项
        max_count = max(option_counts.values())
        if max_count > 0:
            for idx, count in option_counts.items():
                if count == max_count:
                    return idx
        
        return -1  # 无法提取答案
        
    def evaluate_reasoning_quality(self, reasoning_text: str) -> Dict[str, float]:
        """
        自动评估推理过程的质量.
        这是RPQS (Reasoning Process Quality Score)的自动化实现.
        
        Args:
            reasoning_text: 要评估的推理文本
            
        Returns:
            包含各维度评分的字典
        """
        scores = {}
        
        # 1. 结构性 - 检查是否有清晰的步骤结构
        step_patterns = [
            r"步骤\s*\d+",  # 中文: 步骤1
            r"第\s*\d+\s*步",  # 中文: 第1步
            r"step\s*\d+",  # 英文: Step 1
        ]
        
        has_steps = False
        step_count = 0
        
        for pattern in step_patterns:
            steps = re.findall(pattern, reasoning_text, re.IGNORECASE)
            step_count = max(step_count, len(steps))
            if steps:
                has_steps = True
                
        # 根据步骤结构评分
        if has_steps and step_count >= 3:
            scores["structure"] = min(1.0, step_count / 5)  # 最多5分
        elif has_steps:
            scores["structure"] = 0.6  # 有结构但步骤较少
        else:
            # 检查是否有段落结构
            paragraphs = reasoning_text.strip().split("\n\n")
            if len(paragraphs) >= 3:
                scores["structure"] = 0.5
            else:
                scores["structure"] = 0.3
        
        # 2. 逻辑连贯性 - 检查逻辑连接词的使用
        logic_terms = [
            r"因为|所以|如果|那么|否则|但是|然而|因此|由于|故而",  # 中文
            r"because|therefore|if|then|else|but|however|thus|since|hence",  # 英文
        ]
        
        logic_term_count = 0
        for pattern in logic_terms:
            terms = re.findall(pattern, reasoning_text, re.IGNORECASE)
            logic_term_count += len(terms)
            
        # 标准化逻辑术语计数
        reasoning_length = len(reasoning_text)
        normalized_term_density = min(1.0, logic_term_count / (reasoning_length / 100))
        scores["coherence"] = normalized_term_density
        
        # 3. 完整性 - 基于文本长度和是否包含结论
        # 假设合理的推理应该有一定长度
        length_score = min(1.0, len(reasoning_text) / 1000)  # 1000字符以上为满分
        
        # 检查是否包含结论
        conclusion_patterns = [
            r"结论|总结|综上所述|因此可知|所以答案|答案是",  # 中文
            r"conclusion|summary|therefore|thus|so the answer|the answer is",  # 英文
        ]
        
        has_conclusion = False
        for pattern in conclusion_patterns:
            if re.search(pattern, reasoning_text, re.IGNORECASE):
                has_conclusion = True
                break
                
        completeness_score = (length_score + (1.0 if has_conclusion else 0.0)) / 2
        scores["completeness"] = completeness_score
        
        # 4. 术语精确度 - 检测是否使用了精确的术语
        # 逻辑术语
        logic_precision_terms = [
            r"必要条件|充分条件|当且仅当|逻辑推理|演绎|归纳|反例|矛盾|等价|蕴含",  # 中文
            r"necessary|sufficient|if and only if|logical|deduction|induction|counterexample|contradiction|equivalent|implies",  # 英文
        ]
        
        precision_term_count = 0
        for pattern in logic_precision_terms:
            terms = re.findall(pattern, reasoning_text, re.IGNORECASE)
            precision_term_count += len(terms)
            
        precision_score = min(1.0, precision_term_count / 5)  # 最多5个精确术语为满分
        scores["precision"] = precision_score
        
        # 5. 自相矛盾 - 这是一个负面指标，检测是否有自相矛盾的表述
        contradiction_patterns = [
            r"但是.*?前面.*?矛盾",
            r"然而.*?之前.*?相反",
            r"这与.*?冲突",
            r"contradicts.*?earlier",
            r"however.*?opposite",
            r"conflicts with.*?stated",
        ]
        
        contradiction_count = 0
        for pattern in contradiction_patterns:
            contradictions = re.findall(pattern, reasoning_text, re.IGNORECASE)
            contradiction_count += len(contradictions)
            
        # 矛盾越多，一致性分数越低
        consistency_score = max(0.0, 1.0 - contradiction_count * 0.25)
        scores["consistency"] = consistency_score
        
        # 计算综合得分
        weights = {
            "structure": 0.25,
            "coherence": 0.25,
            "completeness": 0.2,
            "precision": 0.15,
            "consistency": 0.15
        }
        
        overall_score = sum(scores[k] * weights[k] for k in scores.keys())
        scores["overall"] = overall_score
        
        return scores
    
    def evaluate_reasoning_similarity(self, reasoning1: str, reasoning2: str) -> Dict[str, float]:
        """
        评估两个推理文本之间的相似度.
        用于鲁棒性与一致性评估，比较原始和干扰样本中的推理.
        
        Args:
            reasoning1: 第一个推理文本
            reasoning2: 第二个推理文本
            
        Returns:
            包含各种相似度指标的字典
        """
        # 对文本进行预处理
        def preprocess(text):
            # 转换为小写
            text = text.lower()
            # 删除标点符号
            text = text.translate(str.maketrans('', '', string.punctuation))
            # 分词
            tokens = nltk.word_tokenize(text)
            return text, tokens
        
        reasoning1_text, reasoning1_tokens = preprocess(reasoning1)
        reasoning2_text, reasoning2_tokens = preprocess(reasoning2)
        
        # 1. Rouge分数
        rouge_scores = self.rouge_scorer.score(reasoning1_text, reasoning2_text)
        rouge1_fscore = rouge_scores['rouge1'].fmeasure
        rouge2_fscore = rouge_scores['rouge2'].fmeasure
        rougeL_fscore = rouge_scores['rougeL'].fmeasure
        
        # 2. BLEU分数
        try:
            bleu_score = sentence_bleu([reasoning1_tokens], reasoning2_tokens, 
                                     smoothing_function=self.bleu_smoothing)
        except:
            bleu_score = 0.0
            
        # 3. 词汇重叠率
        unique_tokens1 = set(reasoning1_tokens)
        unique_tokens2 = set(reasoning2_tokens)
        
        if unique_tokens1 and unique_tokens2:
            overlap = len(unique_tokens1.intersection(unique_tokens2))
            jaccard = overlap / len(unique_tokens1.union(unique_tokens2))
            containment1 = overlap / len(unique_tokens1)
            containment2 = overlap / len(unique_tokens2)
        else:
            jaccard = 0.0
            containment1 = 0.0
            containment2 = 0.0
            
        # 综合相似度分数 (加权平均)
        weights = {
            'rouge1': 0.2,
            'rouge2': 0.25,
            'rougeL': 0.25,
            'bleu': 0.15,
            'jaccard': 0.15
        }
        
        overall_similarity = (
            weights['rouge1'] * rouge1_fscore +
            weights['rouge2'] * rouge2_fscore +
            weights['rougeL'] * rougeL_fscore +
            weights['bleu'] * bleu_score +
            weights['jaccard'] * jaccard
        )
        
        return {
            'rouge1_fscore': rouge1_fscore,
            'rouge2_fscore': rouge2_fscore,
            'rougeL_fscore': rougeL_fscore,
            'bleu_score': bleu_score,
            'jaccard': jaccard,
            'containment1': containment1,  # 1包含在2中的程度
            'containment2': containment2,  # 2包含在1中的程度
            'overall_similarity': overall_similarity
        }
    
    def evaluate_robustness(self, 
                          original_predictions: List[int],
                          perturbed_predictions: List[int],
                          original_reasoning: List[str],
                          perturbed_reasoning: List[str]) -> Dict[str, float]:
        """
        评估模型的鲁棒性，即在干扰样本上保持一致预测的能力.
        
        Args:
            original_predictions: 原始样本的预测答案
            perturbed_predictions: 干扰样本的预测答案
            original_reasoning: 原始样本的推理文本
            perturbed_reasoning: 干扰样本的推理文本
            
        Returns:
            包含鲁棒性指标的字典
        """
        if (len(original_predictions) != len(perturbed_predictions) or
            len(original_predictions) != len(original_reasoning) or
            len(original_predictions) != len(perturbed_reasoning)):
            raise ValueError("所有输入列表的长度必须相同")
            
        # 1. 预测一致性比例
        prediction_consistency = np.mean([1.0 if orig == pert else 0.0 
                                      for orig, pert in zip(original_predictions, perturbed_predictions)])
        
        # 2. 推理相似度
        reasoning_similarities = []
        for orig_r, pert_r in zip(original_reasoning, perturbed_reasoning):
            similarity = self.evaluate_reasoning_similarity(orig_r, pert_r)
            reasoning_similarities.append(similarity['overall_similarity'])
            
        avg_reasoning_similarity = np.mean(reasoning_similarities)
        
        # 3. 推理一致性 - 在预测相同的情况下，推理是否相似
        consistent_reasoning_similarity = []
        for i in range(len(original_predictions)):
            if original_predictions[i] == perturbed_predictions[i]:
                similarity = self.evaluate_reasoning_similarity(
                    original_reasoning[i], perturbed_reasoning[i])
                consistent_reasoning_similarity.append(similarity['overall_similarity'])
                
        avg_consistent_sim = np.mean(consistent_reasoning_similarity) if consistent_reasoning_similarity else 0.0
        
        # 4. 矛盾推理比例 - 推理相似但预测不同
        contradictory_reasoning = []
        for i in range(len(original_predictions)):
            if original_predictions[i] != perturbed_predictions[i]:
                similarity = self.evaluate_reasoning_similarity(
                    original_reasoning[i], perturbed_reasoning[i])
                # 如果相似度高但预测不同，视为矛盾
                if similarity['overall_similarity'] > 0.7:  
                    contradictory_reasoning.append(1.0)
                else:
                    contradictory_reasoning.append(0.0)
                    
        contradictory_rate = np.mean(contradictory_reasoning) if contradictory_reasoning else 0.0
        
        # 综合鲁棒性分数
        robustness_score = 0.5 * prediction_consistency + 0.3 * avg_reasoning_similarity + 0.2 * (1 - contradictory_rate)
        
        return {
            'prediction_consistency': prediction_consistency,
            'avg_reasoning_similarity': avg_reasoning_similarity,
            'consistent_reasoning_similarity': avg_consistent_sim,
            'contradictory_reasoning_rate': contradictory_rate,
            'robustness_score': robustness_score
        }
    
    def compare_methods(self, 
                      baseline_metrics: Dict[str, float], 
                      rvgn_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        比较基线方法和RVGN方法之间的指标差异.
        
        Args:
            baseline_metrics: 基线方法的评估指标
            rvgn_metrics: RVGN方法的评估指标
            
        Returns:
            包含改进百分比的字典
        """
        improvements = {}
        
        for key in baseline_metrics:
            if key in rvgn_metrics:
                # 计算相对改进百分比
                if baseline_metrics[key] > 0:  # 避免除以零
                    rel_improvement = (rvgn_metrics[key] - baseline_metrics[key]) / baseline_metrics[key] * 100
                else:
                    # 如果基线为0，直接计算绝对改进
                    rel_improvement = (rvgn_metrics[key] - baseline_metrics[key]) * 100
                    
                # 计算绝对改进
                abs_improvement = rvgn_metrics[key] - baseline_metrics[key]
                
                improvements[f"{key}_absolute"] = abs_improvement
                improvements[f"{key}_relative"] = rel_improvement
                
        return improvements
    
    def save_evaluation_results(self, results: Dict[str, Any], output_dir: str, 
                              experiment_name: str, timestamp: bool = True) -> str:
        """
        将评估结果保存到文件.
        
        Args:
            results: 评估结果字典
            output_dir: 输出目录
            experiment_name: 实验名称
            timestamp: 是否在文件名中添加时间戳
            
        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if timestamp:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{experiment_name}_{timestamp_str}.json"
        else:
            filename = f"{experiment_name}.json"
            
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"评估结果已保存至: {output_path}")
        return output_path

# 示例用法
if __name__ == "__main__":
    evaluator = RVGNEvaluator()
    
    # 示例推理文本
    example_reasoning = """
    步骤1: 首先，我们需要理解题目的条件。已知"所有的bloops都是razzies"和"所有的razzies都是lazzies"。
    
    步骤2: 让我们分析这两个条件。
    条件1：所有的bloops都是razzies，可以表示为"如果x是bloop，那么x是razzie"。
    条件2：所有的razzies都是lazzies，可以表示为"如果x是razzie，那么x是lazzie"。
    
    步骤3: 现在，让我们使用逻辑推理。根据条件1，任何bloop都是razzie。然后根据条件2，任何razzie都是lazzie。
    因此，通过传递性，任何bloop也必定是lazzie。
    
    步骤4: 检验我们的结论。假设x是一个bloop，那么x是razzie（根据条件1），又因为x是razzie，所以x是lazzie（根据条件2）。
    因此，对于任何bloop，它也都是lazzie。
    
    结论: 所以问题"所有的bloops是否都是lazzies？"的答案是肯定的，是的，所有的bloops都是lazzies。
    
    选项1是正确的。
    """
    
    # 测试推理质量评估
    quality_scores = evaluator.evaluate_reasoning_quality(example_reasoning)
    print("推理质量评分:", quality_scores)
    
    # 测试答案提取
    extracted_answer = evaluator.extract_answer_from_reasoning(example_reasoning)
    print("提取的答案索引:", extracted_answer)
    
    # 测试任务准确率
    predictions = [0, 1, 2, 3, 0]
    ground_truth = [0, 1, 2, 2, 1]
    accuracy_metrics = evaluator.evaluate_task_accuracy(predictions, ground_truth)
    print("任务准确率指标:", accuracy_metrics)
    
    # 测试推理相似度
    example_reasoning2 = """
    步骤1: 我们需要分析两个已知条件。
    条件A：所有bloops都是razzies。
    条件B：所有razzies都是lazzies。
    
    步骤2: 通过逻辑推理，如果所有bloops都是razzies（条件A），而且所有razzies都是lazzies（条件B），
    那么根据三段论，所有bloops也必然都是lazzies。
    
    步骤3: 验证: 任何一个bloop都是razzie，任何一个razzie都是lazzie，所以任何一个bloop都是lazzie。
    
    结论: 答案是肯定的，所有bloops都是lazzies。选项A正确。
    """
    
    similarity_metrics = evaluator.evaluate_reasoning_similarity(example_reasoning, example_reasoning2)
    print("推理相似度指标:", similarity_metrics)
