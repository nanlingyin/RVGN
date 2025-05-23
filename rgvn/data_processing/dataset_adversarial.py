"""
AdversarialLogic 数据集处理模块。
这是一个模拟数据集模块，用于创建带有逻辑干扰的对抗性样本。
该数据集可以用来测试模型对微小变化的鲁棒性。
"""

import os
import json
import random
import copy
from typing import Dict, List, Tuple, Optional, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdversarialLogicProcessor:
    """
    逻辑推理对抗性数据集处理器.
    基于现有的LogiQA或ReClor数据集创建对抗性样本。
    """
    
    def __init__(self, data_dir: str = None):
        """
        初始化对抗性数据处理器.
        
        Args:
            data_dir: 数据目录路径. 如果为None，默认使用项目根目录下的'data/adversarial'.
        """
        if data_dir is None:
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算项目根目录
            project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
            self.data_dir = os.path.join(project_root, 'data', 'adversarial')
        else:
            self.data_dir = data_dir
            
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_adversarial_samples(self, 
                                    original_data: List[Dict], 
                                    perturbation_type: str = 'all',
                                    ratio: float = 1.0) -> List[Dict]:
        """
        基于原始数据生成对抗性样本.
        
        Args:
            original_data: 原始数据列表，可以是LogiQA或ReClor格式.
            perturbation_type: 干扰类型，可选 'irrelevant_info', 'negation', 'reordering', 'all'.
            ratio: 生成对抗性样本的比例，范围[0, 1.0].
            
        Returns:
            对抗性样本列表.
        """
        num_samples = int(len(original_data) * ratio)
        if num_samples <= 0:
            logger.warning("生成样本数量为0，请检查原始数据集大小和比例参数")
            return []
            
        # 选择要修改的样本
        samples_to_perturb = random.sample(original_data, num_samples)
        adversarial_samples = []
        
        for sample in samples_to_perturb:
            if perturbation_type == 'irrelevant_info' or perturbation_type == 'all':
                perturbed = self._add_irrelevant_information(copy.deepcopy(sample))
                adversarial_samples.append(perturbed)
                
            if perturbation_type == 'negation' or perturbation_type == 'all':
                perturbed = self._add_negation(copy.deepcopy(sample))
                adversarial_samples.append(perturbed)
                
            if perturbation_type == 'reordering' or perturbation_type == 'all':
                perturbed = self._reorder_information(copy.deepcopy(sample))
                adversarial_samples.append(perturbed)
        
        logger.info(f"生成了 {len(adversarial_samples)} 个对抗性样本")
        return adversarial_samples
    
    def _add_irrelevant_information(self, sample: Dict) -> Dict:
        """
        向样本的上下文中添加不相关的信息.
        
        Args:
            sample: 原始样本
            
        Returns:
            修改后的样本
        """
        irrelevant_statements = [
            "研究表明，大多数人早晨的记忆力更好。",
            "调查显示，蓝色是最受欢迎的颜色之一。",
            "有些专家认为，非结构化时间对创造力有益。",
            "根据最新报告，平均阅读速度是每分钟250个单词。",
            "科学家发现，定期休息可以提高工作效率。",
            "历史学家指出，文明的发展往往依赖于水源。",
            "统计数据显示，每天锻炼可以降低压力。",
            "心理学研究表明，清晨决策质量往往更高。"
        ]
        
        # 生成一个唯一的ID，以区分对抗性样本
        sample['id'] = f"{sample['id']}_adv_irrelevant"
        
        # 在原始上下文开始或结束处添加不相关信息
        irrelevant_info = random.choice(irrelevant_statements)
        if random.choice([True, False]):
            # 添加到开始
            sample['context'] = irrelevant_info + " " + sample['context']
        else:
            # 添加到结尾
            sample['context'] = sample['context'] + " " + irrelevant_info
            
        # 添加扰动标记
        sample['perturbed'] = True
        sample['perturbation_type'] = 'irrelevant_info'
            
        return sample
    
    def _add_negation(self, sample: Dict) -> Dict:
        """
        在样本的上下文或问题中添加或移除否定.
        
        Args:
            sample: 原始样本
            
        Returns:
            修改后的样本
        """
        sample['id'] = f"{sample['id']}_adv_negation"
        
        # 获取上下文的第一个句子
        context_parts = sample['context'].split('。')
        
        if len(context_parts) > 0:
            # 处理第一个句子
            first_sentence = context_parts[0]
            
            # 我们将在50%的情况下添加否定，在50%的情况下移除否定
            if '不' in first_sentence or '没有' in first_sentence:
                # 移除否定
                modified = first_sentence.replace('不', '').replace('没有', '')
            else:
                # 添加否定
                words = first_sentence.split()
                if len(words) >= 3:
                    # 在中间某处添加一个"不"
                    position = random.randint(1, min(len(words)-1, 3))
                    words.insert(position, "不")
                    modified = ''.join(words)
                else:
                    # 句子太短，在开头添加"并非"
                    modified = "并非" + first_sentence
            
            # 重建上下文
            context_parts[0] = modified
            sample['context'] = '。'.join(context_parts)
            
            # 确保句尾有句号
            if not sample['context'].endswith('。'):
                sample['context'] += '。'
        
        # 添加扰动标记
        sample['perturbed'] = True
        sample['perturbation_type'] = 'negation'
        
        return sample
    
    def _reorder_information(self, sample: Dict) -> Dict:
        """
        重新排序样本中的信息，但不改变整体含义.
        
        Args:
            sample: 原始样本
            
        Returns:
            修改后的样本
        """
        sample['id'] = f"{sample['id']}_adv_reordering"
        
        # 分割上下文为句子
        sentences = sample['context'].split('。')
        sentences = [s + '。' for s in sentences if s]  # 确保每个句子末尾有句号
        
        if len(sentences) >= 3:
            # 只重排前三个句子
            to_shuffle = sentences[:3]
            random.shuffle(to_shuffle)
            sentences[:3] = to_shuffle
        elif len(sentences) >= 2:
            # 只有两个句子时交换它们
            sentences[0], sentences[1] = sentences[1], sentences[0]
        
        # 重建上下文
        sample['context'] = ''.join(sentences)
        
        # 添加扰动标记
        sample['perturbed'] = True
        sample['perturbation_type'] = 'reordering'
        
        return sample
    
    def save_adversarial_data(self, data: List[Dict], name: str = "adversarial") -> str:
        """
        保存对抗性样本到文件.
        
        Args:
            data: 对抗性样本列表
            name: 文件名前缀
            
        Returns:
            保存的文件路径
        """
        path = os.path.join(self.data_dir, f"{name}.json")
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"对抗性样本已保存至: {path}")
            return path
        except Exception as e:
            logger.error(f"保存对抗性样本时出错: {e}")
            return ""
    
    def load_adversarial_data(self, name: str = "adversarial") -> List[Dict]:
        """
        加载对抗性样本.
        
        Args:
            name: 文件名前缀
            
        Returns:
            对抗性样本列表
        """
        path = os.path.join(self.data_dir, f"{name}.json")
        
        if not os.path.exists(path):
            logger.error(f"对抗性样本文件不存在: {path}")
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"从 {path} 加载了 {len(data)} 个对抗性样本")
            return data
        except Exception as e:
            logger.error(f"加载对抗性样本时出错: {e}")
            return []

# 示例用法
if __name__ == "__main__":
    # 创建一些模拟数据用于测试
    test_data = [
        {
            "id": "sample1",
            "context": "所有的A都是B。所有的B都是C。经过研究发现，有些D是A。",
            "question": "以下哪项一定为真？",
            "options": ["所有的A都是C", "所有的C都是B", "有些D是C", "所有的D都是A"],
            "answer": 2
        },
        {
            "id": "sample2",
            "context": "如果明天下雨，学校将取消野餐。学校没有取消野餐。天气预报说明天有60%的降雨概率。",
            "question": "根据以上信息，以下哪项一定为真？",
            "options": ["明天不会下雨", "明天会下雨", "学校取消了野餐", "天气预报不准确"],
            "answer": 0
        }
    ]
    
    processor = AdversarialLogicProcessor()
    
    # 生成各种类型的对抗性样本
    adv_samples = processor.generate_adversarial_samples(test_data, 'all')
    
    # 保存样本
    processor.save_adversarial_data(adv_samples, "test_adversarial")
    
    # 加载样本
    loaded = processor.load_adversarial_data("test_adversarial")
    
    # 显示样本
    if loaded:
        print("对抗性样本示例:")
        print(json.dumps(loaded[0], ensure_ascii=False, indent=2))
