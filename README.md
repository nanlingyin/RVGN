# RVGN: 推理图验证网络

RVGN (Reasoning Graph Verification Network) 是一个用于构建、分析和验证推理过程的框架。该项目提供了一种系统化方法来评估和改善大型语言模型（LLM）的推理能力，特别是在逻辑谜题和推理任务方面。

## 项目概述

RVGN通过将文本推理转换为图结构来分析推理流程，检测潜在错误并提供改进建议。该框架特别关注：

- **结构化分析**：将推理过程表示为有向图，以捕获步骤之间的逻辑依赖关系
- **错误检测**：识别逻辑跳跃、循环推理和不一致性
- **质量评估**：通过多维度指标系统评估推理质量
- **交互式改进**：基于分析结果提供具体反馈，帮助改进推理

## 功能特点

- **图结构构建**：将文本推理自动转换为具有丰富语义的图结构
- **多种验证器**：包括数学验证（SymPy支持）、逻辑分析等
- **全面评估指标**：包括任务准确度、推理质量评分和鲁棒性测试
- **多数据集支持**：兼容LogiQA、ReClor等逻辑推理数据集
- **对抗性测试**：通过系统性扰动测试推理的健壮性
- **可视化工具**：直观展示推理图结构和错误检测结果

## 安装说明

1. 克隆仓库:
```
git clone https://github.com/yourusername/rvgn.git
cd rvgn
```

2. 安装依赖:
```
pip install -r requirements.txt
```

3. 下载数据集:
```
python scripts/download_datasets.py
```

## 使用指南

### 基本使用

运行以下命令来测试基本功能:

```python
python rgvn/examples/basic_example.py
```

### 运行完整实验

1. 配置实验参数 (修改 `experiments/config.yaml`)

2. 执行实验:
```
python scripts/run_experiment.py --config experiments/config.yaml
```

3. 生成对抗性样本:
```
python scripts/generate_adversarial.py --dataset logiqa --perturbation all
```

## 项目结构

```
RVGN/
├── data/                   # 数据目录
│   ├── logiqa/            # LogiQA数据集
│   ├── reclor/            # ReClor数据集
│   └── adversarial/       # 生成的对抗性样本
├── experiments/           # 实验配置和脚本
│   └── config.yaml        # 实验配置文件
├── results/               # 实验结果输出目录
│   ├── figures/           # 生成的图表和可视化
│   └── tables/            # 结果表格和数据
├── rgvn/                  # 主要源代码
│   ├── core/              # 核心组件
│   │   ├── error_detector.py    # 错误检测器
│   │   └── graph_builder.py     # 推理图构建器
│   ├── data_processing/   # 数据处理模块
│   │   ├── data_manager.py         # 数据管理器
│   │   ├── dataset_logiqa.py       # LogiQA处理器
│   │   ├── dataset_reclor.py       # ReClor处理器
│   │   └── dataset_adversarial.py  # 对抗样本生成器
│   ├── evaluation/        # 评估模块
│   │   ├── evaluator.py          # 实验管理器
│   │   └── metrics.py            # 评估指标实现
│   ├── examples/          # 示例代码
│   │   └── basic_example.py      # 基本使用示例
│   ├── llm/               # LLM交互模块
│   │   └── api_client.py         # LLM API客户端
│   ├── utils/             # 实用工具
│   │   └── text_processing.py    # 文本处理工具
│   └── validators/        # 验证器模块
│       └── math_validator.py     # 数学验证器
├── scripts/               # 工具脚本
│   ├── download_datasets.py      # 下载数据集脚本
│   ├── generate_adversarial.py   # 生成对抗样本脚本
│   └── run_experiment.py         # 运行实验脚本
├── tests/                 # 测试代码
├── requirements.txt       # 依赖列表
├── LICENSE                # 许可证文件
└── README.md              # 项目说明文件
```

## 评估指标

RVGN使用三类主要评估指标:

1. **任务准确率 (Task Accuracy)**:
   - 测量模型在解决逻辑问题时的准确度
   - 包括分类准确率和F1分数

2. **推理过程质量评分 (Reasoning Process Quality Score, RPQS)**:
   - 结构性：推理步骤的清晰度和组织性
   - 逻辑连贯性：步骤之间的逻辑流是否连贯
   - 事实正确性：所述事实的准确程度
   - 相关性：内容与问题的相关程度
   - 完整性：推理过程是否完整

3. **鲁棒性与一致性 (Robustness & Consistency)**:
   - 对扰动的稳定性 (如不相关信息、否定、重排序)
   - 不同问题表述下推理的一致性

## 实验方法

RVGN采用以下实验步骤进行系统评估:

1. **基线实验**: 评估原始LLM在各数据集上的表现
2. **RVGN增强实验**: 使用RVGN分析和改进LLM推理
3. **对抗性实验**: 通过扰动测试推理鲁棒性
4. **比较分析**: 多维度对比RVGN增强前后的差异

## 贡献指南

欢迎对RVGN项目做出贡献! 请参考以下步骤:

1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 引用

如果您在研究中使用了RVGN，请引用本项目:

```bibtex
@misc{rvgn2025,
  author = {Your Name},
  title = {RVGN: Reasoning Graph Verification Network},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/rvgn}}
}
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
