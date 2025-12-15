# Bad Case样本扩充系统 - 模块设计详解

## 目录

1. [系统概述](#1-系统概述)
2. [核心架构](#2-核心架构)
3. [模块设计](#3-模块设计)
4. [数据模型](#4-数据模型)
5. [技术选型](#5-技术选型)
6. [部署方案](#6-部署方案)
7. [扩展性设计](#7-扩展性设计)

---

## 1. 系统概述

### 1.1 背景与目标

在实际应用中，基于Qwen3 LoRA微调的文生文模型可能在部分场景下表现不佳。本系统旨在通过自动化的Bad Case诊断、样本生成、质量验证和数据混合流程，精准修复模型缺陷，同时防止灾难性遗忘。

### 1.2 核心理念

**D-G-V-M闭环流程**：
- **Diagnose (诊断)**: 精准定位错误类型与根源
- **Generate (生成)**: 大模型针对性生成高质量样本
- **Verify (验证)**: 多级质量保障确保样本可靠性
- **Mix (混合)**: 科学配比防止灾难性遗忘

### 1.3 设计原则

1. **精准诊断**: 五维错误分类体系，准确定位问题根源
2. **靶向生成**: 根据错误类型采用不同的Prompt策略
3. **严格验证**: 三级防火墙机制确保样本质量
4. **防遗忘混合**: 科学配比原始数据与增强数据
5. **可追溯性**: 完整记录样本生成和验证过程
6. **模块化**: 各组件职责清晰，接口明确
7. **可扩展性**: 支持新的错误类型、Teacher Model和任务类型

---

## 2. 核心架构

### 2.1 系统架构图

系统采用分层架构，包含输入层、处理层和输出层：

```
┌─────────────────────────────────────────────────────────────┐
│                         输入层                               │
├─────────────────────────────────────────────────────────────┤
│  Bad Cases Collection  │  Original Training Data  │  Config │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        处理层                                │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ 诊断模块      │→  │ 生成模块      │→  │ 验证模块      │   │
│  │ Diagnoser    │   │ Generator    │   │ Validator    │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         ↓                                      ↓            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ 混合模块      │←  │ 训练模块      │←  │ 评估模块      │   │
│  │ Mixer        │   │ Trainer      │   │ Evaluator    │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                         输出层                               │
├─────────────────────────────────────────────────────────────┤
│  Enhanced Samples  │  Training Dataset  │  Fine-tuned Model │
│  Validation Reports  │  Training Logs  │  Evaluation Reports│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流程

```
Bad Cases → 诊断 → 生成 → 验证 → 混合 → 训练 → 评估 → 微调模型
    ↑                                    ↓
    └────────────── 持续优化 ←────────────┘
```

### 2.3 关键流程说明

1. **诊断阶段**: 对Bad Cases进行多维度分析，确定错误类型（内容失真、逻辑断层、风格不符、表达低质、格式违规）
2. **生成阶段**: 根据错误类型调用Teacher Model（如Qwen-Max、GPT-4）生成针对性的增强样本
3. **验证阶段**: 通过L1格式验证、L2逻辑验证、L3价值验证三级防火墙确保样本质量
4. **混合阶段**: 按照科学比例混合增强样本和原始高质量数据，对P0类型样本进行上采样
5. **训练阶段**: 使用混合数据集进行LoRA微调，采用优化的配置参数
6. **评估阶段**: 评估修复效果、样本质量、通用能力保持等指标，实施质量门禁

---

## 3. 模块设计

### 3.1 诊断模块 (Diagnoser)

**职责**: 对Bad Cases进行多维度分析，确定错误类型和根本原因

**核心组件**:

1. **MultiDimensionScorer (多维度评分器)**
   - 计算事实一致性评分
   - 计算逻辑连贯性评分
   - 计算风格相似度评分
   - 计算语言流畅度评分
   - 计算格式合规性评分

2. **ErrorClassifier (错误分类器)**
   - 根据评分结果和优先级规则分类错误类型
   - 支持五种错误类型：内容失真、逻辑断层、风格不符、表达低质、格式违规
   - 按P0/P1/P2优先级标记

3. **OODDetector (分布外检测器)**
   - 计算样本与训练集的嵌入向量相似度
   - 识别超出训练数据分布的样本
   - 支持可选的OOD检测功能

4. **DiagnosisReporter (诊断报告生成器)**
   - 生成包含错误类型、评分、OOD分数的诊断报告
   - 提供主要问题描述和修复建议

**接口设计**:
- `diagnose_single(bad_case)`: 诊断单个Bad Case
- `diagnose_batch(bad_cases)`: 批量诊断并按错误类型分组

**输入**: Bad Case样本（包含input、output、reference等字段）
**输出**: 诊断结果（包含错误类型、优先级、各维度评分、OOD评分）

---

### 3.2 生成模块 (Generator)

**职责**: 根据诊断结果调用Teacher Model生成针对性的增强样本

**核心组件**:

1. **StrategySelector (策略选择器)**
   - 根据错误类型选择生成策略
   - 支持五种策略：事实增强、CoT注入、风格重写、表达多样化、模板强化

2. **PromptTemplateEngine (Prompt模板引擎)**
   - 管理不同错误类型的Prompt模板
   - 支持模板变量替换和渲染
   - 从配置文件加载模板

3. **TeacherModelClient (Teacher Model客户端)**
   - 封装Teacher Model API调用
   - 支持多种模型（Qwen-Max、GPT-4等）
   - 通过Provider接口适配不同API协议

4. **RetryManager (重试管理器)**
   - 实现指数退避重试机制
   - 最多重试3次
   - 记录重试日志

5. **MetadataAnnotator (元数据标注器)**
   - 为生成的样本添加元数据
   - 包含难度等级、变体类型、目标错误类型、业务场景等

**Prompt策略矩阵**:

| 错误类型 | 生成策略 | 核心思想 |
|---------|---------|---------|
| 内容失真 | 事实增强 | 保持输入事实不变，生成多种表达方式 |
| 逻辑断层 | CoT注入 | 要求输出包含显式的推理步骤 |
| 风格不符 | 风格重写 | 保持核心语义，调整场景/角色/语气 |
| 表达低质 | 表达多样化 | 提供不同质量梯度的表达方式 |
| 格式违规 | 模板强化 | 确保输出符合指定格式 |

**接口设计**:
- `generate_for_case(bad_case, diagnosis, num_samples)`: 为单个Bad Case生成增强样本
- `generate_batch(diagnosed_cases, max_retries)`: 批量生成增强样本

**输入**: 诊断结果、Bad Case样本
**输出**: 增强样本列表（包含input、output、metadata）

---

### 3.3 验证模块 (Validator)

**职责**: 对生成的样本执行三级验证，确保质量

**核心组件**:

1. **L1FormatValidator (格式验证器)**
   - 检查JSON结构完整性
   - 验证字段类型正确性
   - 确认必填字段存在性
   - 使用Schema验证

2. **L2LogicValidator (逻辑验证器)**
   - 使用LLM-as-a-Judge评估逻辑一致性
   - 评估事实准确性
   - 检查推理完整性
   - 阈值：0.85

3. **L3ValueValidator (价值验证器)**
   - 计算嵌入向量相似度
   - 评估样本多样性
   - 防止过度相似或偏离过大
   - 相似度范围：0.4-0.9

4. **QualityScorer (质量评分器)**
   - 计算综合质量分数
   - 公式：quality_score = l2_score * 0.6 + l3_score * 0.4

**三级验证流程**:
```
样本 → L1格式验证 → L2逻辑验证 → L3价值验证 → 通过/拒绝
       ↓ 失败         ↓ 失败         ↓ 失败
      拒绝          拒绝           拒绝
```

**接口设计**:
- `validate_sample(original_case, enhanced_sample)`: 验证单个样本
- `validate_batch(cases_with_samples)`: 批量验证样本

**输入**: 原始Bad Case、增强样本
**输出**: 验证结果（包含通过状态、各级验证结果、质量分数、拒绝原因）

---

### 3.4 混合模块 (Mixer)

**职责**: 科学混合增强样本和原始数据，防止灾难性遗忘

**核心组件**:

1. **SampleClassifier (样本分类器)**
   - 按错误类型分组样本
   - 统计各类型样本数量

2. **Upsampler (上采样器)**
   - 对P0类型样本（内容失真、逻辑断层）进行2倍上采样
   - 增加难例权重

3. **QualitySorter (质量排序器)**
   - 按质量分数降序排序
   - 优先选择高质量样本

4. **RatioController (比例控制器)**
   - 控制增强样本与原始样本的混合比例
   - 默认比例：增强样本70%，原始样本30%

5. **Shuffler (打乱器)**
   - 随机打乱最终数据集
   - 确保训练时的随机性

**混合策略**:
```
增强样本 (70%) + 原始高质量样本 (30%) → 随机打乱 → 最终训练集
    ↑
P0类型样本 × 2 (上采样)
```

**接口设计**:
- `create_training_dataset(enhanced_samples, original_samples)`: 创建训练数据集

**输入**: 验证通过的增强样本、原始高质量训练数据
**输出**: 训练数据集（包含samples、statistics）

---

### 3.5 训练模块 (Trainer)

**职责**: 使用混合数据集进行LoRA微调

**核心组件**:

1. **LoRAConfigGenerator (LoRA配置生成器)**
   - 根据任务复杂度生成优化的LoRA配置
   - 支持simple/medium/complex三种复杂度

2. **DataLoader (数据加载器)**
   - 加载训练数据集
   - 支持批量加载和预处理

3. **TrainingEngine (训练引擎)**
   - 执行LoRA微调训练
   - 支持梯度累积、混合精度训练
   - 实施早停机制

4. **CheckpointManager (检查点管理器)**
   - 保存训练检查点
   - 管理最佳模型
   - 支持断点续训

**LoRA配置优化**:

| 任务复杂度 | rank | alpha | 目标模块 | 学习率 |
|-----------|------|-------|---------|--------|
| Simple | 32 | 64 | q_proj, v_proj | 1e-5 |
| Medium | 64 | 128 | q_proj, v_proj, k_proj, o_proj | 1e-5 |
| Complex | 64 | 128 | 全部线性层 | 5e-6 |

**关键参数**:
- rank: 64 (提供足够的参数调整空间)
- alpha: 128 (等于2倍rank)
- dropout: 0.05
- 目标模块: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
- 学习率: 1e-5 (防止破坏原有通用能力)
- 训练轮数: 3
- 批次大小: 2
- 梯度累积步数: 8

**接口设计**:
- `train(train_dataset, val_dataset)`: 执行训练
- `get_lora_config(task_complexity)`: 获取LoRA配置

**输入**: 训练数据集、验证数据集
**输出**: 训练结果（包含模型路径、最佳检查点、训练日志、最终指标）

---

### 3.6 评估模块 (Evaluator)

**职责**: 评估微调后模型的性能和质量门禁

**核心组件**:

1. **MetricsCalculator (指标计算器)**
   - 计算Bad Cases修复率
   - 计算增强样本合格率
   - 计算通用能力保持率
   - 计算多样性评分
   - 计算推理效率比值

2. **QualityGate (质量门禁)**
   - 检查各指标是否达标
   - 阻止未达标模型部署
   - 生成改进建议

3. **ReportGenerator (报告生成器)**
   - 生成评估报告
   - 包含各项指标、通过状态、改进建议

**质量门禁标准**:

| 指标 | 目标值 | 说明 |
|-----|--------|------|
| Bad Cases修复率 | ≥80% | 修复的Bad Cases占总数的比例 |
| 增强样本合格率 | ≥85% | 通过三级验证的样本比例 |
| 通用能力保持率 | ≥95% | 微调后通用测试集得分与微调前的比值 |
| 多样性评分 | <0.6 | 增强样本间的平均余弦相似度 |
| 推理效率比值 | ≤110% | 微调后推理延迟与微调前的比值 |

**接口设计**:
- `evaluate(model, test_bad_cases, general_test_set)`: 全面评估模型

**输入**: 微调后模型、测试Bad Cases、通用测试集
**输出**: 评估报告（包含各项指标、门禁通过状态、失败门禁列表、改进建议）

---

### 3.7 管道编排 (Pipeline)

**职责**: 编排整个D-G-V-M流程

**核心功能**:
- 协调各模块的执行顺序
- 管理数据在模块间的流转
- 处理异常和错误
- 记录执行日志
- 生成最终报告

**执行流程**:
1. 加载配置和输入数据
2. 执行诊断模块
3. 执行生成模块
4. 执行验证模块
5. 执行混合模块
6. 执行训练模块
7. 执行评估模块
8. 生成最终报告

**接口设计**:
- `run(bad_cases_path, original_data_path, output_dir)`: 执行完整流程

**输入**: Bad Cases文件路径、原始训练数据路径、输出目录
**输出**: 管道结果（包含增强样本路径、训练数据集路径、模型路径、报告路径、成功状态）

---

## 4. 数据模型

### 4.1 Bad Case

```
Bad Case:
  - id: 唯一标识符
  - task_type: 任务类型 (summarization/translation/dialogue/qa/general)
  - input: 输入文本
  - output: 模型当前输出
  - reference: 参考答案（可选）
  - format_rules: 格式规则（可选）
  - notes: 备注（可选）
  - error_type: 错误类型标注（auto/manual）
```

### 4.2 诊断结果

```
DiagnosisResult:
  - case_id: Bad Case ID
  - error_type: 错误类型 (content_distortion/logic_failure/style_mismatch/poor_expression/format_violation)
  - priority: 优先级 (P0/P1/P2)
  - scores: 各维度评分
    - fact_consistency: 事实一致性
    - logical_coherence: 逻辑连贯性
    - style_similarity: 风格相似度
    - fluency: 语言流畅度
    - format_compliance: 格式合规性
  - ood_score: 分布外评分
  - primary_issue: 主要问题描述
  - timestamp: 诊断时间
```

### 4.3 增强样本

```
EnhancedSample:
  - id: 唯一标识符
  - source_case_id: 源Bad Case ID
  - task_type: 任务类型
  - input: 输入文本
  - output: 输出文本（可能包含reasoning字段）
  - metadata: 元数据
    - difficulty: 难度等级 (easy/medium/hard)
    - variation_type: 变体类型
    - target_error_type: 目标错误类型
    - generator_strategy: 生成策略
    - business_scene: 业务场景（可选）
    - user_profile: 用户画像（可选）
    - created_at: 创建时间
  - quality_score: 质量分数
  - validation_passed: 验证通过状态
```

### 4.4 验证结果

```
ValidationResult:
  - passed: 是否通过
  - level: 验证级别 (L1/L2/L3/ALL)
  - l1_result: L1验证结果
    - passed: 是否通过
    - errors: 错误列表
  - l2_result: L2验证结果
    - passed: 是否通过
    - score: 逻辑评分
  - l3_result: L3验证结果
    - passed: 是否通过
    - similarity: 相似度
  - quality_score: 综合质量分数
  - rejection_reason: 拒绝原因（如果未通过）
```

### 4.5 训练数据集

```
TrainingDataset:
  - samples: 样本列表
  - statistics: 统计信息
    - total_count: 总样本数
    - enhanced_count: 增强样本数
    - original_count: 原始样本数
    - type_distribution: 类型分布
    - difficulty_distribution: 难度分布
    - quality_score_avg: 平均质量分数
```

---

## 5. 技术选型

### 5.1 编程语言和框架

- **Python 3.9+**: 主要开发语言
- **PyTorch 2.0+**: 深度学习框架
- **Transformers 4.30+**: 模型加载和推理
- **PEFT 0.4+**: LoRA微调
- **Sentence-Transformers**: 嵌入向量计算

### 5.2 数据处理

- **JSONL**: 数据存储格式
- **Pydantic**: 数据验证和Schema定义
- **PyYAML**: 配置文件解析

### 5.3 API调用

- **OpenAI SDK**: Teacher Model API调用
- **Requests**: HTTP请求
- **aiohttp**: 异步HTTP请求

### 5.4 测试

- **pytest**: 单元测试框架
- **Hypothesis**: 属性测试库
- **pytest-mock**: Mock工具
- **pytest-cov**: 覆盖率工具

### 5.5 日志和监控

- **logging**: Python标准日志库
- **tensorboard**: 训练可视化
- **tqdm**: 进度条显示

---

## 6. 部署方案

### 6.1 环境要求

**硬件要求**:
- CPU: 8核以上
- 内存: 32GB以上
- GPU: 可选，用于加速嵌入向量计算和模型训练
- 存储: 100GB以上可用空间

**软件要求**:
- 操作系统: Linux/Windows/macOS
- Python: 3.9+
- CUDA: 11.8+ (如使用GPU)

### 6.2 配置管理

系统使用YAML格式的配置文件，包含以下配置项：

- **pipeline**: 管道配置（名称、版本）
- **diagnoser**: 诊断器配置（任务类型、阈值）
- **generator**: 生成器配置（模型、API、并发数）
- **validator**: 验证器配置（各级阈值）
- **mixer**: 混合器配置（比例、上采样规则）
- **trainer**: 训练器配置（LoRA参数、训练参数）
- **evaluator**: 评估器配置（各指标阈值）
- **security**: 安全配置（PII检测、内容安全）

### 6.3 运行模式

1. **开发模式**: 使用小规模测试数据，启用详细日志
2. **生产模式**: 使用完整数据集，启用所有验证步骤
3. **调试模式**: 保存中间结果，启用性能分析

### 6.4 监控和告警

**监控指标**:
- 处理进度和吞吐量
- API调用成功率和延迟
- 验证通过率
- 资源使用情况（CPU、内存、GPU）
- 错误率和错误类型分布

**告警规则**:
- API调用失败率 > 10%
- 验证通过率 < 70%
- 内存使用 > 90%
- 处理时间超过预期 2倍
- 连续错误次数 > 10

---

## 7. 扩展性设计

### 7.1 支持新的错误类型

添加新的错误类型只需：
1. 在配置文件中定义新的Prompt模板
2. 在诊断器中添加新的分类规则
3. 在生成器中注册新的策略处理器

### 7.2 支持新的Teacher Model

通过Provider接口适配新的API：
- 实现`BaseProvider`接口
- 实现`generate()`方法
- 实现`validate_response()`方法

### 7.3 支持新的任务类型

系统设计为任务类型无关，支持各种文生文任务：
- 摘要生成 (summarization)
- 文本翻译 (translation)
- 对话系统 (dialogue)
- 问答系统 (qa)
- 内容创作 (creative_writing)
- 结构化输出 (structured_output)
- 通用文生文 (general)

### 7.4 插件机制

系统支持通过插件扩展功能：
1. **自定义评分器插件**: 实现`ScorerPlugin`接口
2. **自定义验证器插件**: 实现`ValidatorPlugin`接口
3. **自定义后处理插件**: 实现`PostProcessorPlugin`接口

---

## 8. 总结

本架构方案提供了一个完整的Bad Case样本扩充系统设计，具有以下特点：

1. **模块化设计**: 各组件职责清晰，接口明确，易于测试和维护
2. **可扩展性**: 支持新的错误类型、Teacher Model和任务类型
3. **质量保证**: 三级验证机制确保样本质量
4. **性能优化**: 批量处理、并发控制和缓存策略提高处理效率
5. **安全可靠**: PII检测、内容安全和完善的错误处理机制
6. **可追溯性**: 完整的元数据记录和日志系统

系统设计遵循软件工程最佳实践，为Bad Case驱动的模型优化提供了一个可靠、高效、可扩展的解决方案。
