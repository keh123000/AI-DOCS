# 通用文生文模型 Bad Cases 修复与样本扩充方案

## 1\. 核心理念：D-G-V-M 闭环修复

传统数据增强往往是随机扩充，容易引入噪音。本方案采用 **D-G-V-M（诊断-生成-验证-混合）** 闭环流程，确保每一条生成的数据都能精确打击模型的痛点。

  * **Diagnose (诊断)**：不仅知道错了，还要分类是“没见过”、“逻辑差”还是“格式错”。
  * **Generate (生成)**：利用大模型定向生成高价值样本（Teacher-Student Distillation）。
  * **Verify (验证)**：构建三级防火墙，清洗大模型生成的“幻觉”数据。
  * **Mix (混合)**：科学配比原始数据与增强数据，防止灾难性遗忘。

-----

## 2\. 第一阶段：深度诊断 (Diagnosis)

不要将所有 Bad Cases 一视同仁。建立四维坐标系，对错误进行归类，以便采用不同的扩充策略。

| 错误类型 | 典型表现 | 根本原因 | 修复/扩充策略 |
| :--- | :--- | :--- | :--- |
| **Type A: OOD (知识盲区)** | 遇到新术语、方言、特定场景时胡说八道。 | 训练数据分布未覆盖（相似度 \< 0.7）。 | **场景泛化 & 风格迁移** (重写Input) |
| **Type B: Logic (逻辑断层)** | 因果倒置、多步推理失败、长文本前后矛盾。 | 模型推理深度不足，缺乏中间步骤。 | **CoT 思维链注入** (重写Output) |
| **Type C: Ambiguity (边界模糊)** | 把合规判为违规，被语气误导。 | 决策边界不清晰，主要看表象而非内核。 | **对抗样本生成** (Hard Negatives) |
| **Type D: Format (指令遵循)** | JSON 缺字段、标签未闭合、未按指定格式输出。 | 对结构化指令的关注度（Attention）不足。 | **结构化强校验** (Template Reinforcement) |

-----

## 3\. 第二阶段：大模型定向生成 (Generation)

利用 Teacher Model 的能力，通过 Prompt 工程生产针对性数据。

### 策略 1：场景泛化 (针对 OOD)

**核心逻辑**：保持核心 Output（如分类结果、提取的实体）不变，大幅改写 Input（变换场景、角色、语气）。

> **Prompt Template:**
> "作为数据增强专家，请将以下原始样本改写为 5 个新样本。
> 要求：
>
> 1.  **核心不变**：保持 Output 中的分类结果和推理逻辑完全一致。
> 2.  **变量重构**：将[医疗场景]替换为[金融/电商/法律场景]；将用户语气改为[急躁/啰嗦/专业]。
> 3.  **Input**: {{Original\_Input}}"

### 策略 2：CoT 思维链注入 (针对 Logic)

**核心逻辑**：显式地将推理步骤写在训练数据中，强迫模型学习“如何思考”。

> **Prompt Template:**
> "针对此复杂案例，请重写 Output。不要直接给出结论，必须在 JSON 的 `reasoning` 字段中按步骤拆解：
>
> 1.  引用 Input 中的关键事实 X。
> 2.  结合规则 Y 进行比对。
> 3.  发现矛盾点/符合点 Z。
> 4.  结论：给出最终判断。"

### 策略 3：对抗样本攻击 (针对 Ambiguity)

**核心逻辑**：生成“似是而非”的样本（Hard Negatives），例如语气像违规但实际合规的样本。

> **Prompt Template:**
> "生成一对对抗样本：
> 样本 A（负例）：语气非常像正例（包含敏感词、情绪激动），但逻辑内核完全合规。
> 样本 B（正例）：语气平和隐蔽，但核心逻辑包含违规信息。
> 目的：强迫模型忽略语气，只看逻辑内核。"

-----

## 4\. 第三阶段：三级质量验证 (Verification)

这是自动化流程中最关键的一环，防止“垃圾进，垃圾出”。

### 验证流水线 (Python 实现思路)

```python
import json
from sentence_transformers import SentenceTransformer, util

class QualityGuard:
    def __init__(self, teacher_llm_func):
        self.encoder = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.teacher_llm = teacher_llm_func # 调用GPT-4/DeepSeek的函数

    def check(self, original_case, generated_case):
        # L1: 句法层 (Syntax) - 100% 通过率要求
        try:
            data = json.loads(generated_case['output'])
            # 这里可以加 Pydantic 或 JSON Schema 校验
            if 'reasoning' not in data: return False
        except:
            return False

        # L2: 逻辑层 (Logic/Consistency) - Teacher 回测
        # 将生成的数据喂回 Teacher 模型，看其判断是否与 Label 一致
        check_prompt = f"Input: {generated_case['input']}\nOutput: {generated_case['output']}\n请判断Input和Output逻辑是否自洽？(Yes/No)"
        if "No" in self.teacher_llm(check_prompt):
            return False

        # L3: 价值层 (Diversity) - 向量相似度过滤
        # 太像原始数据无效(重复)，太不像可能是幻觉(OOD)
        orig_emb = self.encoder.encode(original_case['input'])
        gen_emb = self.encoder.encode(generated_case['input'])
        sim = util.cos_sim(orig_emb, gen_emb).item()
        
        # 也就是：必须要有一定差异，但不能完全离谱
        if sim > 0.90 or sim < 0.4: 
            return False

        return True
```

-----

## 5\. 第四阶段：混合与训练 (Mixing & Training)

### 1\. 数据混合策略

为了防止 **灾难性遗忘 (Catastrophic Forgetting)**，严禁只使用增强数据训练。

  * **混合公式**：
    $$Dataset_{final} = (Data_{aug} \times 70\%) + (Data_{original\_hq} \times 30\%)$$
  * **难例加权 (Upsampling)**：对于 "Logic/Hard" 类别的增强样本，建议在训练集中复制 2 份，增加其权重。

### 2\. LoRA 参数优化建议

针对修复特定 Bad Case（通常涉及逻辑修正），需要比常规微调更大的参数调整空间。

  * **Target Modules**: 不要只微调 Attention (`q_proj`, `v_proj`)。**必须覆盖 MLP 层** (`gate_proj`, `up_proj`, `down_proj`)，因为 MLP 层通常存储知识和逻辑模式。
  * **Rank (r)**: 建议设为 **64** (常规微调通常为 8 或 16)。
  * **Alpha**: 建议设为 **128** ($\alpha = 2r$)，增强适配器权重。
  * **Learning Rate**: 保持低位 **1e-5**，避免破坏原有通用能力。

-----

## 6\. 实施路线图

| 阶段 | 时间 | 关键动作 | 交付物 |
| :--- | :--- | :--- | :--- |
| **Day 1** | 诊断 | 收集 50-100 条 Bad Cases，人工标注错误类型（OOD/Logic/...）。 | 诊断报告 & 种子数据 |
| **Day 2** | 生成 | 编写 Prompt，跑通大模型批量生成（目标扩充至 1000 条）。 | 原始增强数据集 |
| **Day 3** | 清洗 | 运行 Python 验证脚本（L1/L2/L3），人工抽检前 50 条。 | 高质量训练集 |
| **Day 4** | 训练 | 混合 30% 原始高质量数据，开启 LoRA 全线性层微调。 | 微调后的 Adapter |
| **Day 5** | 验收 | 在测试集上验证，关注“修复率”和“通用能力保持率”。 | 评估报告 |
