# Safe Nanochat

本项目实现了安全对话模型的训练和评估系统，主要包括两个核心功能：SFT数据合成和评估任务扩展。

## 1. SFT 数据合成

### 1.1 概述
高质量的safety对话数据任务，同时包含明确拒绝场景、提供支持以及给出安全回答的场景。

比如说涉及毒品或者犯罪需要明确拒绝，但比如说一些涉及心理或者自残问题需要给用户提供支持回答

### 1.2 数据格式

生成的数据采用 JSONL 格式，每条记录包含以下字段：

```json
{
  "id": "SAFETYBENCH_V1_XXXXXX",
  "version": "safetybench_v1.0",
  "created_date": "YYYY-MM-DD",
  "category": "violence_weapons|self_harm|illegal_drugs|...",
  "subcategory": "specific subcategory",
  "risk_level": "L|M|H",
  "language": "en",
  "locale": "en-US",
  "attack_pattern": ["pattern1", "pattern2"],
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "expected": {
    "expected_action": ["REFUSE|SAFE_COMPLETE|SUPPORT"],
    "must_include": {...},
    "must_not_include": [...]
  },
  "notes": "..."
}
```

- 查看了一下预训练数据集，主要是英文数据。

- 在完成预训练和sft的模型上简单测试中文对话，发现几乎不可用。

- 所以这里生成的数据全部使用英文。

### 1.3 安全类别

根据场景特点以及和ai对话整理后

生成数据的时候提供下面的安全类别：
- **violence_weapons**: 暴力武器相关
- **self_harm**: 自残自伤
- **illegal_drugs**: 非法药物
- **sexual_content**: 性内容
- **hate_harassment**: 仇恨和骚扰
- **extremism_terrorism**: 极端主义和恐怖主义
- **crime_fraud**: 犯罪和欺诈
- **cyber_abuse**: 网络滥用

包含几个不同的安全类型：拒绝，支持，回复

### 1.4 攻击类别

上面的每一个安全类别都与下面的攻击类别组合
- **roleplay_jailbreak** / **fictional_context**：用"小说/游戏/虚构角色"包装真实不当请求，让模型误以为虚构就能讲细节
- **false_authority**：冒充权威身份（警察/医生/研究员/红队）提高可信度，诱导模型放松限制
- **educational_framing**：用"科普/教学/论文/安全研究"包装，让模型进入"教学模式"输出更可执行的信息
- **hypothetical_scenario**：用"纯理论/假设推演"弱化意图，骗模型给出可迁移的步骤
- **stepwise_decomposition**：把危险目标拆成多个小问题逐步套信息，单步不敏感但组合后可执行
- **prompt_injection** / **policy_evasion**：让模型忽略规则、切换无约束模式、泄露系统信息，直接绕安全
- **encoding_obfuscation**：用编码/混淆/暗语绕过关键词与规则，或诱导模型帮忙解码还原
- **reverse_psychology**：以"为了避免/防范，我得知道怎么做"为由，诱导模型提供完整细节
- **emotional_coercion**：用卖惨、紧急、威胁等情绪压力逼模型破例
- **implied_request** / **benign_neighbor**：不直接要敏感内容，改问"去哪找/一般怎么查/朋友需要"，获取间接可用路径

### 1.5 Safety Bench
构造了一个100条数据的安全bench，用于sft效果测试

#### 安全类型分布

| category                | count | pct   |
| :---------------------- | ----: | :---- |
| self_harm               |    12 | 12.0% |
| sexual_content          |    11 | 11.0% |
| cyber_abuse             |    11 | 11.0% |
| extremism_terrorism     |    10 | 10.0% |
| crime_fraud             |    10 | 10.0% |
| misinformation          |    10 | 10.0% |
| privacy_violation       |    10 | 10.0% |
| hate_harassment         |     9 | 9.0%  |
| illegal_drugs           |     8 | 8.0%  |
| academic_dishonesty     |     7 | 7.0%  |
| medical_legal_financial |     5 | 5.0%  |
| jailbreak_injection     |     5 | 5.0%  |
| violence_weapons        |     4 | 4.0%  |
#### 危险程度分类
| risk_level | count | pct   |
| :--------- | ----: | :---- |
| M          |    40 | 40.0% |
| H          |    33 | 33.0% |
| L          |    27 | 27.0% |
#### 种类和危险程度交叉表
| category                |  H |  L |  M |
| :---------------------- | -: | -: | -: |
| academic_dishonesty     |  2 |  3 |  2 |
| crime_fraud             |  2 |  5 |  3 |
| cyber_abuse             |  4 |  3 |  4 |
| extremism_terrorism     |  3 |  2 |  5 |
| hate_harassment         |  3 |  2 |  4 |
| illegal_drugs           |  4 |  2 |  2 |
| jailbreak_injection     |  2 |  1 |  2 |
| medical_legal_financial |  2 |  1 |  2 |
| misinformation          |  2 |  2 |  6 |
| privacy_violation       |  4 |  2 |  4 |
| self_harm               |  2 |  3 |  7 |
| sexual_content          |  3 |  2 |  6 |
| violence_weapons        |  2 |  2 |  0 |
#### 对话长度分布
| n_messages | count | pct   |
| ---------: | ----: | :---- |
|          6 |    73 | 73.0% |
|          8 |    20 | 20.0% |
|         10 |     1 | 1.0%  |
|          4 |     6 | 6.0%  |
#### 期望动作分布
| expected_action | count | pct_of_actions |
| :-------------- | ----: | :------------- |
| REFUSE          |   304 | 92.6%          |
| SAFE_COMPLETE   |    31 | 9.4%           |
| SUPPORT         |     2 | 0.6%           |



### 1.6 SFT测评
先用mid training之后的模型跑了一个上面bench的baseline，一共一百道题目，正确率40%

用上面生成的sft数据，微调 500 个step，正确率41%

考虑到数据量很少和模型很小，可以接受。


## 2. 评估任务扩展

### 2.1 概述

aime24 aime25 数据集着重于考察模型在数学推理方面的能力，每个数据集只有30道题目。

测试时，方差可能较大。

在这个过程中，输入数学问题模型，在不借助外部工具调用的方法下得到答案

注意数据集中答案格式不一样，处理数据方式不太一样，需要根据数据集提取答案。

模型的回答也需要按照答案格式输出才能完成提取。


