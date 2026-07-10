# Custom Llama3 Agreeableness — LoRA Fine-Tuning Artifacts

本仓库存放论文 Web Appendix 9（"Text generation with Custom LLM"）对应研究工作的代码、训练数据与模型 checkpoint，用于证明该项工作的实际完成情况，供审稿/存档引用。

## 项目背景

以 `Meta-Llama-3-8B-Instruct` 为基座模型，用 LoRA 对其进行监督微调（SFT），使其能够根据指令生成展示"高宜人性"或"低宜人性"（agreeableness）特征的短视频带货脚本（产品：综合果干）。随后分别用微调后模型（custom LLM）与未微调的原始模型（non-custom LLM）生成脚本，并用中文 BERT embedding 计算生成文本与"宜人性""果干"两个概念词的余弦相似度，做统计检验对比效果。

## 目录结构

```
data/         用于 SFT 的训练/测试数据（JSONL，Human/Assistant 指令-脚本配对格式）
code/         数据构建、训练、生成、评估脚本
checkpoint/   LoRA adapter 权重（基于 Meta-Llama-3-8B-Instruct）
results/      生成对比结果与相似度评估结果
```

### data/

- `train.pkl`（800条）/ `test.pkl`（200条），Python pickle 格式，反序列化后是一个 dict 列表 `[{"text": "..."}, ...]`，各为高宜人性/低宜人性各半，按 80/20 分层切分，train/test 之间无重复、无泄漏。
- 每条记录的 `text` 字段格式：`"### Human:  <生成指令> ### Assistant:  <真实高/低宜人性带货脚本>"`。
- 原始爬取语料（社交媒体带货脚本 xlsx/docx 原文）不包含在本次发布范围内，仅发布处理后的训练数据。

### code/

- `build_generation_corpus.py` —— 从原始标注语料（未随本仓库发布）构建上面 `data/` 里的 SFT 数据格式；保留在此仅为方法透明化记录，无法在缺少原始语料文件的情况下直接运行。
- `sft.py` —— LoRA 监督微调训练脚本（基于 `trl.SFTTrainer`），包含 completion-only loss masking（只在 Assistant 回复部分计算 loss）。**注意**：脚本内部数据加载写的是训练时实际用的 `datasets.load_dataset("json", ...)` 读取 `./train.jsonl` / `./test.jsonl`，而本仓库 `data/` 下发布的是 `.pkl` 格式，两者不直接匹配，如需运行需自行改成用 `pickle.load` 读取 `.pkl` 文件、或将 `.pkl` 转回 jsonl。
- `generate_comparison.py` —— 分别用原始基座模型和合并 LoRA adapter 后的模型，对"高宜人性"/"低宜人性"两组固定 prompt 各生成若干条带货脚本。
- `evaluate_agreeableness_similarity.py` —— 用 `bert-base-chinese` 计算生成文本与"宜人性""果干"的余弦相似度，并做 Welch t 检验。

运行以上脚本均需要额外安装 `Meta-Llama-3-8B-Instruct` 基座模型权重（Meta 官方发布，遵循 Meta Llama 3 Community License，未随本仓库分发）以及 `transformers`/`trl`/`peft`/`torch` 等依赖（见 `requirements.txt`）。

### checkpoint/

LoRA adapter（`r=64, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj","v_proj"]`），训练配置：`per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=1.41e-5, num_train_epochs=3`（有效 batch size = 8，共 300 steps，`train_loss` 收敛至约 1.96）。需配合 `Meta-Llama-3-8B-Instruct` 基座模型通过 `peft.PeftModel` 加载使用。

权重文件 `adapter_model.safetensors`（约105MB）超过 GitHub 单文件 100MB 的限制，通过 **Git LFS** 管理（见 `.gitattributes`），`git clone` 时会自动一并拉取，无需额外下载步骤。

### results/

- `generation_comparison_results.json` —— custom / non-custom 模型在高/低宜人性 prompt 下各生成 10 条（共 40 条）脚本原文。
- `agreeableness_similarity_results.csv` —— 对应的 BERT 相似度评分（宜人性/果干两个维度）。

**核心结果**（n=10 每组，Welch t-test）：

| 对比 | 结果 |
| --- | --- |
| 果干相关度：custom vs non-custom（高宜人性组，对应原论文 Table 5） | Custom 0.346 vs Non-custom 0.288，t=4.664，**p < 0.001** |
| 宜人性相似度：custom vs non-custom（高宜人性组，对应原论文 Table 4） | Custom 0.398 vs Non-custom 0.379，t=1.457，p = 0.168 |
| 操纵检验：custom 模型内部高 vs 低宜人性 | 高 0.398 vs 低 0.385，t=1.304，p = 0.210 |

## License

- 代码与本仓库内数据：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)（署名-非商业性使用）。
- 基座模型 `Meta-Llama-3-8B-Instruct` 权重不包含在本仓库中，其使用需遵循 Meta 官方的 Llama 3 Community License。

详见 `LICENSE` 文件。
