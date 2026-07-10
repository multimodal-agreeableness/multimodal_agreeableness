# Custom Llama3 Agreeableness — LoRA Fine-Tuning Artifacts

本仓库存放论文 Web Appendix 9（"Text generation with Custom LLM"）对应研究工作的代码、训练数据与模型 checkpoint，用于证明该项工作的实际完成情况，供审稿/存档引用。

## 项目背景

以 `Meta-Llama-3-8B-Instruct` 为基座模型，用 LoRA 对其进行监督微调（SFT），使其能够根据指令生成展示"高宜人性"或"低宜人性"（agreeableness）特征的短视频带货脚本（产品：综合果干）。

## 目录结构

```
data/         用于 SFT 的训练/测试数据（pickle，Human/Assistant 指令-脚本配对格式）
code/         训练、生成、评估脚本
checkpoint/   LoRA adapter 权重（基于 Meta-Llama-3-8B-Instruct）
results/      生成对比结果与相似度评估结果
```

### data/

- `train.pkl`（800条）/ `test.pkl`（200条），Python pickle 格式，反序列化后是一个 dict 列表 `[{"text": "..."}, ...]`，各为高宜人性/低宜人性各半，按 80/20 分层切分。

### code/

- `sft.py` —— LoRA 监督微调训练脚本（基于 `trl.SFTTrainer`），包含 completion-only loss masking（只在 Assistant 回复部分计算 loss）；数据加载读取 `./train.pkl` / `./test.pkl`。
- `generate_comparison.py` —— 分别用原始基座模型和合并 LoRA adapter 后的模型，对"高宜人性"/"低宜人性"两组固定 prompt 各生成若干条带货脚本。


运行以上脚本均需要额外安装 `Meta-Llama-3-8B-Instruct` 基座模型权重（Meta 官方发布【增加连接】，遵循 Meta Llama 3 Community License，未随本仓库分发）以及 `transformers`/`trl`/`peft`/`torch` 等依赖（见 `requirements.txt`）。

### checkpoint/

LoRA adapter（`r=64, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj","v_proj"]`），训练配置：`per_device_train_batch_size=4, gradient_accumulation_steps=2, learning_rate=1.41e-5, num_train_epochs=3`

权重文件 `adapter_model.safetensors`



## License

- 代码与本仓库内数据：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)（署名-非商业性使用）。
- 基座模型 `Meta-Llama-3-8B-Instruct` 权重不包含在本仓库中，其使用需遵循 Meta 官方的 Llama 3 Community License。

详见 `LICENSE` 文件。
