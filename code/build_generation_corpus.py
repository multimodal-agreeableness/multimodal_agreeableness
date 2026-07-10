"""构建"生成式"SFT语料：Human=写作指令（与 generate_comparison.py 的推理prompt对齐），Assistant=真实高/低宜人性脚本。

区别于 Step1_agreeableness创建corpus202500210.py（那是"给脚本判断高低宜人性"的分类任务语料，
训练/推理prompt不对齐，且序列长度远超 sft.py 默认 max_seq_length=512，导致模型几乎学不到目标）。
这里让训练用的指令直接复用推理时会用到的 prompt，模型才能学到"看到这个指令 -> 生成这种风格的脚本"。
"""
import json
import random
from pathlib import Path

import pandas as pd

# 与 llama3/generate_comparison.py 的 PROMPTS 保持逐字一致，确保训练/推理分布对齐
PROMPTS = {
    "高宜人性": ("请作为短视频带货主播生成一段短视频带货脚本，脚本要求展示出【高宜人性】，销售【综合果干】。"
              "编写一段300-400字的【中文】短视频脚本，吸引观众互动和购买。（只要脚本内容，不要拍摄镜头等提示或指导）"
              "脚本内容要求展示产品特点和优点，突出主播的高宜人性。"
              "生成完整的一段短视频带货脚本，包括开场白、产品介绍、互动交流和促销信息。以下是生成内容："),
    "低宜人性": ("请作为短视频带货主播生成一段短视频带货脚本，脚本要求展示出【低宜人性】，销售【综合果干】。"
              "编写一段300-400字的【中文】短视频脚本，吸引观众互动和购买。（只要脚本内容，不要拍摄镜头等提示或指导）"
              "脚本内容要求展示产品特点和优点，突出主播的低宜人性。"
              "生成完整的一段短视频带货脚本，包括开场白、产品介绍、互动交流和促销信息。以下是生成内容："),
}

SAMPLE_SIZE = 500
SOURCES = [
    ("综合果干/【综合果干】直播带货-高agreeableness语料库【595条】.xlsx", "高宜人性"),
    ("综合果干/【综合果干】直播带货-低agreeableness语料库【630条】.xlsx", "低宜人性"),
]

TRAIN_RATIO = 0.8
SEED = 42


def main():
    ft_data_dir = Path(__file__).resolve().parent
    llama3_dir = ft_data_dir.parent / "llama3"

    rng = random.Random(SEED)
    train_records = []
    test_records = []

    for rel_path, label in SOURCES:
        df = pd.read_excel(ft_data_dir / rel_path)
        texts = df["text"].astype(str).drop_duplicates().head(SAMPLE_SIZE).tolist()
        if len(texts) < SAMPLE_SIZE:
            raise ValueError(f"{rel_path} 去重后仅剩 {len(texts)} 条，不足 {SAMPLE_SIZE} 条")

        records = [
            {"text": "### Human:  {} ### Assistant:  {}".format(PROMPTS[label], text)}
            for text in texts
        ]
        rng.shuffle(records)
        split_idx = int(len(records) * TRAIN_RATIO)
        train_records.extend(records[:split_idx])
        test_records.extend(records[split_idx:])
        print(f"{label}: {len(records)} 条 -> {split_idx} train / {len(records) - split_idx} test")

    rng.shuffle(train_records)
    rng.shuffle(test_records)

    with open(llama3_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(llama3_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"写入 {len(train_records)} 条训练样本, {len(test_records)} 条测试样本")
    print(f"train: {llama3_dir / 'train.jsonl'}")
    print(f"test: {llama3_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
