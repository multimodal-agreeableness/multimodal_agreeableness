import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROMPTS = {
    "high": ("请作为短视频带货主播生成一段短视频带货脚本，脚本要求展示出【高宜人性】，销售【综合果干】。"
              "编写一段300-400字的【中文】短视频脚本，吸引观众互动和购买。（只要脚本内容，不要拍摄镜头等提示或指导）"
              "脚本内容要求展示产品特点和优点，突出主播的高宜人性。"
              "生成完整的一段短视频带货脚本，包括开场白、产品介绍、互动交流和促销信息。以下是生成内容："),
    "low": ("请作为短视频带货主播生成一段短视频带货脚本，脚本要求展示出【低宜人性】，销售【综合果干】。"
             "编写一段300-400字的【中文】短视频脚本，吸引观众互动和购买。（只要脚本内容，不要拍摄镜头等提示或指导）"
             "脚本内容要求展示产品特点和优点，突出主播的低宜人性。"
             "生成完整的一段短视频带货脚本，包括开场白、产品介绍、互动交流和促销信息。以下是生成内容："),
}


def generate_for_model(model, tokenizer, model_type, num_samples, max_new_tokens, temperature, top_p):
    results = []
    for level, prompt in PROMPTS.items():
        for i in range(num_samples):
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to("cuda")
            output = model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                attention_mask=attention_mask,
            )
            generated_text = tokenizer.decode(output[:, input_ids.shape[1]:][0], skip_special_tokens=True)
            results.append({
                "model_type": model_type,
                "agreeableness_level": level,
                "sample_index": i,
                "generated_text": generated_text,
            })
            print(f"[{model_type}][{level}] sample {i} done")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter-path", default="llama3_lora_agreeableness_20260709")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", default="generation_comparison_results.json")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    all_results = []
    all_results += generate_for_model(model, tokenizer, "non_custom", args.num_samples, args.max_new_tokens,
                                       args.temperature, args.top_p)

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    model.eval()

    all_results += generate_for_model(model, tokenizer, "custom", args.num_samples, args.max_new_tokens,
                                       args.temperature, args.top_p)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
