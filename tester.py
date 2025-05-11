import os
import random
import csv
import json
import torch
import re
from PIL import Image
from dataclasses import dataclass
from collections import defaultdict

import openai
from dataloader import dataloader
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ————————————————————————————————————————————————————————————
# 1) Configuration
openai.api_key = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES_PER_TYPE = 10

# ————————————————————————————————————————————————————————————
# 2) ChatGPT-based prefix generator
_prefix_cache = {}
def generate_prefix(question: str) -> str:
    if question in _prefix_cache:
        return _prefix_cache[question]

    system_msg = {
        "role": "system",
        "content": (
            "You are a prefix making assistant. Do NOT answer the question in any way."
            "Given a single question, output a prefix that leads into the one-word English Answer to the question."
 
        )
    }
    user_msg = {"role": "user", "content": f"Question: \"{question}\"\nPrefix:"}
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_msg, user_msg],
            max_tokens=6,
            temperature=0.0,
            stop=["\n"]
        )
        prefix = resp.choices[0].message.content.strip().rstrip(".") + ":"
    except Exception:
        prefix = "Answer:"
    _prefix_cache[question] = prefix
    return prefix

# ————————————————————————————————————————————————————————————
# 3) Build prompt + prefix
def build_instruction(question: str, qtype: str) -> str:
    if qtype == "Object Identification":
        prompt = f"Answer with only one full English word or phrase. Dont output any numbers, spaces or anything else that is not a full word {question}"
    elif qtype == "Classification":
        prompt = f"Answer only with one full English word or phrase. Dont output any numbers, spaces or anything else that is not a full word {question}"
    elif qtype == "Color Recognition":
        prompt = f"Only respond with a color: {question}"
    elif qtype == "Yes/No":
        prompt = f"Answer yes **only if clearly visible**, otherwise say no. {question}"
    elif qtype == "Counting":
        prompt = f"Count and answer with a digit: {question}"
    else:
        prompt = f"Give a one-word answer: {question}"

    if qtype not in ("Yes/No", "Counting"):
        prefix = generate_prefix(question)
        return f"{prompt} {prefix}"
    else:
        return f"{prompt} Answer:"

# ————————————————————————————————————————————————————————————
# 4) LLaVA wrapper
@dataclass
class LlavaWrapper:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    device: torch.device = DEVICE

    def __post_init__(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            revision="a272c74"
        ).to(self.device)
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(self.model_id, revision="a272c74")

    def infer_top4(self, pil_img, instruction: str):
        full = f"USER: <image>\n{instruction} "
        inputs = self.proc(text=full, images=pil_img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        logits = out.logits[0, -1]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idxs = torch.argsort(probs, descending=True)[:10]
        raw = [self.proc.tokenizer.decode(i.item(), skip_special_tokens=True).strip() for i in idxs]

        # Clean top tokens
        bad = {"a", "the", "</s>", "", ".", ",", "?", "on", "in", "at", "and", "of", "to", "for"}
        filtered = [t for t in raw if t.lower() not in bad and len(t) > 2]
        top4 = filtered[:4]
        if len(top4) < 4:
            for t in raw:
                if t not in top4:
                    top4.append(t)
                if len(top4) == 4:
                    break

        return top4[0], top4

# ————————————————————————————————————————————————————————————
# 5) Group question indices by type
raw = dataloader.dataset.data
type_to_idxs = defaultdict(list)
for i, sample in enumerate(raw):
    type_to_idxs[sample.get("category", "<None>")].append(i)

# ————————————————————————————————————————————————————————————
# 6) Run model and write results to CSV
csv_path = "results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(["question_type", "instruction", "ground_truth", "top_4_tokens", "model_answer"])

    llava = LlavaWrapper()
    total = 0

    for qtype, idxs in type_to_idxs.items():
        if len(idxs) < NUM_SAMPLES_PER_TYPE:
            continue
        sampled = random.sample(idxs, NUM_SAMPLES_PER_TYPE)

        for idx in sampled:
            question, img_t, _, gt = dataloader.dataset[idx]

            # convert tensor to PIL
            if img_t.dim() == 3 and img_t.shape[0] == 3:
                pil = Image.fromarray((img_t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
            else:
                continue

            instr = build_instruction(question, qtype)
            answer, top4 = llava.infer_top4(pil, instr)

            writer.writerow([qtype, instr, gt, json.dumps(top4), answer])
            total += 1

print(f"✅ Done! Wrote {total} rows to {csv_path}")
