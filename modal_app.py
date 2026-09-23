"""Open-model 'decision' server on Modal.

Serves any HF causal LM (default Qwen/Qwen3.5-4B, the SemIf/OpenJev baseline) and
answers typed decisions the way SemIf does: one forward pass, read the next-token
logits at the answer position, restrict them to the option letters, softmax.
No generation, no JSON parsing.

    modal deploy modal_app.py            # once
    modal run modal_app.py               # smoke test

Then from Python:  modal.Cls.from_name("jev-vs-open", "DirectLogits")().decide.remote(rows)
"""


import json
import time

import modal

APP_NAME = "jev-vs-open"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
GPU = "L4"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.10.0", "transformers>=5.17,<6", "accelerate>=1.12", "safetensors>=0.8", "hf-transfer", "flash-linear-attention")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface"})
)
hf_cache = modal.Volume.from_name("jev-vs-hf-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def messages(row: dict) -> list[dict]:
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [{"letter": LETTERS[i], "description": o["description"]} for i, o in enumerate(row["options"])],
    }
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


@app.cls(gpu=GPU, volumes={"/cache": hf_cache}, scaledown_window=300, timeout=600, max_containers=4)
class DirectLogits:
    model_name: str = modal.parameter(default=DEFAULT_MODEL)

    @modal.enter()
    def load(self):
        import torch
        import transformers

        cfg = transformers.AutoConfig.from_pretrained(self.model_name)
        self.tok = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.tok.padding_side = "left"
        cls = transformers.AutoModelForCausalLM
        if cfg.model_type in {"qwen3_5", "qwen3_5_text"}:  # multimodal checkpoint -> text tower only
            cls = transformers.Qwen3_5ForCausalLM
            cfg = cfg.get_text_config()
        self.model = cls.from_pretrained(self.model_name, config=cfg, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
        self.slots = {}
        for letter in LETTERS:
            ids = self.tok.encode(letter, add_special_tokens=False)
            assert len(ids) == 1, f"option letter {letter} is not a single token"
            self.slots[letter] = ids[0]
        hf_cache.commit()
        # Warm up kernels so the first real batch is not charged for compilation.
        self.decide.local([{"id": "warm", "state": "hello", "question": "Is this a greeting?", "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]}])

    @modal.method()
    def decide(self, rows: list, model: "str | None" = None, max_tokens: int = 8192) -> list:
        """Score a batch of SemIf-style rows: {id, state, question, options:[{id,description}]}"""
        import torch

        t_wall = time.perf_counter()
        prompts = []
        for row in rows:
            n_opts = len(row["options"])
            assert 2 <= n_opts <= len(LETTERS), f"{n_opts} options; max {len(LETTERS)}"
            p = self.tok.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(p)
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=max_tokens).to("cuda:0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        torch.cuda.synchronize()
        fwd = time.perf_counter() - t0
        out = []
        n_tok = enc["attention_mask"].sum(dim=1).tolist()
        for i, row in enumerate(rows):
            ids = [o["id"] for o in row["options"]]
            sel = logits[i, [self.slots[LETTERS[k]] for k in range(len(ids))]]
            probs = torch.softmax(sel, dim=0).tolist()
            out.append(
                {
                    "id": row["id"],
                    "option_ids": ids,
                    "probabilities": probs,
                    "input_tokens": int(n_tok[i]),
                    "forward_seconds": fwd / len(rows),
                    "wall_seconds_per_decision": (time.perf_counter() - t_wall) / len(rows),
                    "model": self.model_name,
                    "gpu": GPU,
                }
            )
        return out


@app.local_entrypoint()
def main():
    rows = [
        {
            "id": "smoke",
            "state": "Sehr geehrte Damen und Herren, anbei erhalten Sie die Rechnung für Ihre Bestellung.",
            "question": "Which language is this text written in?",
            "options": [{"id": "en", "description": "English"}, {"id": "de", "description": "German"}, {"id": "fr", "description": "French"}],
        }
    ]
    print(json.dumps(DirectLogits().decide.remote(rows), indent=2))
    many = {**rows[0], "id": "many", "options": [{"id": f"o{i}", "description": f"Option {i}"} for i in range(17)] + [{"id": "de", "description": "German"}]}
    print("18 options ->", DirectLogits().decide.remote([many])[0]["probabilities"][-1])
