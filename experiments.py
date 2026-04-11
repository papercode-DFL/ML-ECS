import os
import json
import math
import time
import copy
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import evaluate
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ====== 你的工程依赖 ======
from standalone_instruction import UrDataset, get_encoder, collate_fn, set_seed
from encoder.encoder_model import Encoder


# -------------------------
# Model blocks (keep yours)
# -------------------------
class CondSoftPrompt(nn.Module):
    """
    Split into:
      part1: LayerNorm + Linear1 -> fused_repr  (server will supervise clients)
      part2: GELU + Dropout + Linear2 -> prompt tokens
    """
    def __init__(self, enc_dim: int, mod_num:int, hidden_size: int,
                 num_soft_tokens: int = 16, mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.P = num_soft_tokens
        self.H = hidden_size

        in_dim = enc_dim * mod_num
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, mlp_hidden)  # <<< 第一层 linear 输出就是 fused_repr
        self.act = nn.GELU()
        self.dp = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_hidden, num_soft_tokens * hidden_size)

    def forward(self, fea_list: list) -> Tuple[torch.Tensor, torch.Tensor]:
        # fused input
        fused = torch.cat(fea_list, dim=-1)            # (B, mod_num*enc_dim)
        h = self.ln(fused)
        fused_repr = self.fc1(h)                       # (B, mlp_hidden)  << supervision signal

        out = self.fc2(self.dp(self.act(fused_repr)))  # (B, P*H)
        prompt = out.view(-1, self.P, self.H)          # (B, P, H)
        return prompt, fused_repr


class LLMCondSoftPromptInstruction(nn.Module):
    def __init__(
        self,
        llm,
        tokenizer,
        mod_dict:dict,
        enc_dim: int = 64,
        num_soft_tokens: int = 16,
        dropout: float = 0.1,
        freeze_non_lora_llm: bool = True,
        max_new_tokens: int = 4,
        enc_aft_fea = None
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.mod_dict = mod_dict
        self.mod_keys = list(self.mod_dict.keys())
        self.enc_dim = enc_dim
        self.num_soft_tokens = num_soft_tokens
        self.max_new_tokens = max_new_tokens



        H = llm.base_model.model.config.hidden_size if hasattr(llm, "base_model") else llm.config.hidden_size
        self.H = H
        self.enc_proj = nn.ModuleDict({m: nn.Linear(enc_aft_fea[m], 64)for m in self.mod_keys})
        # IMPORTANT: 每次构建 model 都会 new 一个 CondSoftPrompt
        self.cond_prompt = CondSoftPrompt(enc_dim=enc_dim, mod_num = len(self.mod_keys),hidden_size=H, num_soft_tokens=num_soft_tokens, mlp_hidden=256, dropout=dropout)

        # Freeze all non-LoRA params in LLM
        if freeze_non_lora_llm:
            for name, p in self.llm.named_parameters():
                if "lora_" not in name:
                    p.requires_grad = False

        # Ensure encoders and prompt MLP trainable (由工厂函数再控制 requires_grad)
        for mod in self.mod_keys:
            for p in self.mod_dict[mod].parameters():
                p.requires_grad = True

        for p in self.cond_prompt.parameters():
            p.requires_grad = True

        self.instr_tok = "<|Instruction|>"
        self.out_tok = "<|Output|>"

    def _build_prompt_text(self) -> str:
        return (
            f"{self.instr_tok} You are given multimodal sensor features. "
            f"Classify the sample into one of 0, 1, or 2. "
            f"Return only the label.\n"
            f"{self.out_tok}"
        )

    def _tokenize_batch(self, texts: List[str], device: torch.device):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _make_train_inputs(self, labels: torch.Tensor, device: torch.device):
        B = labels.size(0)
        prompt = self._build_prompt_text()
        targets = [f" {int(y.item())}{self.tokenizer.eos_token}" for y in labels]

        prompt_texts = [prompt for _ in range(B)]
        prompt_ids, prompt_attn = self._tokenize_batch(prompt_texts, device)
        tgt_ids, tgt_attn = self._tokenize_batch(targets, device)

        input_ids = torch.cat([prompt_ids, tgt_ids], dim=1)
        attention_mask = torch.cat([prompt_attn, tgt_attn], dim=1)

        lm_labels = torch.full_like(input_ids, -100)
        prompt_len = prompt_ids.size(1)
        lm_labels[:, prompt_len:] = tgt_ids
        lm_labels[attention_mask == 0] = -100
        return input_ids, attention_mask, lm_labels, prompt_len

    def _make_infer_inputs(self, B: int, device: torch.device):
        prompt = self._build_prompt_text()
        texts = [prompt for _ in range(B)]
        input_ids, attention_mask = self._tokenize_batch(texts, device)
        return input_ids, attention_mask

    def forward(self, batch: Dict[str, torch.Tensor], return_fused: bool = False):
        labels = batch["labels"]
        device = labels.device
        B = labels.size(0)

        mod_fea = []
        for mod in self.mod_keys:
            xa = batch[mod]
            z1 = self.mod_dict[mod](xa).view(B, -1)
            z1 = self.enc_proj[mod](z1)
            mod_fea.append(z1)

        prompt_embeds, fused_repr = self.cond_prompt(mod_fea)
        input_ids, attention_mask, lm_labels, prompt_len = self._make_train_inputs(labels, device)

        token_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)

        prompt_mask = torch.ones((B, self.num_soft_tokens), dtype=attention_mask.dtype, device=device)
        ext_attn = torch.cat([prompt_mask, attention_mask], dim=1)

        ext_labels = torch.full((B, self.num_soft_tokens), -100, dtype=lm_labels.dtype, device=device)
        ext_labels = torch.cat([ext_labels, lm_labels], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=ext_attn,
            labels=ext_labels,
            return_dict=True,
        )
        loss = out.loss

        if return_fused:
            return loss, fused_repr
        return loss

    @torch.no_grad()
    def greedy_decode_label(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        labels = batch["labels"]
        device = labels.device
        B = labels.size(0)


        mod_fea = []
        for mod in self.mod_keys:
            xa = batch[mod]
            z1 = self.mod_dict[mod](xa).view(B, -1)
            mod_fea.append(z1)

        prompt_embeds, _ = self.cond_prompt(mod_fea)
        input_ids, attention_mask = self._make_infer_inputs(B, device)

        token_embeds = self.llm.get_input_embeddings()(input_ids)
        cur_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)

        prompt_mask = torch.ones((B, self.num_soft_tokens), dtype=attention_mask.dtype, device=device)
        cur_attn = torch.cat([prompt_mask, attention_mask], dim=1)

        generated_ids = []
        eos_id = self.tokenizer.eos_token_id

        for _ in range(self.max_new_tokens):
            out = self.llm(inputs_embeds=cur_embeds, attention_mask=cur_attn, return_dict=True)
            next_id = out.logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id)

            next_embed = self.llm.get_input_embeddings()(next_id).unsqueeze(1)
            cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
            cur_attn = torch.cat([cur_attn, torch.ones((B, 1), dtype=cur_attn.dtype, device=device)], dim=1)

            if eos_id is not None and torch.all(next_id == eos_id):
                break

        gen = torch.stack(generated_ids, dim=1) if len(generated_ids) > 0 else torch.empty((B, 0), dtype=torch.long, device=device)

        prompt_text = self._build_prompt_text()
        preds = []
        for i in range(B):
            gen_text = self.tokenizer.decode(gen[i], skip_special_tokens=False)
            full = prompt_text + gen_text
            if self.out_tok in full:
                tail = full.split(self.out_tok, 1)[1]
            else:
                tail = full
            tail = tail.strip()
            pred_char = None
            for ch in tail:
                if ch in ["0", "1", "2"]:
                    pred_char = ch
                    break
            preds.append(pred_char if pred_char is not None else "NA")
        return preds


# -------------------------
# Config
# -------------------------
@dataclass
class Cfg:
    model_name: str = "../model_raw/MiniLLM-gpt2-720M"
    server_model_name: str = "/root/autodl-tmp/gpt-j-6b"

    # === paths (你需要按实际修改) ===
    # 3 clients each has private train/test
    client_train_paths: Tuple[str, str, str] = (
        "data/3client07/client1/train.pkl",
        "data/3client07/client2/train.pkl",
        "data/3client07/client3/train.pkl",
    )
    client_test_paths: Tuple[str, str, str] = (
        "data/3client07/client1/test.pkl",
        "data/3client07/client2/test.pkl",
        "data/3client07/client3/test.pkl",
    )
    client_public_train_paths: Tuple[str, str, str] = (
        "data/3client07/client1/pub_train.pkl",
        "data/3client07/client2/pub_train.pkl",
        "data/3client07/client3/pub_train.pkl",
    )
    client_public_test_paths: Tuple[str, str, str] = (
        "data/3client07/client1/pub_test.pkl",
        "data/3client07/client2/pub_test.pkl",
        "data/3client07/client3/pub_test.pkl",
    )

    # public train/test
    public_train_path: str = "data/3client07/server/train.pkl"
    public_test_path: str = "data/3client07/server/test.pkl"

    # === modality config ===
    mod_a: Optional[str] = None
    mod_b: Optional[str] = None
    enc_dim: int = 64
    num_soft_tokens: int = 16
    enc_aft_fea =  dict(acce = 64, rgb = 64, depth = 64)

    # === LoRA config ===
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # target_modules: Tuple[str, str] = ("c_attn", "c_proj")
    target_modules: Tuple[str, str] = ()

    # === FL/Train config ===
    num_clients: int = 3
    batch_size: int = 4
    comm_interval: int = 10       # every 10 batches -> one communication round
    rounds: int = 5              # MVP: run 1~2 rounds first
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    use_fp16: bool = True
    seed: int = 46

    # === experiment output ===
    exp_root: str = "./exp_runs"
    exp_name: str = "mvp_fl_run"


# -------------------------
# Helpers
# -------------------------
from peft import PeftModel

def get_client_modality_count_from_path(pkl_path: str) -> int:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    dataset = UrDataset(data)
    mod_list = dataset.get_all_mod()
    return len(mod_list)


def weighted_average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]):
    assert len(state_dicts) > 0, "state_dicts should not be empty"
    assert len(state_dicts) == len(weights), "weights length mismatch"

    total = sum(weights)
    norm_weights = [w / total for w in weights]

    avg_state = OrderedDict()
    keys = state_dicts[0].keys()

    for k in keys:
        avg_tensor = None
        for sd, w in zip(state_dicts, norm_weights):
            v = sd[k].detach().float()
            avg_tensor = v * w if avg_tensor is None else avg_tensor + v * w
        avg_state[k] = avg_tensor

    return avg_state


def aggregate_client_loras_by_modality_count(
    *,
    recv_llm,
    client_lora_dirs: List[str],
    client_weights: List[float],
):
    adapter_states = []

    for lora_dir in client_lora_dirs:
        tmp_model = PeftModel.from_pretrained(copy.deepcopy(recv_llm.base_model.model if hasattr(recv_llm, "base_model") else recv_llm), lora_dir)
        # 更稳妥一些：直接从 peft model 里取 trainable adapter state
        state = {k: v.detach().cpu() for k, v in tmp_model.state_dict().items() if "lora_" in k}
        adapter_states.append(state)
        del tmp_model
        torch.cuda.empty_cache()

    agg_state = weighted_average_state_dicts(adapter_states, client_weights)

    recv_state = recv_llm.state_dict()
    for k, v in agg_state.items():
        if k in recv_state:
            recv_state[k] = v.to(recv_state[k].device, dtype=recv_state[k].dtype)
    recv_llm.load_state_dict(recv_state, strict=False)

    return recv_llm

def gram_volume(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    vectors: (B, M, D)  M modalities (baseline + others)
    volume = sqrt(det(Gram)) ; Gram = V V^T  -> (B, M, M)
    Use Cholesky for stability: det(G) = prod(diag(L))^2 => sqrt(det)=prod(diag(L))
    """
    B, M, D = vectors.shape
    G = vectors @ vectors.transpose(-1, -2)  # (B,M,M)
    I = torch.eye(M, device=vectors.device, dtype=vectors.dtype).unsqueeze(0)
    G = G + eps * I
    L = torch.linalg.cholesky(G)             # (B,M,M)
    vol = torch.prod(torch.diagonal(L, dim1=-2, dim2=-1), dim=-1)  # (B,)
    return vol

def load_supervision_index(sup_path: str) -> Dict[Any, torch.Tensor]:
    sup = torch.load(sup_path, map_location="cpu")
    ids = sup["ids"]
    fused = sup["fused"]  # (N, mlp_hidden)
    idx = {ids[i]: fused[i] for i in range(len(ids))}
    return idx


def load_or_init_lora(
    *,
    base_model: nn.Module,
    lora_dir: str,
    lora_cfg: LoraConfig,
):
    """
    如果 lora_dir 存在 → load
    否则 → 用 base_model + lora_cfg 初始化
    """
    if os.path.exists(lora_dir):
        print(f"[LoRA] Load existing LoRA from {lora_dir}")
        model = PeftModel.from_pretrained(base_model, lora_dir)
    else:
        print(f"[LoRA] Init new LoRA")
        model = get_peft_model(base_model, lora_cfg)
    return model


def load_or_init_encoders(
    *,
    dataset: UrDataset,
    mod_list: List[str],
    enc_file: str,
):
    """
    如果 enc_dir 下存在 encoder.pt → load
    否则 → new encoder
    """

    encoders = build_encoders_for_mods(dataset, mod_list)

    if os.path.exists(enc_file):
        print(f"[Encoder] Load encoder from {enc_file}")
        state = torch.load(enc_file, map_location="cpu")
        encoders.load_state_dict(state, strict=False)
    else:
        print("[Encoder] Init new encoder")

    return encoders

def save_encoders(encoders: nn.ModuleDict, enc_dir: str):
    ensure_dir(enc_dir)
    torch.save(encoders.state_dict(), os.path.join(enc_dir, "encoder.pt"))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def dump_yaml(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def dump_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def build_lora_cfg(cfg: Cfg) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none"
    )

def build_tokenizer(cfg: Cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": ["<|Instruction|>", "<|Output|>"]}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def build_base_llm(cfg: Cfg, tokenizer):
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    base.config.pad_token_id = tokenizer.pad_token_id
    base.resize_token_embeddings(len(tokenizer))
    return base

def infer_modalities_from_dataset(dataset: UrDataset, cfg: Cfg):
    mod_list = dataset.get_all_mod()
    return mod_list

def build_encoders_for_mods(dataset: UrDataset, mod_list: List[str]) -> Dict[str, nn.Module]:
    enc_dict = nn.ModuleDict({
        m: get_encoder(Encoder, m, dataset.get_inp_mod_size(m))
        for m in mod_list
    })
    return enc_dict

def build_instruction_model(
    *,
    llm,
    tokenizer,
    mod_dict,
    cfg: Cfg,
    train_encoder: bool,
    train_prompt: bool,
    freeze_non_lora_llm: bool = True,
):
    model = LLMCondSoftPromptInstruction(
        llm=llm,
        tokenizer=tokenizer,
        mod_dict = mod_dict,
        enc_dim=cfg.enc_dim,
        num_soft_tokens=cfg.num_soft_tokens,
        dropout=0.1,
        freeze_non_lora_llm=freeze_non_lora_llm,
        max_new_tokens=4,
        enc_aft_fea = cfg.enc_aft_fea
    )

    for mod in model.mod_keys:
        for p in model.mod_dict[mod].parameters():
            p.requires_grad = bool(train_encoder)

    for p in model.cond_prompt.parameters():
        p.requires_grad = bool(train_prompt)

    if not train_encoder:
        for mod in model.mod_keys:
            model.mod_dict[mod].eval()

    return model

def build_optimizer(model: nn.Module, cfg: Cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return opt, params

def train_k_batches(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer,
    scaler,
    params,
    cfg: Cfg,
    k_batches: int,
    sup_index  = None
):
    model.train()
    it = 0
    for batch in train_loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(cfg.use_fp16 and device.type == "cuda")):
            loss = model(batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        it += 1
        if it >= k_batches:
            break

@torch.no_grad()
@torch.no_grad()
def eval_text_metrics(model: LLMCondSoftPromptInstruction, loader: DataLoader, device: torch.device):
    model.eval()

    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")

    preds_all, refs_all = [], []

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        preds = model.greedy_decode_label(batch)
        refs = [str(int(x)) for x in batch["labels"].tolist()]
        preds_all.extend(preds)
        refs_all.extend(refs)

    # 保持和原任务一致：label 任务仍然按文本计算
    rouge_scores = rouge_metric.compute(
        predictions=preds_all,
        references=refs_all,
        use_stemmer=True
    )

    bert_scores = bertscore_metric.compute(
        predictions=preds_all,
        references=refs_all,
        lang="en"
    )

    gen_acc = sum(int(p == t) for p, t in zip(preds_all, refs_all)) / max(1, len(refs_all))

    scores = {
        "acc": gen_acc,
        "rouge1": float(rouge_scores.get("rouge1", 0.0)),
        "rouge2": float(rouge_scores.get("rouge2", 0.0)),
        "rougeL": float(rouge_scores.get("rougeL", 0.0)),
        "rougeLsum": float(rouge_scores.get("rougeLsum", 0.0)),
        "bertscore": float(sum(bert_scores["f1"]) / max(1, len(bert_scores["f1"]))),
    }

    return scores, preds_all[:10], refs_all[:10]

def unload_model(model: nn.Module):
    del model
    torch.cuda.empty_cache()


# -------------------------
# Per-node runners
# -------------------------
def run_client_stage(
    *,
    client_id: int,
    round_id: int,
    stage: str,                 # "stage1_private" or "stage2_public"
    cfg: Cfg,
    tokenizer,
    device: torch.device,
    exp_dir: str,
):
    assert stage in ["stage1_public", "stage2_private"]

    # ---- select dataset ----
    if stage == "stage2_private":
        train_path = cfg.client_train_paths[client_id]
        test_path = cfg.client_test_paths[client_id]
        train_encoder = True
        train_prompt = True
    else:
        sup_path = os.path.join(exp_dir, "supervision", "_server_fused.pt")
        sup_index = load_supervision_index(sup_path)
        train_path = cfg.client_public_train_paths[client_id]
        test_path = cfg.client_public_test_paths[client_id]
        train_encoder = False
        train_prompt = True

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    train_dataset = UrDataset(train_data)

    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    test_dataset = UrDataset(test_data)

    mod_list = infer_modalities_from_dataset(train_dataset, cfg)
    # enc_dict = build_encoders_for_mods(train_dataset, mod_list)
    enc_file = os.path.join(
        exp_dir, "encoder", f"client{client_id}_enc.pt"
    )

    enc_dict = load_or_init_encoders(
        dataset=train_dataset,
        mod_list=mod_list,
        enc_file=enc_file,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ---- build model fresh (separate LoRA per stage) ----
    base = build_base_llm(cfg, tokenizer)
    llm = get_peft_model(base, build_lora_cfg(cfg))

    model = build_instruction_model(
        llm=llm,
        tokenizer=tokenizer,
        mod_dict=enc_dict,
        cfg=cfg,
        train_encoder=train_encoder,
        train_prompt=train_prompt,
        freeze_non_lora_llm=True,
    ).to(device)

    optimizer, params = build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_fp16 and device.type == "cuda"))

    # ---- train K batches (one communication interval) ----
    train_k_batches(
        model=model,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        params=params,
        cfg=cfg,
        k_batches=cfg.comm_interval,
    )

    # ---- eval rouge on test ----
    scores, pred_preview, ref_preview = eval_text_metrics(model, test_loader, device)

    # ---- save artifacts ----
    node_dir = os.path.join(exp_dir, f"client{client_id}", stage)
    ensure_dir(node_dir)

    # save config snapshot
    dump_yaml(
        {
            "client_id": client_id,
            "round_id": round_id,
            "stage": stage,
            "train_path": train_path,
            "test_path": test_path,
            "mod_list":mod_list,
            "train_encoder": train_encoder,
            "train_prompt": train_prompt,
            "cfg": asdict(cfg),
        },
        os.path.join(exp_dir, "configs", f"client{client_id}_{stage}.yaml"),
    )

    # save lora adapter
    lora_dir = os.path.join(exp_dir, "lora", f"client{client_id}_{stage}")
    ensure_dir(os.path.dirname(lora_dir))
    model.llm.save_pretrained(lora_dir)

    # save encoder ONLY for stage1 (private) because stage2 doesn't train encoder
    if stage == "stage1_private":
        enc_dir = os.path.join(exp_dir, "encoder")
        ensure_dir(enc_dir)
        torch.save(model.mod_dict.state_dict(), os.path.join(enc_dir, f"client{client_id}_enc.pt"))

    # save metrics
    metric_path = os.path.join(exp_dir, "metrics", f"client{client_id}_{stage}_round{round_id}.json")
    ensure_dir(os.path.dirname(metric_path))
    # dump_json(
    #     {
    #         "client_id": client_id,
    #         "round_id": round_id,
    #         "stage": stage,
    #         "score": scores,
    #         "preview": {"pred": pred_preview, "ref": ref_preview},
    #         "lora_dir": lora_dir,
    #     },
    #     metric_path,
    # )

    # ---- free memory ----
    unload_model(model)
    return scores, lora_dir

@torch.no_grad()
def build_server_supervision(
    model: LLMCondSoftPromptInstruction,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Expect batch contains batch["ids"] : (B,) or list[str]/list[int]
    Return dict with:
      ids: tensor/int64 or list
      fused: tensor (N, mlp_hidden)
    """
    model.eval()
    all_ids = []
    all_fused = []

    for batch in loader:
        for k in batch:
            if k == "ids":
                continue
            batch[k] = batch[k].to(device)

        ids = batch.get("ids", None)
        if ids is None:
            raise ValueError("Need batch['ids'] for supervision alignment.")

        # normalize ids to python list
        if torch.is_tensor(ids):
            ids_list = ids.detach().cpu().tolist()
        else:
            ids_list = list(ids)

        # forward to get fused_repr
        _, fused_repr = model(batch, return_fused=True)  # fused_repr: (B, mlp_hidden)
        all_ids.extend(ids_list)
        all_fused.append(fused_repr.detach().cpu())

    fused = torch.cat(all_fused, dim=0)  # (N, mlp_hidden)
    return {"ids": all_ids, "fused": fused}


def run_server_public(
    *,
    round_id: int,
    cfg: Cfg,
    tokenizer,
    device: torch.device,
    exp_dir: str,
    get_fusion_first=False,
    client_lora_dirs: Optional[List[str]] = None,
    client_modality_counts: Optional[List[int]] = None,
):
    # server uses public only
    base_recv = AutoModelForCausalLM.from_pretrained(cfg.model_name)  # client model
    recv_llm = get_peft_model(base_recv, build_lora_cfg(cfg))

    # ---- aggregate client LoRA into recv_llm ----
    if client_lora_dirs is not None and len(client_lora_dirs) > 0:
        if client_modality_counts is None:
            raise ValueError("client_modality_counts is required when client_lora_dirs is provided")

        recv_llm = aggregate_client_loras_by_modality_count(
            recv_llm=recv_llm,
            client_lora_dirs=client_lora_dirs,
            client_weights=client_modality_counts,
        )
    with open(cfg.public_train_path, "rb") as f:
        train_data = pickle.load(f)
    train_dataset = UrDataset(train_data)

    with open(cfg.public_test_path, "rb") as f:
        test_data = pickle.load(f)
    test_dataset = UrDataset(test_data)

    mod_list= infer_modalities_from_dataset(train_dataset, cfg)

    # server "full modality encoder": here we still build all encoders, but pick two for current model.
    # (your current model supports 2 modalities. Full-modality extension can be done later.)
    # enc_dict = build_encoders_for_mods(train_dataset, mod_list)
    enc_file = os.path.join(
        exp_dir, "encoder", f"server_enc.pt"
    )

    enc_dict = load_or_init_encoders(
        dataset=train_dataset,
        mod_list=mod_list,
        enc_file=enc_file,
    )


    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

# tokenizer需要更改
    # base = build_base_llm(cfg, tokenizer)
    # llm = get_peft_model(base, build_lora_cfg(cfg))

    # 原来：只有一个 llm
    # 现在：两个 llm

    # A. server 自用 llm（可以是另一个模型）
    base_server = AutoModelForCausalLM.from_pretrained(cfg.server_model_name)
    server_llm = get_peft_model(base_server, build_lora_cfg(cfg))

    # B. client-compatible llm（只用于 LoRA 聚合）
    base_recv = AutoModelForCausalLM.from_pretrained(cfg.model_name)  # client model
    recv_llm = get_peft_model(base_recv, build_lora_cfg(cfg))

    model = build_instruction_model(
        llm=server_llm,
        tokenizer=tokenizer,
        mod_dict = enc_dict,
        cfg=cfg,
        train_encoder=False,
        train_prompt=True,
        freeze_non_lora_llm=True,
    ).to(device)

    if get_fusion_first:
        sup_dir = os.path.join(exp_dir, "supervision")
        ensure_dir(sup_dir)

        sup = build_server_supervision(model, train_loader, device)
        sup_path = os.path.join(sup_dir, f"round{round_id}_server_fused.pt")
        torch.save(sup, sup_path)
        print(f"[Server] saved supervision -> {sup_path}")
        return

    optimizer, params = build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_fp16 and device.type == "cuda"))

    train_k_batches(
        model=model,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        params=params,
        cfg=cfg,
        k_batches=cfg.comm_interval
    )

    scores, pred_preview, ref_preview = eval_text_metrics(model, test_loader, device)
    print("Server Acc:{}".format(scores))

    dump_yaml(
        {
            "round_id": round_id,
            "stage": "server_public",
            "train_path": cfg.public_train_path,
            "test_path": cfg.public_test_path,
            "train_encoder": False,
            "train_prompt": True,
            "cfg": asdict(cfg),
        },
        os.path.join(exp_dir, "configs", f"server_public.yaml"),
    )

    lora_dir = os.path.join(exp_dir, "lora", f"server_public")
    ensure_dir(os.path.dirname(lora_dir))
    model.llm.save_pretrained(lora_dir)

    metric_path = os.path.join(exp_dir, "metrics", f"server_public_round{round_id}.json")
    ensure_dir(os.path.dirname(metric_path))
    # dump_json(
    #     {
    #         "round_id": round_id,
    #         "stage": "server_public",
    #         "score": scores,
    #         "preview": {"pred": pred_preview, "ref": ref_preview},
    #         "lora_dir": lora_dir,
    #     },
    #     metric_path,
    # )
    # ---- build & save supervision for clients ----
    sup_dir = os.path.join(exp_dir, "supervision")
    ensure_dir(sup_dir)

    sup = build_server_supervision(model, train_loader, device)
    sup_path = os.path.join(sup_dir, f"_server_fused.pt")
    torch.save(sup, sup_path)
    print(f"[Server] saved supervision -> {sup_path}")

    unload_model(model)
    return scores, lora_dir


# -------------------------
# Best tracking (by rougeL)
# -------------------------
def rougeLsum_of(scores: dict) -> float:
    # evaluate rouge keys usually include rouge1/rouge2/rougeL/rougeLsum
    return float(scores.get("rougeLsum", 0.0))


def main():
    cfg = Cfg()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = f"{cfg.exp_name}_{now_tag()}"
    exp_dir = os.path.join(cfg.exp_root, run_id)
    ensure_dir(exp_dir)
    ensure_dir(os.path.join(exp_dir, "configs"))
    ensure_dir(os.path.join(exp_dir, "metrics"))
    ensure_dir(os.path.join(exp_dir, "lora"))
    ensure_dir(os.path.join(exp_dir, "encoder"))
    ensure_dir(os.path.join(exp_dir, "logs"))

    # dump global cfg
    dump_yaml(asdict(cfg), os.path.join(exp_dir, "configs", "global.yaml"))

    tokenizer = build_tokenizer(cfg)

    # Track best (store best artifacts paths)
    best = {
        "client": {str(i): {"rougeLsum": -1, "bertscore":None} for i in range(cfg.num_clients)},
        "server": {"rougeLsum": -1, "bertscore":None},
    }

    for r in range(cfg.rounds):
        print(f"\n========== Communication Round {r} ==========")

        round_client_stage2_loras = []
        round_client_modality_counts = []

        # ---- clients (sequential) ----
        for cid in range(cfg.num_clients):
            # Initialize
            run_server_public(
                round_id=r, cfg=cfg, tokenizer=tokenizer,
                device=device, exp_dir=exp_dir, get_fusion_first=True
            )

            # stage1: public
            s1_scores, s1_lora_dir = run_client_stage(
                client_id=cid, round_id=r, stage="stage1_public",
                cfg=cfg, tokenizer=tokenizer, device=device, exp_dir=exp_dir
            )


            round_client_stage2_loras.append(s1_lora_dir)
            round_client_modality_counts.append(
                get_client_modality_count_from_path(cfg.client_public_train_paths[cid])
            )
            # stage 2: private
            s2_scores, s2_lora_dir = run_client_stage(
                client_id=cid, round_id=r, stage="stage2_private",
                cfg=cfg, tokenizer=tokenizer, device=device, exp_dir=exp_dir
            )
            s2_metric = rougeLsum_of(s2_scores)
            if s2_metric > best["client"][cid]["rougeLsum"]:
                best["client"][str(cid)]["bertscore"] = s2_scores['bertscore']
                best["client"][str(cid)]["rougeLsum"] = s2_metric

            # stage2: public


        # ---- server ----
        sv_scores, sv_lora_dir = run_server_public(
            round_id=r,
            cfg=cfg,
            tokenizer=tokenizer,
            device=device,
            exp_dir=exp_dir,
            client_lora_dirs=round_client_stage2_loras,
            client_modality_counts=round_client_modality_counts,
        )

        sv_metric = rougeLsum_of(sv_scores)
        if sv_metric > best["server"]["rougeLsum"]:
            best["server"]["rougeLsum"] = sv_metric
            best["server"]["bertscore"] = sv_scores['bertscore']

        dump_json(best, os.path.join(exp_dir, "metrics", "best_summary_{}.json".format(str(cfg.seed))))



if __name__ == "__main__":
    main()
