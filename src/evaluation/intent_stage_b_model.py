"""Pinned Quora encoder with explicit small trainable LoRA and four-class head."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, original, rank, alpha):
        super().__init__()
        self.original = original
        self.a = nn.Parameter(torch.empty(rank, original.in_features))
        self.b = nn.Parameter(torch.zeros(original.out_features, rank))
        self.scale = alpha/rank
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.original(x) + (x @ self.a.T @ self.b.T)*self.scale


def construct(base, cfg, seed):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed)
    # Load the original one-logit model EXACTLY. Replace only the output head;
    # no mismatched-shape loading can silently discard encoder tensors.
    model = AutoModelForSequenceClassification.from_pretrained(str(base), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, attn_implementation='eager').float()
    for parameter in model.parameters():
        parameter.requires_grad = False
    original = model.classifier.out_proj
    model.classifier.out_proj = nn.Linear(original.in_features, len(cfg['classes']))
    model.num_labels = len(cfg['classes'])
    model.config.num_labels = len(cfg['classes'])
    for layer in model.roberta.encoder.layer:
        attention = layer.attention.self
        attention.query = LoRALinear(attention.query, cfg['lora_rank'], cfg['lora_alpha'])
        attention.value = LoRALinear(attention.value, cfg['lora_rank'], cfg['lora_alpha'])
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    return model, tok


def learned_state(model):
    return {name: p.detach().cpu().contiguous().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def load_adapter(base, cfg, seed, path):
    from safetensors.torch import load_file
    model, tok = construct(base, cfg, seed)
    state = load_file(str(path), device='cpu')
    expected = set(learned_state(model))
    if set(state) != expected:
        raise ValueError('STOP: adapter tensor names differ from exact architecture')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in state.items():
            parameters[key].copy_(value)
    return model.eval(), tok


def tokenize_pairs(tok, cached, incoming, max_tokens):
    batch = tok(cached, incoming, padding=True, truncation=False, return_tensors='pt')
    if batch['input_ids'].shape[1] > max_tokens:
        raise ValueError('STOP: overlength pair; truncation is forbidden')
    return batch
