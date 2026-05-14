import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VisualAdapter(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return self.net(x) * self.alpha + x


class TextualAdapter(nn.Module):
    def __init__(self, prompt_length=8, embed_dim=512):
        super().__init__()
        self.prompt_length = prompt_length
        self.soft_prompts = nn.Parameter(
            torch.randn(2, prompt_length, embed_dim)
        )

    def _soft_forward(self, clip_model, class_name, is_anomaly, device):
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-L-14")

        if is_anomaly:
            text = f"a photo of a defective {class_name}"
        else:
            text = f"a photo of {class_name}"

        idx = 1 if is_anomaly else 0

        tok_ids = tokenizer(text).to(device)

        with torch.no_grad():
            text_emb = clip_model.token_embedding(tok_ids)

        B, T, D = text_emb.shape
        P = self.prompt_length
        prompts = self.soft_prompts[idx].unsqueeze(0)

        if P + T > 77:
            keep = 77 - P
            if keep > 0:
                combined = torch.cat([prompts, text_emb[:, :keep, :]], dim=1)
            else:
                combined = prompts[:, :77, :]
        else:
            combined = torch.cat([prompts, text_emb], dim=1)

        S = combined.shape[1]
        with torch.no_grad():
            pos = clip_model.positional_embedding[:S]
            if pos.dim() == 3:
                pos = pos.squeeze(0)
        combined = combined + pos.unsqueeze(0)
        combined = combined.permute(1, 0, 2)

        attn_mask = getattr(clip_model, "attn_mask", None)
        try:
            if attn_mask is not None:
                combined = clip_model.transformer(combined, attn_mask=attn_mask)
            else:
                combined = clip_model.transformer(combined)
        except TypeError:
            combined = clip_model.transformer(combined)

        combined = combined.permute(1, 0, 2)
        combined = clip_model.ln_final(combined)

        with torch.no_grad():
            eos_idx = tok_ids.argmax(dim=-1)

        if eos_idx.shape[0] > 1:
            eos_idx = eos_idx[0:1]
        feat = combined[:, eos_idx, :].squeeze(1)
        feat = feat @ clip_model.text_projection
        return feat

    def forward(self, clip_model, class_name, device):
        normal = self._soft_forward(clip_model, class_name, False, device)
        anom = self._soft_forward(clip_model, class_name, True, device)
        return torch.stack([normal, anom], dim=0)


class PromptQueryAdapter(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.contextual = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, query_feat, memory_bank):
        if memory_bank.dim() == 1:
            memory_bank = memory_bank.unsqueeze(0)

        B = query_feat.shape[0]
        N = memory_bank.shape[0]

        q_exp = query_feat.unsqueeze(1).expand(-1, N, -1)
        m_exp = memory_bank.unsqueeze(0).expand(B, -1, -1)
        cat_feat = torch.cat([q_exp, m_exp], dim=-1)

        ctx = self.contextual(cat_feat)

        res = torch.cdist(
            query_feat.unsqueeze(1),
            memory_bank.unsqueeze(0),
            p=2.0,
        )
        res_dist = res.mean(dim=1, keepdim=True)

        score = torch.sigmoid(ctx.mean(dim=1) + res_dist * self.beta)
        return score


class AdaptCLIPModel(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.clip_model = clip_model
        self.device = device
        self.visual_adapter = VisualAdapter()
        self.textual_adapter = TextualAdapter()
        self.prompt_query_adapter = PromptQueryAdapter()

        self._patch_features = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        target = getattr(self.clip_model.visual, "ln_post", None)
        if target is None:
            return

        def hook(module, inp, out):
            self._patch_features = out.detach()

        self._hook_handle = target.register_forward_hook(hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, image, memory_bank, class_name):
        _ = self.clip_model.encode_image(image)
        patch_feats = self._patch_features
        if patch_feats is None:
            with torch.no_grad():
                _ = self.clip_model.encode_image(image)
                patch_feats = self._patch_features

        with torch.no_grad():
            global_feat = self.clip_model.encode_image(image)
        adapted = self.visual_adapter(global_feat)

        score = self.prompt_query_adapter(adapted, memory_bank)
        return score, patch_feats
