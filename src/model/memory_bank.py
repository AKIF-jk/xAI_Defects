import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank:
    def __init__(self, feat_dim=768, mode="global", patch_layer="ln_post"):
        self.feat_dim = feat_dim
        self.mode = mode
        self.patch_layer = patch_layer
        self.index = faiss.IndexFlatL2(feat_dim)
        self.patch_bank = []
        self._patch_features = None
        self._hook_handle = None
        self._n_stored = 0

        if mode == "hybrid":
            self.proj = nn.Linear(feat_dim * 2, feat_dim)
        else:
            self.proj = None
        self._proj_matrix = None

    def _register_hook(self, model):
        target = self._get_patch_hook_target(model)
        if target is None:
            return

        def hook(module, inp, out):
            self._patch_features = self._standardize_patch_features(out)

        self._hook_handle = target.register_forward_hook(hook)

    def _get_patch_hook_target(self, model):
        if self.patch_layer in (None, "ln_post"):
            return getattr(model.visual, "ln_post", None)

        layer = str(self.patch_layer)
        if layer.startswith("resblock:"):
            layer = layer.split(":", 1)[1]

        try:
            idx = int(layer)
        except ValueError:
            return getattr(model.visual, "ln_post", None)

        transformer = getattr(model.visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if blocks is None:
            return getattr(model.visual, "ln_post", None)
        if idx < -len(blocks) or idx >= len(blocks):
            return getattr(model.visual, "ln_post", None)
        return blocks[idx]

    @staticmethod
    def _standardize_patch_features(out):
        if isinstance(out, (tuple, list)):
            out = out[0]

        out = out.detach()
        if out.dim() == 3 and out.shape[0] > out.shape[1] and out.shape[0] > 32:
            out = out.permute(1, 0, 2)
        return out

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _extract_feature(self, global_feat, patch_feats):
        if self.mode == "global":
            return global_feat
        elif self.mode == "patch":
            patch_tokens = patch_feats[1:]
            feat = patch_tokens.mean(dim=0)
            if self._proj_matrix is not None:
                feat = feat @ self._proj_matrix.to(feat.device)
            return feat
        elif self.mode == "hybrid":
            patch_mean = patch_feats[1:].mean(dim=0)
            if self._proj_matrix is not None:
                patch_mean = patch_mean @ self._proj_matrix.to(patch_mean.device)
            concat = torch.cat([global_feat, patch_mean])
            if self.proj.weight.device != concat.device:
                self.proj = self.proj.to(concat.device)
            return self.proj(concat)
        else:
            return global_feat

    def build(self, model, dataloader, n_shots, device):
        self.index.reset()
        self.patch_bank = []
        self._patch_features = None
        self._n_stored = 0

        model_has_hook = hasattr(model, '_patch_features') and model._hook_handle is not None
        if not model_has_hook:
            self._register_hook(model)

        self._proj_matrix = getattr(model.visual, "proj", None)
        if self._proj_matrix is not None and isinstance(self._proj_matrix, nn.Parameter):
            self._proj_matrix = self._proj_matrix.detach()
        if self.proj is not None:
            self.proj = self.proj.to(device)

        count = 0
        for batch in dataloader:
            if len(batch) == 2:
                images, _ = batch
            else:
                images = batch[0]
            if count >= n_shots:
                break

            images = images.to(device)
            with torch.no_grad():
                global_feats = model.encode_image(images)

            patch_feats = model._patch_features if model_has_hook else self._patch_features

            for i in range(images.size(0)):
                if count >= n_shots:
                    break
                gf = global_feats[i]
                pf = patch_feats[i] if patch_feats is not None else None
                feat = self._extract_feature(gf, pf)
                feat_np = feat.detach().cpu().numpy().reshape(1, -1)
                self.index.add(feat_np)

                if pf is not None:
                    proj = getattr(model.visual, "proj", None)
                    patch_tokens = pf[1:]
                    if proj is not None:
                        patch_tokens = patch_tokens @ proj.detach().to(patch_tokens.device)
                    self.patch_bank.append(patch_tokens.cpu())
                self._n_stored += 1
                count += 1

        if not model_has_hook:
            self._remove_hook()
        return self._n_stored

    def query(self, query_feat, k=1):
        if isinstance(query_feat, torch.Tensor):
            query_feat = query_feat.detach().cpu().numpy()
        if query_feat.ndim == 1:
            query_feat = query_feat.reshape(1, -1)
        distances, indices = self.index.search(query_feat, k)
        return distances, indices

    def get_patch_bank(self):
        if not self.patch_bank:
            return torch.empty(0, self.feat_dim)
        return torch.cat(self.patch_bank, dim=0)

    @property
    def size(self):
        return self._n_stored

    def reset(self):
        self.index.reset()
        self.patch_bank = []
        self._n_stored = 0

    def encode(self, model, image, device):
        had_hook = self._hook_handle is not None
        if not had_hook:
            self._register_hook(model)

        with torch.no_grad():
            global_feat = model.encode_image(image)
        patch_feats = self._patch_features

        if self.mode == "global":
            feat = global_feat
        elif self.mode == "patch":
            patch_mean = patch_feats[:, 1:, :].mean(dim=1)
            if self._proj_matrix is not None:
                patch_mean = patch_mean @ self._proj_matrix.to(patch_mean.device)
            feat = patch_mean
        elif self.mode == "hybrid":
            patch_mean = patch_feats[:, 1:, :].mean(dim=1)
            if self._proj_matrix is not None:
                patch_mean = patch_mean @ self._proj_matrix.to(patch_mean.device)
            concat = torch.cat([global_feat, patch_mean], dim=-1)
            if self.proj.weight.device != concat.device:
                self.proj = self.proj.to(concat.device)
            feat = self.proj(concat)
        else:
            feat = global_feat

        if not had_hook:
            self._remove_hook()

        return feat
