import torch.nn as nn
import torch
import torch.nn.functional as F

class PixelMultiClassInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, max_samples=4096):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples  # to avoid huge pixel matrix

    def forward(self, emb, labels):
        B, C, H, W = emb.shape
        emb = emb.permute(0,2,3,1).reshape(B*H*W, C)  # (N, C)
        labels = labels.reshape(B*H*W)               # (N)

        # sub-sample pixels if too many
        if emb.shape[0] > self.max_samples:
            idx = torch.randperm(emb.shape[0], device=emb.device)[:self.max_samples]
            emb, labels = emb[idx], labels[idx]

        emb = F.normalize(emb, dim=1)

        logits = emb @ emb.T  # similarity matrix (N, N)
        logits /= self.temperature

        # mask out self-similarity
        self_mask = torch.eye(logits.size(0), device=logits.device).bool()
        logits = logits.masked_fill(self_mask, -1e9)

        loss = 0
        classes = torch.unique(labels)

        for cls in classes:
            pos_mask = (labels == cls)
            neg_mask = (labels != cls)

            # positives: similarity to same-class pixels
            pos_logits = logits[pos_mask][:, pos_mask]
            neg_logits = logits[pos_mask][:, neg_mask]

            # InfoNCE: log( sum(exp(sim_pos)) / sum(exp(sim_all)) )
            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)

            cls_loss = -torch.log(
                pos_exp.sum(dim=1) / (pos_exp.sum(dim=1) + neg_exp.sum(dim=1) + 1e-6)
            ).mean()

            loss += cls_loss

        return loss / len(classes)