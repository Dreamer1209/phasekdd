"""Physiologically Constrained Alignment (PCA): dual semantic banks (frozen
source-negative B_S, online target-clinical B_T) with reliability-gated prototype
distribution alignment. Operates on the MST++ bottleneck feature."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


@torch.no_grad()
def cosine_kmeans(feats, n_clusters, n_iter=20, seed=0):
    """K-means with cosine distance.
    feats: [N, C']  ->  centroids: [n_clusters, C'] (L2-normalized)."""
    feats = l2norm(feats, dim=-1)
    N = feats.shape[0]
    n_clusters = min(n_clusters, N)
    g = torch.Generator(device=feats.device).manual_seed(seed)
    init_idx = torch.randperm(N, generator=g, device=feats.device)[:n_clusters]
    centroids = feats[init_idx].clone()
    for _ in range(n_iter):
        sim = feats @ centroids.t()                  # [N, n_clusters], cosine
        assign = sim.argmax(dim=1)
        new_centroids = centroids.clone()
        for k in range(n_clusters):
            mask = assign == k
            if mask.any():
                new_centroids[k] = l2norm(feats[mask].mean(dim=0), dim=0)
        centroids = new_centroids
    return l2norm(centroids, dim=-1)


@torch.no_grad()
def collect_teacher_features(ema_model, loader, device,
                             max_pixels=50000, pixels_per_image=2048):
    """Run the EMA teacher over `loader`, collect a subsample of flattened intermediate
    features f in R^{C'}. Returns [N, C'] on CPU. Used to initialize the prototype banks."""
    was_training = ema_model.training
    ema_model.eval()
    pool, total = [], 0
    for batch in loader:
        rgb = batch[0] if isinstance(batch, (list, tuple)) else batch
        rgb = rgb.to(device).float()
        _, feat = ema_model(rgb)                     # feat: [B, C', H', W']
        C = feat.shape[1]
        f = feat.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H'*W', C']
        if f.shape[0] > pixels_per_image:
            sel = torch.randperm(f.shape[0], device=f.device)[:pixels_per_image]
            f = f[sel]
        pool.append(f.cpu())
        total += f.shape[0]
        if total >= max_pixels:
            break
    if was_training:
        ema_model.train()
    return torch.cat(pool, dim=0)


class DualPrototypeBank(nn.Module):
    """Source Negative Bank (frozen) + Target Clinical Bank (online EMA)."""

    def __init__(self, feat_dim, num_target=16, num_source=64,
                 mu=0.9, tau_p=0.2, tau_g=0.1):
        super().__init__()
        self.K = num_target
        self.M = num_source
        self.mu = mu
        self.tau_p = tau_p
        self.tau_g = tau_g
        self.register_buffer("B_T", torch.zeros(num_target, feat_dim))   # online (clinical)
        self.register_buffer("B_S", torch.zeros(num_source, feat_dim))   # frozen (source negative)
        self.register_buffer("ready", torch.zeros(1))

    @property
    def initialized(self):
        return bool(self.ready.item() > 0)

    @torch.no_grad()
    def initialize(self, source_feats, target_feats):
        """source_feats/target_feats: [N, C'] EMA-teacher features (CPU or GPU)."""
        dev = self.B_S.device
        self.B_S.copy_(cosine_kmeans(source_feats.to(dev), self.M))
        self.B_T.copy_(cosine_kmeans(target_feats.to(dev), self.K))
        self.ready.fill_(1.0)

    @torch.no_grad()
    def update_target_bank(self, teacher_feats):
        """EMA update of B_T from unlabeled-target teacher features.
        teacher_feats: [N, C']. B_S stays frozen."""
        f = l2norm(teacher_feats, dim=-1)
        sim = f @ l2norm(self.B_T, dim=-1).t()       # [N, K] cosine
        assign = sim.argmax(dim=1)                   # nearest prototype
        for k in range(self.K):
            mask = assign == k
            if mask.any():
                mean_k = l2norm(f[mask].mean(dim=0), dim=0)
                self.B_T[k] = l2norm(self.mu * self.B_T[k] + (1.0 - self.mu) * mean_k, dim=0)

    def affinity(self, feats, bank):
        """Cosine affinity. feats: [N, C'], bank: [P, C'] -> [N, P]."""
        return l2norm(feats, dim=-1) @ l2norm(bank, dim=-1).t()


def pca_loss(feat_stu, feat_tea, bank: DualPrototypeBank):
    """Reliability-gated prototype distribution alignment (L_PCA).
    feat_stu/feat_tea: [B,C',H',W'] student/teacher feats on the unlabeled target
    (feat_tea detached). Returns (loss, mean gate)."""
    B, C, H, W = feat_stu.shape
    fs = feat_stu.permute(0, 2, 3, 1).reshape(-1, C)
    ft = feat_tea.permute(0, 2, 3, 1).reshape(-1, C)

    aT_stu = bank.affinity(fs, bank.B_T)                    # [N, K]
    aT_tea = bank.affinity(ft, bank.B_T)                    # [N, K]
    aS_tea = bank.affinity(ft, bank.B_S)                    # [N, M]

    # gate: high target affinity, low source affinity
    A_T = aT_tea.max(dim=1).values
    A_S = aS_tea.max(dim=1).values
    gate = torch.sigmoid((A_T - A_S) / bank.tau_g)

    q = F.softmax(aT_tea / bank.tau_p, dim=1).detach()
    log_p = F.log_softmax(aT_stu / bank.tau_p, dim=1)
    ce = -(q * log_p).sum(dim=1)
    loss = (gate * ce).mean()
    return loss, gate.mean().detach()
