"""
CS550 Final Project: Movie Recommender System
Dataset : MovieLens Small (ml-latest-small)

Models
------
[A] FlowNeuMF  – Neural Collaborative Filtering whose user/item
    representations are learned via Conditional Flow Matching
    (Lipman et al., NeurIPS 2022).  A velocity-field network
    v_θ(xₜ, u, i, t) is trained to transport Gaussian noise to the
    GMF target embedding via straight-line ODE paths, regularising
    the embedding space while a NeuMF-style MLP head predicts ratings.

[B] ClassicNeuMF – Standard NeuMF baseline without flow matching
    (He et al., WWW 2017), trained with the same embedding size,
    MLP tower, optimizer, and evaluation protocol as FlowNeuMF.

[C] BiasedMF – Classic matrix factorization with user/item biases.

[D] ItemCF – Item-based collaborative filtering with cosine similarity.

Metrics: MAE, RMSE  |  Precision@10, Recall@10, F-measure@10, NDCG@10
"""

import argparse
import math, os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = "./data/ml-latest-small"
DEFAULT_PAPER_DIR = "/research/cbim/vast/sf895/code/Rutgers/cs550/movielens_acm_paper"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

LARGE_DATASET_THRESHOLD = 5_000_000
FLOW_HIGHLIGHT = "#D62728"
BASELINE_BLUE = "#4C78A8"
BASELINE_ORANGE = "#F58518"
BASELINE_GREEN = "#54A24B"


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def style_barplot(ax, labels, values, title, ascending_good=False, highlight_label="FlowNeuMF",
                  yscale=None):
    x = np.arange(len(labels))
    colors, edgecolors, linewidths = [], [], []
    for label in labels:
        if label == highlight_label:
            colors.append("#FADBD8")
            edgecolors.append(FLOW_HIGHLIGHT)
            linewidths.append(2.5)
        else:
            colors.append(BASELINE_BLUE)
            edgecolors.append("black")
            linewidths.append(0.8)
    ax.bar(x, values, color=colors, edgecolor=edgecolors, linewidth=linewidths)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if yscale == "log" and np.all(np.asarray(values) > 0):
        ax.set_yscale("log")
    elif yscale == "symlog":
        positive_vals = np.asarray([v for v in values if v > 0], dtype=float)
        linthresh = max(positive_vals.min() / 2, 1e-4) if len(positive_vals) else 1e-4
        ax.set_yscale("symlog", linthresh=linthresh)
    if ascending_good:
        ax.invert_yaxis()

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Data utilities
# ═════════════════════════════════════════════════════════════════════════════

def load_data(data_path):
    ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(data_path, "movies.csv"))
    return ratings, movies


def train_test_split(ratings: pd.DataFrame, test_ratio=0.2, seed=42):
    """Per-user 80/20 hold-out split."""
    test = ratings.groupby("userId", group_keys=False).sample(frac=test_ratio, random_state=seed)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


class RatingDataset(Dataset):
    def __init__(self, df, u2i, m2i):
        # Vectorized mapping using pandas .map() - 100x faster than list comprehension
        self.u = torch.from_numpy(df.userId.map(u2i).values).long()
        self.i = torch.from_numpy(df.movieId.map(m2i).values).long()
        self.r = torch.from_numpy(df.rating.values).float()
    def __len__(self): return len(self.r)
    def __getitem__(self, k): return self.u[k], self.i[k], self.r[k]


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FlowNeuMF / ClassicNeuMF
# ═════════════════════════════════════════════════════════════════════════════

class SiLUMLP(nn.Module):
    """MLP with LayerNorm + SiLU, used for the velocity field."""
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        dims, layers = [in_dim] + hidden, []
        for a, b in zip(dims, dims[1:] + [out_dim]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers[:-3])   # strip final norm/act/drop
    def forward(self, x): return self.net(x)


class ReluMLP(nn.Module):
    """Standard ReLU MLP used for the NeuMF interaction tower."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        dims, layers = [in_dim] + hidden, []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1] if hidden else in_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class NeuMFBase(nn.Module):
    """Shared NeuMF backbone used by both classic and flow-matching variants."""
    def __init__(self, n_users, n_items, emb_dim=64,
                 mlp_hidden=None):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [128, 64]
        self.emb_dim   = emb_dim

        self.ue_gmf = nn.Embedding(n_users, emb_dim)
        self.ie_gmf = nn.Embedding(n_items, emb_dim)
        self.ue_mlp = nn.Embedding(n_users, emb_dim)
        self.ie_mlp = nn.Embedding(n_items, emb_dim)
        for emb in (self.ue_gmf, self.ie_gmf, self.ue_mlp, self.ie_mlp):
            nn.init.normal_(emb.weight, 0, 0.01)

        self.mlp_tower = ReluMLP(2 * emb_dim, mlp_hidden, mlp_hidden[-1])
        self.rating_head = nn.Linear(emb_dim + mlp_hidden[-1], 1)

    def _embed(self, uid, mid):
        ug = self.ue_gmf(uid); ig = self.ie_gmf(mid)
        um = self.ue_mlp(uid); im = self.ie_mlp(mid)
        return ug, ig, um, im

    def _rate(self, x1, um, im):
        mlp_out = self.mlp_tower(torch.cat([um, im], dim=-1))
        return self.rating_head(torch.cat([x1, mlp_out], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict_rating(self, uid, mid):
        return self.predict_score(uid, mid).clamp(1.0, 5.0)

    def predict_score(self, uid, mid):
        ug, ig, um, im = self._embed(uid, mid)
        x1 = ug * ig
        return self._rate(x1, um, im)

    def bpr_loss(self, uid, pos_mid, neg_mid):
        pos_scores = self.predict_score(uid, pos_mid)
        neg_scores = self.predict_score(uid, neg_mid)
        return F.softplus(-(pos_scores - neg_scores)).mean()


class ClassicNeuMF(NeuMFBase):
    """Standard NeuMF baseline from He et al. (WWW 2017)."""

    def forward(self, uid, mid, r):
        ug, ig, um, im = self._embed(uid, mid)
        x1 = ug * ig
        r_hat     = self._rate(x1, um, im)
        return F.mse_loss(r_hat, r)


class FlowNeuMF(NeuMFBase):
    """
    NeuMF backbone plus Conditional Flow Matching regularization.
    The only difference from ClassicNeuMF is the added velocity field
    and the flow loss in training.
    """

    def __init__(self, n_users, n_items, emb_dim=64,
                 mlp_hidden=None, ode_steps=10, lambda_r=1.0,
                 flow_weight=0.15, consistency_weight=0.5,
                 velocity_hidden=None):
        super().__init__(n_users, n_items, emb_dim=emb_dim, mlp_hidden=mlp_hidden)
        self.ode_steps = ode_steps
        self.lambda_r  = lambda_r
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight
        if velocity_hidden is None:
            velocity_hidden = [128, 128]
        self.velocity = SiLUMLP(3 * emb_dim + 1, velocity_hidden, emb_dim)

    def _velocity(self, xt, ug, ig, t):
        inp = torch.cat([xt, ug, ig, t], dim=-1)
        return self.velocity(inp)

    def forward(self, uid, mid, r):
        ug, ig, um, im = self._embed(uid, mid)
        x1 = ug * ig

        x0 = torch.randn_like(x1)
        t  = torch.rand(x1.size(0), 1, device=x1.device)
        xt = (1.0 - t) * x0 + t * x1
        v_pred   = self._velocity(xt, ug, ig, t)
        loss_cfm = F.mse_loss(v_pred, x1 - x0)
        x1_hat = xt + (1.0 - t) * v_pred
        loss_consistency = F.mse_loss(x1_hat, x1)

        r_hat     = self._rate(x1, um, im)
        loss_rate = F.mse_loss(r_hat, r)
        return (
            self.lambda_r * loss_rate
            + self.flow_weight * loss_cfm
            + self.consistency_weight * loss_consistency
        )

    @torch.no_grad()
    def predict_stochastic(self, uid, mid):
        """Transport noise → preference via Euler integration of v_θ."""
        ug, ig, um, im = self._embed(uid, mid)
        x = torch.randn_like(ug)
        dt = 1.0 / self.ode_steps
        for step in range(self.ode_steps):
            t = torch.full((x.size(0), 1), step * dt, device=x.device)
            x = x + self._velocity(x, ug, ig, t) * dt
        return self._rate(x, um, im).clamp(1.0, 5.0)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Additional baselines
# ═════════════════════════════════════════════════════════════════════════════

class BiasedMF(nn.Module):
    """Biased matrix factorization for explicit rating prediction."""

    def __init__(self, n_users, n_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, uid, mid, r):
        pred = self._predict(uid, mid, clamp_output=False)
        return F.mse_loss(pred, r)

    def set_global_bias(self, mean_rating):
        with torch.no_grad():
            self.global_bias.fill_(float(mean_rating))

    def _predict(self, uid, mid, clamp_output=True):
        dot = (self.user_emb(uid) * self.item_emb(mid)).sum(dim=-1)
        pred = dot + self.user_bias(uid).squeeze(-1) + self.item_bias(mid).squeeze(-1) + self.global_bias
        return pred.clamp(1.0, 5.0) if clamp_output else pred

    @torch.no_grad()
    def predict_rating(self, uid, mid):
        return self._predict(uid, mid, clamp_output=True)

    def predict_score(self, uid, mid):
        return self._predict(uid, mid, clamp_output=False)

    def bpr_loss(self, uid, pos_mid, neg_mid):
        pos_scores = self.predict_score(uid, pos_mid)
        neg_scores = self.predict_score(uid, neg_mid)
        return F.softplus(-(pos_scores - neg_scores)).mean()


class ItemCF:
    """Item-based collaborative filtering using cosine similarity over centered ratings."""

    def __init__(self, k=100):
        self.k = k

    def fit(self, train_df, u2i, m2i, users, items):
        self.mu = float(train_df.rating.mean())
        self.u2i = u2i
        self.m2i = m2i
        self.users = np.array(users)
        self.items = np.array(items)

        nu, ni = len(users), len(items)
        R = np.full((nu, ni), np.nan, dtype=np.float32)
        user_idx = train_df.userId.map(u2i).to_numpy()
        item_idx = train_df.movieId.map(m2i).to_numpy()
        R[user_idx, item_idx] = train_df.rating.to_numpy(dtype=np.float32)

        self.umean = np.nanmean(R, axis=1)
        self.umean = np.where(np.isnan(self.umean), self.mu, self.umean)
        Rc = np.where(np.isnan(R), 0.0, R - self.umean[:, None]).astype(np.float32)
        norms = np.linalg.norm(Rc, axis=0, keepdims=True)
        norms[norms == 0] = 1e-9
        normed = Rc / norms
        self.sim = (normed.T @ normed).astype(np.float32)
        np.fill_diagonal(self.sim, 0.0)
        self.Rc = Rc
        self.rated_mask = R == R
        self.item_popularity = np.sum(self.rated_mask, axis=0)

    def predict(self, uid, mid):
        u = self.u2i.get(uid)
        i = self.m2i.get(mid)
        if u is None:
            return self.mu
        if i is None:
            return float(self.umean[u])

        rated = self.rated_mask[u]
        sims = self.sim[i].copy()
        sims[~rated] = 0.0
        if sims.sum() == 0:
            return float(self.umean[u])
        k = min(self.k, int((sims > 0).sum()))
        if k == 0:
            return float(self.umean[u])
        top_idx = np.argpartition(sims, -k)[-k:]
        weights = sims[top_idx]
        mask = weights > 0
        top_idx = top_idx[mask]
        weights = weights[mask]
        if len(top_idx) == 0:
            return float(self.umean[u])
        centered = self.Rc[u, top_idx]
        pred = self.umean[u] + np.dot(weights, centered) / np.abs(weights).sum()
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, df):
        return np.array([self.predict(row.userId, row.movieId) for row in df.itertuples()], dtype=np.float32)

    def score_all_items(self, uid):
        u = self.u2i.get(uid)
        if u is None:
            popularity_scores = self.item_popularity.astype(np.float32)
            return popularity_scores / max(popularity_scores.max(), 1)

        user_mean = self.umean[u]
        rated = self.rated_mask[u]
        centered = self.Rc[u]
        scores = np.full(len(self.items), user_mean, dtype=np.float32)

        rated_idx = np.flatnonzero(rated)
        if len(rated_idx) == 0:
            return scores

        for item_idx in range(len(self.items)):
            sims = self.sim[item_idx, rated_idx]
            k = min(self.k, len(sims))
            if k == 0:
                continue
            top = np.argpartition(sims, -k)[-k:]
            weights = sims[top]
            pos = weights > 0
            if not np.any(pos):
                continue
            weights = weights[pos]
            neigh_idx = rated_idx[top[pos]]
            scores[item_idx] = user_mean + np.dot(weights, centered[neigh_idx]) / np.abs(weights).sum()
        return np.clip(scores, 1.0, 5.0)


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# 4.  Metrics
# ═════════════════════════════════════════════════════════════════════════════

def mae_rmse(preds, actuals):
    e = np.asarray(preds) - np.asarray(actuals)
    return float(np.mean(np.abs(e))), float(math.sqrt(np.mean(e ** 2)))


def ndcg_at_k(rec, rel, k):
    dcg   = sum(1 / math.log2(r+2) for r, m in enumerate(rec[:k]) if m in rel)
    ideal = sum(1 / math.log2(r+2) for r in range(min(len(rel), k)))
    return dcg / ideal if ideal > 0 else 0.0


def neural_topn(model, u2i, m2i, item_arr, train_df, test_df, n=10, batch=512, sample_users=None):
    model.eval()
    seen  = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    relev = test_df.groupby("userId")["movieId"].apply(set).to_dict()

    # Sample users if specified (for large datasets)
    if sample_users and sample_users < len(relev):
        import random
        sampled_uids = random.sample(list(relev.keys()), sample_users)
        relev = {uid: relev[uid] for uid in sampled_uids}
        print(f"  Sampled {sample_users} users from {len(test_df.userId.unique())} for evaluation")

    all_iidx = torch.tensor([m2i.get(m, 0) for m in item_arr], dtype=torch.long, device=DEVICE)
    Ps, Rs, Fs, Ns = [], [], [], []
    total = len(relev)
    print(f"  Generating Top-{n} for {total} users …")
    for idx, (uid, rel) in enumerate(relev.items()):
        if idx % 100 == 0: print(f"    {idx}/{total}")
        uidx = u2i.get(uid)
        if uidx is None: continue
        mask  = np.array([m not in seen.get(uid, set()) for m in item_arr], dtype=bool)
        c_mid = item_arr[mask]
        c_iidx = all_iidx[torch.tensor(mask, dtype=torch.bool)]
        if len(c_mid) == 0: continue
        uid_t = torch.full((len(c_mid),), uidx, dtype=torch.long, device=DEVICE)
        scores = []
        for s in range(0, len(c_mid), batch):
            scores.append(model.predict_rating(uid_t[s:s+batch], c_iidx[s:s+batch]).cpu().numpy())
        scores = np.concatenate(scores)
        k_ = min(n, len(c_mid))
        ti = np.argpartition(scores, -k_)[-k_:]
        ti = ti[np.argsort(scores[ti])[::-1]]
        top = c_mid[ti].tolist(); hits = set(top) & rel
        p = len(hits)/n; r = len(hits)/len(rel) if rel else 0
        f = 2*p*r/(p+r) if p+r else 0
        Ps.append(p); Rs.append(r); Fs.append(f)
        Ns.append(ndcg_at_k(top, rel, n))
    return float(np.mean(Ps)), float(np.mean(Rs)), float(np.mean(Fs)), float(np.mean(Ns))


def numpy_topn(score_fn, item_arr, train_df, test_df, n=10, sample_users=None):
    seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    relev = test_df.groupby("userId")["movieId"].apply(set).to_dict()

    if sample_users and sample_users < len(relev):
        import random
        sampled_uids = random.sample(list(relev.keys()), sample_users)
        relev = {uid: relev[uid] for uid in sampled_uids}
        print(f"  Sampled {sample_users} users from {len(test_df.userId.unique())} for evaluation")

    Ps, Rs, Fs, Ns = [], [], [], []
    total = len(relev)
    print(f"  Generating Top-{n} for {total} users …")
    for idx, (uid, rel) in enumerate(relev.items()):
        if idx % 100 == 0:
            print(f"    {idx}/{total}")
        scores = score_fn(uid)
        mask = np.array([m not in seen.get(uid, set()) for m in item_arr], dtype=bool)
        cand_items = item_arr[mask]
        cand_scores = scores[mask]
        if len(cand_items) == 0:
            continue
        k_ = min(n, len(cand_items))
        top_idx = np.argpartition(cand_scores, -k_)[-k_:]
        top_idx = top_idx[np.argsort(cand_scores[top_idx])[::-1]]
        top = cand_items[top_idx].tolist()
        hits = set(top) & rel
        p = len(hits) / n
        r = len(hits) / len(rel) if rel else 0
        f = 2 * p * r / (p + r) if p + r else 0
        Ps.append(p); Rs.append(r); Fs.append(f)
        Ns.append(ndcg_at_k(top, rel, n))
    return float(np.mean(Ps)), float(np.mean(Rs)), float(np.mean(Fs)), float(np.mean(Ns))


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Training loop
# ═════════════════════════════════════════════════════════════════════════════

def train_model(model, model_name, train_df, u2i, m2i,
                n_epochs=60, batch_size=32768, lr=1e-3,
                ranking_weight=0.0, positive_threshold=4.0,
                user_seen_items=None, n_items=None):
    ds     = RatingDataset(train_df, u2i, m2i)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                       pin_memory=True)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    model.train()
    history = []
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        total = 0.0
        total_rank = 0.0
        total_rank_examples = 0
        for u, i, r in loader:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            loss = model(u, i, r)
            if ranking_weight > 0 and user_seen_items is not None and n_items is not None:
                pos_mask = r >= positive_threshold
                if pos_mask.any():
                    pos_u = u[pos_mask]
                    pos_i = i[pos_mask]
                    neg_i = sample_negative_items(pos_u, user_seen_items, n_items)
                    rank_loss = model.bpr_loss(pos_u, pos_i, neg_i)
                    loss = loss + ranking_weight * rank_loss
                    total_rank += rank_loss.item() * int(pos_mask.sum().item())
                    total_rank_examples += int(pos_mask.sum().item())
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(r)
        sched.step()
        epoch_loss = total / len(ds)
        history.append({
            "model": model_name,
            "epoch": epoch,
            "loss": epoch_loss,
            "ranking_loss": total_rank / max(total_rank_examples, 1),
            "time_min": (time.time() - epoch_start) / 60.0,
        })
        if epoch % 10 == 0 or epoch == 1:
            rank_msg = f"  rank = {history[-1]['ranking_loss']:.4f}" if total_rank_examples > 0 else ""
            print(f"  {model_name} Epoch {epoch:3d}/{n_epochs}  loss = {epoch_loss:.4f}{rank_msg}  time = {history[-1]['time_min']:.2f} min")
    return model, history


def build_user_seen_items(train_df, u2i, m2i):
    seen = {}
    for row in train_df.itertuples(index=False):
        uidx = u2i[row.userId]
        midx = m2i[row.movieId]
        seen.setdefault(uidx, set()).add(midx)
    return seen


def sample_negative_items(user_indices, user_seen_items, n_items):
    negatives = []
    for uidx in user_indices.detach().cpu().tolist():
        seen = user_seen_items.get(int(uidx), set())
        if len(seen) >= n_items:
            negatives.append(0)
            continue
        neg = np.random.randint(0, n_items)
        tries = 0
        while neg in seen and tries < 20:
            neg = np.random.randint(0, n_items)
            tries += 1
        if neg in seen:
            start = neg
            while neg in seen:
                neg = (neg + 1) % n_items
                if neg == start:
                    break
        negatives.append(neg)
    return torch.tensor(negatives, dtype=torch.long, device=DEVICE)


def predict_batch(model, df, u2i, m2i):
    model.eval()
    uids = torch.tensor([u2i.get(u, 0) for u in df.userId],  dtype=torch.long, device=DEVICE)
    mids = torch.tensor([m2i.get(m, 0) for m in df.movieId], dtype=torch.long, device=DEVICE)
    out = []
    with torch.no_grad():
        for s in range(0, len(uids), 2048):
            out.append(model.predict_rating(uids[s:s+2048], mids[s:s+2048]).cpu().numpy())
    return np.concatenate(out)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Summary printer
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(results):
    cols = list(results.keys()); w = 13
    hdr = f"{'Metric':<{w}}" + "".join(f"{c:>{w}}" for c in cols)
    sep = "-" * len(hdr)
    summary = "\n" + "=" * len(hdr) + "\nRESULTS SUMMARY\n" + sep + "\n"
    summary += hdr + "\n" + sep + "\n"
    metric_labels = [
        ("MAE", "MAE↓"),
        ("RMSE", "RMSE↓"),
        ("Precision@10", "Precision@10↑"),
        ("Recall@10", "Recall@10↑"),
        ("F-measure@10", "F-measure@10↑"),
        ("NDCG@10", "NDCG@10↑"),
    ]
    for metric_key, metric_label in metric_labels:
        row = f"{metric_label:<{w}}"
        for c in cols: row += f"{results[c].get(metric_key, float('nan')):>{w}.4f}"
        summary += row + "\n"
    summary += "=" * len(hdr)
    print(summary)

    # Save results to file
    with open("results/final_results.txt", "w") as f:
        f.write(summary)
    print(f"\nResults saved to results/final_results.txt")

    return summary


def ensure_export_dirs(paper_dir):
    fig_dir = os.path.join(paper_dir, "figures")
    table_dir = os.path.join(paper_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    return fig_dir, table_dir


def export_results_artifacts(results, histories, paper_dir):
    fig_dir, table_dir = ensure_export_dirs(paper_dir)

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    results_df = results_df[["MAE", "RMSE", "Precision@10", "Recall@10", "F-measure@10", "NDCG@10"]]
    results_df.to_csv(os.path.join(table_dir, "results_summary.csv"))

    latex_df = results_df.rename(columns={
        "MAE": "MAE$\\downarrow$",
        "RMSE": "RMSE$\\downarrow$",
        "Precision@10": "Precision@10$\\uparrow$",
        "Recall@10": "Recall@10$\\uparrow$",
        "F-measure@10": "F-measure@10$\\uparrow$",
        "NDCG@10": "NDCG@10$\\uparrow$",
    })
    latex_df.index.name = ""
    with open(os.path.join(table_dir, "results_summary.tex"), "w") as f:
        f.write(latex_df.round(4).to_latex(index=True, escape=False, index_names=False))

    if histories:
        history_df = pd.DataFrame(histories)
        history_df.to_csv(os.path.join(table_dir, "training_history.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        palette = {
            "BiasedMF": BASELINE_BLUE,
            "ClassicNeuMF": BASELINE_GREEN,
            "FlowNeuMF": FLOW_HIGHLIGHT,
        }
        for model_name, grp in history_df.groupby("model"):
            grp = grp.sort_values("epoch")
            start_loss = max(float(grp["loss"].iloc[0]), 1e-8)
            normalized = grp["loss"] / start_loss
            ax.plot(
                grp["epoch"],
                normalized,
                marker="o",
                markersize=3,
                linewidth=2.0 if model_name == "FlowNeuMF" else 1.8,
                color=palette.get(model_name, BASELINE_BLUE),
                label=model_name,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized Loss")
        ax.set_title("Normalized Training Loss Curves")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "training_loss_curves.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    rating_cols = ["MAE", "RMSE"]
    ranking_cols = ["Precision@10", "Recall@10", "F-measure@10", "NDCG@10"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, metric in zip(axes, rating_cols):
        vals = results_df[metric]
        style_barplot(ax, list(vals.index), vals.values, metric, ascending_good=False, yscale=None)
    fig.suptitle("Rating Prediction Metrics", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rating_metrics_bar.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    axes = axes.flatten()
    for ax, metric in zip(axes, ranking_cols):
        vals = results_df[metric]
        style_barplot(ax, list(vals.index), vals.values, metric, ascending_good=False, yscale="symlog")
    fig.suptitle("Top-10 Recommendation Metrics", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "topn_metrics_bar.png"), dpi=200)
    plt.close(fig)

    for metric in results_df.columns:
        plt.figure(figsize=(7, 4.5))
        ascending = metric in {"MAE", "RMSE"}
        values = results_df[metric].sort_values(ascending=ascending)
        plt.bar(values.index, values.values)
        plt.title(metric)
        plt.ylabel("Value")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        safe_name = metric.replace("@", "_at_").replace("-", "_").replace("/", "_").lower()
        plt.savefig(os.path.join(fig_dir, f"{safe_name}.png"), dpi=200)
        plt.close()

    print(f"\nArtifacts exported to {fig_dir} and {table_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowNeuMF on MovieLens")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                        help="Path to MovieLens dataset directory")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs for FlowNeuMF")
    parser.add_argument("--batch-size", type=int, default=32768,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--topn-sample-users", type=int, default=None,
                        help="Number of users to sample for Top-N evaluation")
    parser.add_argument("--flow-lambda", type=float, default=1.0,
                        help="Weight of the rating loss inside FlowNeuMF")
    parser.add_argument("--flow-weight", type=float, default=0.15,
                        help="Weight of the flow-matching loss inside FlowNeuMF")
    parser.add_argument("--consistency-weight", type=float, default=0.5,
                        help="Weight of the endpoint consistency loss inside FlowNeuMF")
    parser.add_argument("--emb-dim", type=int, default=64,
                        help="Embedding size for both models")
    parser.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR,
                        help="Directory where paper figures/tables should be exported")
    parser.add_argument("--ranking-weight", type=float, default=0.3,
                        help="Weight of the pairwise ranking loss for neural models")
    parser.add_argument("--positive-threshold", type=float, default=4.0,
                        help="Ratings at or above this value are treated as positive interactions for ranking loss")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible training")
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_global_seed(args.seed)
    print("=" * 65)
    print("CS550 Final Project – FlowNeuMF Movie Recommender")
    print(f"Device : {DEVICE}")
    print(f"Data   : {args.data_path}")
    print(f"Seed   : {args.seed}")
    print("=" * 65)

    print("\n[1] Loading data …")
    ratings, movies = load_data(args.data_path)
    print(f"  Ratings : {len(ratings):,}  Users : {ratings.userId.nunique()}  Movies : {ratings.movieId.nunique()}")
    is_large_dataset = len(ratings) >= LARGE_DATASET_THRESHOLD
    topn_sample_users = args.topn_sample_users
    if topn_sample_users is None:
        topn_sample_users = 200 if is_large_dataset else None
    if is_large_dataset:
        print("  Large dataset detected: using reduced Top-N evaluation.")

    print("\n[2] Train / Test split (80/20 per user) …")
    split_start = time.time()
    train, test = train_test_split(ratings)
    print(f"  Train : {len(train):,}   Test : {len(test):,}")
    print(f"  Split time: {(time.time() - split_start)/60:.2f} min")

    users    = sorted(ratings.userId.unique())
    items    = sorted(ratings.movieId.unique())
    u2i      = {u: i for i, u in enumerate(users)}
    m2i      = {m: i for i, m in enumerate(items)}
    item_arr = np.array(items)
    results  = {}
    histories = []
    user_seen_items = build_user_seen_items(train, u2i, m2i)

    mlp_hidden = [128, 64]
    neural_models = {
        "BiasedMF": BiasedMF(len(users), len(items), emb_dim=args.emb_dim).to(DEVICE),
        "ClassicNeuMF": ClassicNeuMF(len(users), len(items), emb_dim=args.emb_dim, mlp_hidden=mlp_hidden).to(DEVICE),
        "FlowNeuMF": FlowNeuMF(len(users), len(items), emb_dim=args.emb_dim, mlp_hidden=mlp_hidden,
                                ode_steps=10, lambda_r=args.flow_lambda,
                                flow_weight=args.flow_weight,
                                consistency_weight=args.consistency_weight).to(DEVICE),
    }
    item_cf = ItemCF(k=100)
    neural_models["BiasedMF"].set_global_bias(train.rating.mean())

    print("\n[3] Building models …")
    print(f"  {'ItemCF':<12} Parameters : non-parametric")
    for model_name, model in neural_models.items():
        print(f"  {model_name:<12} Parameters : {sum(p.numel() for p in model.parameters()):,}")

    print("\n[4] Fitting ItemCF …")
    fit_start = time.time()
    item_cf.fit(train, u2i, m2i, users, items)
    print(f"  ItemCF fit time: {(time.time() - fit_start)/60:.2f} min")

    print(f"\n[5] Training neural models ({args.epochs} epochs each) …")
    print(f"  Batch size: {args.batch_size}, Learning rate: {args.lr}, Embedding dim: {args.emb_dim}")
    print(f"  Ranking weight: {args.ranking_weight}, Positive threshold: {args.positive_threshold}")
    for model_name, model in neural_models.items():
        print(f"\n  -> Training {model_name}")
        model, model_history = train_model(model, model_name, train, u2i, m2i,
                                           n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                           ranking_weight=args.ranking_weight if model_name != "BiasedMF" else 0.0,
                                           positive_threshold=args.positive_threshold,
                                           user_seen_items=user_seen_items,
                                           n_items=len(items))
        neural_models[model_name] = model
        histories.extend(model_history)

    print("\n[6] ItemCF – Rating Prediction …")
    itemcf_preds = item_cf.predict_batch(test)
    itemcf_mae, itemcf_rmse = mae_rmse(itemcf_preds, test.rating.values)
    print(f"  MAE  = {itemcf_mae:.4f}   RMSE = {itemcf_rmse:.4f}")

    print("\n[7] ItemCF – Top-10 Recommendation …")
    itemcf_p, itemcf_r, itemcf_f, itemcf_ndcg = numpy_topn(
        item_cf.score_all_items, item_arr, train, test, sample_users=topn_sample_users)
    print(f"  P={itemcf_p:.4f}  R={itemcf_r:.4f}  F={itemcf_f:.4f}  NDCG={itemcf_ndcg:.4f}")
    results["ItemCF"] = {
        "MAE": itemcf_mae, "RMSE": itemcf_rmse,
        "Precision@10": itemcf_p, "Recall@10": itemcf_r,
        "F-measure@10": itemcf_f, "NDCG@10": itemcf_ndcg,
    }

    step_id = 8
    for model_name, model in neural_models.items():
        print(f"\n[{step_id}] {model_name} – Rating Prediction …")
        preds = predict_batch(model, test, u2i, m2i)
        mae, rmse = mae_rmse(preds, test.rating.values)
        print(f"  MAE  = {mae:.4f}   RMSE = {rmse:.4f}")
        step_id += 1

        print(f"\n[{step_id}] {model_name} – Top-10 Recommendation …")
        p_at_10, r_at_10, f_at_10, ndcg = neural_topn(
            model, u2i, m2i, item_arr, train, test, sample_users=topn_sample_users)
        print(f"  P={p_at_10:.4f}  R={r_at_10:.4f}  F={f_at_10:.4f}  NDCG={ndcg:.4f}")
        results[model_name] = {
            "MAE": mae, "RMSE": rmse,
            "Precision@10": p_at_10, "Recall@10": r_at_10,
            "F-measure@10": f_at_10, "NDCG@10": ndcg,
        }
        step_id += 1

    print_summary(results)
    export_results_artifacts(results, histories, args.paper_dir)

    print(f"\n[{step_id}] Saving trained models ...")
    for model_name, model in neural_models.items():
        model_path = f"results/{model_name.lower()}_model.pt"
        torch.save({
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'n_users': len(users),
            'n_items': len(items),
            'u2i': u2i,
            'm2i': m2i,
            'results': results.get(model_name, {}),
        }, model_path)
        print(f"  {model_name} saved to {model_path}")


if __name__ == "__main__":
    main()
