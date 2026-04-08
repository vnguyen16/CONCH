# ============================================================
# Save attention weights for heatmap visualization and multi-scale fusion
# - Uses .pt OR .h5 features (set FEAT_BACKEND)
# - Runs 5-fold CV from your existing split folders
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import h5py

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*"
)

# -----------------
# PATHS (EDIT THESE)
# -----------------
# out dir for attn weights 
OUT_DIR = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\attn_npz\10x"

# configs for concept guided attn
PATCH_CONCEPT_CSV = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\noCAP_patch_ptfile_notopk_conf_PATCH.csv" # 10x og 
# PATCH_CONCEPT_CSV = r'C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\2.5x\noCAP_patch_ptfile_notopk_conf_PATCH.csv' # 2.5x og
# PATCH_CONCEPT_CSV = r'C:\Users\Vivian\Documents\CONCH\test_text_encoder\slide_concept_scores\5x\noCAP_patch_ptfile_notopk_conf_PATCH.csv' # 5x og
# PATCH_CONCEPT_CSV = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\V3_concept_prior\V3_top3_concept_prior_PATCH.csv" # top3/avg variants
# PATCH_CONCEPT_CSV = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\V3_concept_prior\separate_variants_10x\V3_concept_prior_PATCH_VARIANTS.csv" # 10x expanded variants

# PATCH_CONCEPT_CSV = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\V3_concept_prior\separate_variants_25x\V3_concept_prior_PATCH_VARIANTS.csv" # 2.5x expanded variants    
# PATCH_CONCEPT_CSV = r'C:\Users\Vivian\Documents\CONCH\test_text_encoder\V3_concept_prior\separate_variants_5x\V3_concept_prior_PATCH_VARIANTS.csv' # 5x expanded variants

SLIDE_COL = "slide_id"
LABEL_COL = "true_label_str"
X_COL, Y_COL = "x", "y"
CONCEPT_PREFIX = "concept_" # og
# CONCEPT_PREFIX = "variant_" # for your new concept prior csv with separate variants instead of topk
# -----------------------------
CV_ROOT = r"C:/Users/Vivian/Documents/PANTHER/PANTHER/src/splits/cross-val"

# Virchow2 deep features location
# If you have one file per slide named "<slide_id>.pt" or "<slide_id>.h5"
FEAT_BACKEND = "pt"   # "pt" or "h5"

# PT_FEAT_DIR = r"C:\Users\Vivian\Documents\CONCH\virchow2_img_feats\10x_feats\40x\pt"   # e.g. ...\virchow2\feats_pt
# PT_FEAT_DIR = r'C:\Users\Vivian\Documents\CONCH\conch_img_feats\10x_feats\conchextracted_mag10x_patch224_fp\feats_pt' # CONCH 10x pt features
PT_FEAT_DIR = r"C:\Users\Vivian\Documents\CONCH\conch_img_feats\10x_unnorm_feats\20x\pt"        # CONCH unnorm
# PT_FEAT_DIR = r'C:\Users\Vivian\Documents\CONCH\uni2h_img_feats\pt'       # UNI2 10x pt features
# PT_FEAT_DIR = r"C:\Users\Vivian\Documents\CONCH\run2_conch15_img_feats\pt"       # conchv1.5 pt features
# H5_FEAT_DIR = r"C:\Users\Vivian\Documents\CLAM\CLAM\FEATURES_DIR_5x\FEATURES_DIR_10x\uniextracted_mag10x_patch224_fp\feats_h5"        # UNI
# H5_FEAT_DIR = r'C:\Users\Vivian\Documents\CLAM\CLAM\FEATURES_DIR_5x\spider_run2\feats_h5' #SPIDER h5 feats

# PT_FEAT_DIR = r'C:\Users\Vivian\Documents\CONCH\conch_img_feats\2.5x_feats\40x\pt' # CONCH 2.5x
# PT_FEAT_DIR = r'C:\Users\Vivian\Documents\CONCH\conch_img_feats\25x_unnorm_feats\20x\pt' # CONCH 2.5x unnorm
# PT_FEAT_DIR = r'C:\Users\Vivian\Documents\CONCH\conch_img_feats\5x_feats\40x\pt' # CONCH 5x
# PT_FEAT_DIR = r'C:\Users\Vivian\Documents\CONCH\conch_img_feats\5x_unnorm_feats\20x\pt' # CONCH 5x unnorm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# Bag controls
BAG_CAP = None      # e.g., 4096
TOPK = None         # e.g., 1024 (rank by deep_norm)
TOPK_MODE = "deep_norm"

# Train controls
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
HID = 128 # og = 128
DROPOUT = 0.1

# OG
torch.manual_seed(SEED)
np.random.seed(SEED)

EXPERIMENTS = ["vision_pool", "vision_abmil","vision_concept_gated"]
# , "vision_pool", "vision_abmil","vision_concept_gated"


# ============================================================
# HELPERS
# ============================================================

def labels_to_int(arr):
    s = pd.Series(arr).astype(str).str.upper().str.strip()
    y = s.map({"FA": 0, "PT": 1})
    if y.isna().any():
        raise ValueError(f"Bad labels: {s[y.isna()].unique()}")
    return y.astype(int).to_numpy()

def read_slide_list(csv_path: str):
    d = pd.read_csv(csv_path)
    if "slide" in d.columns:
        slides = d["slide"].astype(str).tolist()
    elif "slide_id" in d.columns:
        slides = d["slide_id"].astype(str).tolist()
    else:
        slides = d.iloc[:, 0].astype(str).tolist()
    return [os.path.splitext(s.strip())[0] for s in slides]

def find_fold_dirs(cv_root: str):
    out = []
    for name in sorted(os.listdir(cv_root)):
        p = os.path.join(cv_root, name)
        if os.path.isdir(p) and all(os.path.isfile(os.path.join(p, f"{x}.csv")) for x in ["train","val","test"]):
            out.append(p)
    return out

def eval_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if mask.sum() == 0:
        return {"auc": np.nan, "balacc": np.nan, "acc": np.nan}
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask]
    y_pred = (y_prob >= thr).astype(int)
    auc = np.nan
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    return {
        "auc": auc,
        "balacc": balanced_accuracy_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
    }

def summarize_percent(df_metrics, cols=("auc","balacc","acc")):
    out = {}
    for c in cols:
        m = df_metrics[c].mean() * 100.0
        s = df_metrics[c].std(ddof=1) * 100.0
        out[c] = (round(m, 2), round(s, 2))
    return out

# LOAD CONCEPT SCORES (OPTIONAL, for later use in concept ablation) =====================
df_patch = pd.read_csv(PATCH_CONCEPT_CSV)

# ============================================ Max pool variants
def max_pool_variants(df_patch, concept_cols, variants_per_concept=5):
    n_total = len(concept_cols)
    if n_total % variants_per_concept != 0:
        raise ValueError(
            f"Number of concept columns ({n_total}) is not divisible by "
            f"variants_per_concept ({variants_per_concept})"
        )

    n_concepts = n_total // variants_per_concept
    pooled = pd.DataFrame(index=df_patch.index)

    for i in range(n_concepts):
        cols_i = concept_cols[i * variants_per_concept : (i + 1) * variants_per_concept]
        pooled[f"concept_{i}"] = df_patch[cols_i].max(axis=1)

    return pooled
# ============================================= Max pool variants


df_patch.columns = [c.strip() for c in df_patch.columns]

concept_cols = sorted([c for c in df_patch.columns if c.startswith(CONCEPT_PREFIX)],
                      key=lambda s: int(s.split("_")[1]))
if not concept_cols:
    raise ValueError("No concept_* columns found")

df_patch[concept_cols] = df_patch[concept_cols].apply(pd.to_numeric, errors="coerce")
df_patch[concept_cols] = df_patch[concept_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ========================================== Max pool variants
# pooled_concepts = max_pool_variants(df_patch, concept_cols, variants_per_concept=5)

# meta_cols = [SLIDE_COL, LABEL_COL]
# if X_COL in df_patch.columns:
#     meta_cols.append(X_COL)
# if Y_COL in df_patch.columns:
#     meta_cols.append(Y_COL)

# df_patch = pd.concat([df_patch[meta_cols].copy(), pooled_concepts], axis=1)

# # concept_cols = sorted([c for c in df_patch.columns if c.startswith(CONCEPT_PREFIX)],
# #                       key=lambda s: int(s.split("_")[1]))
# concept_cols = sorted([c for c in df_patch.columns if c.startswith("concept_")],
#                       key=lambda s: int(s.split("_")[1]))
# print("n concept cols after pooling:", len(concept_cols))
# print(concept_cols[:10])
# ============================================= Max pool variants

lab = df_patch.groupby(SLIDE_COL)[LABEL_COL].first()
y_map_global = {sid: int(labels_to_int([lab.loc[sid]])[0]) for sid in lab.index.tolist()}

# ============================================================
# DEEP FEATURE LOADER (NO CONCEPTS)
# ============================================================

def load_slide_deep(slide_id: str):
    if FEAT_BACKEND == "pt":
        pt_path = os.path.join(PT_FEAT_DIR, f"{slide_id}.pt")
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(pt_path)
        obj = torch.load(pt_path, map_location="cpu")

        # Accept common formats:
        # 1) dict with "features"
        # 2) plain tensor/ndarray
        if isinstance(obj, dict):
            if "features" in obj:
                feats = obj["features"]
            elif "feats" in obj:
                feats = obj["feats"]
            else:
                # if dict contains a single tensor-like value, take it
                for v in obj.values():
                    if torch.is_tensor(v) or isinstance(v, np.ndarray):
                        feats = v
                        break
                else:
                    raise ValueError(f"Could not find features in {pt_path}. keys={list(obj.keys())}")
        else:
            feats = obj

        if torch.is_tensor(feats):
            feats = feats.float().cpu().numpy()
        else:
            feats = np.asarray(feats, dtype=np.float32)

        return feats.astype(np.float32)

    elif FEAT_BACKEND == "h5":
        if not H5_FEAT_DIR:
            raise ValueError("Set H5_FEAT_DIR when FEAT_BACKEND='h5'")
        h5_path = os.path.join(H5_FEAT_DIR, f"{slide_id}.h5")
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)
        with h5py.File(h5_path, "r") as f:
            # common keys
            for k in ["features", "feats", "embeddings"]:
                if k in f and not isinstance(f[k], h5py.Group):
                    feats = f[k][:]
                    break
            else:
                raise KeyError(f"No usable feature dataset in {h5_path}. Keys={list(f.keys())}")

        return feats.astype(np.float32)

    else:
        raise ValueError(f"Bad FEAT_BACKEND='{FEAT_BACKEND}'")


# concept-guided attn ================
def load_slide_deep_with_coords(slide_id: str):
    if FEAT_BACKEND == "pt":
        path = os.path.join(PT_FEAT_DIR, f"{slide_id}.pt")
        obj = torch.load(path, map_location="cpu")

        if not isinstance(obj, dict) or "features" not in obj or "coords" not in obj:
            raise ValueError(f"Virchow2 pt must have keys 'features' and 'coords': {path}")

        feats = obj["features"].float().cpu().numpy()
        coords = obj["coords"].int().cpu().numpy()
        return feats.astype(np.float32), coords.astype(np.int32)

    elif FEAT_BACKEND == "h5":
        path = os.path.join(H5_FEAT_DIR, f"{slide_id}.h5")
        with h5py.File(path, "r") as f:
            # features
            if "features" in f: feats = f["features"][:]
            elif "embeddings" in f and not isinstance(f["embeddings"], h5py.Group): feats = f["embeddings"][:]
            else: raise KeyError(f"No feature dataset in {path}. keys={list(f.keys())}")

            # coords
            if "coords" in f: coords = f["coords"][:]
            elif "embed_loc" in f: coords = f["embed_loc"][:]   # your TRIDENT-style
            else: raise KeyError(f"No coords/embed_loc in {path}. keys={list(f.keys())}")

        return feats.astype(np.float32), coords.astype(np.int32)

    else:
        raise ValueError(FEAT_BACKEND)
# =================================================

def label_from_slide_id(slide_id: str) -> int:
    s = str(slide_id).upper().strip()
    if s.startswith("FA"):
        return 0
    if s.startswith("PT"):
        return 1
    raise ValueError(f"Cannot infer label from slide_id='{slide_id}' (must start with FA/PT)")


# ============================================================
# DATASET
# ============================================================

class VisionOnlyBagDataset(Dataset):
    def __init__(self, slide_ids, bag_cap=None, topk=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.bag_cap = bag_cap
        self.topk = topk

        keep = []
        for sid in slide_ids:
            try:
                # _ = load_slide_deep(sid)
                _ = load_slide_deep_with_coords(sid) # save attn heatmaps
                keep.append(sid)
            except Exception:
                pass
        self.slide_ids = keep

    def __len__(self): return len(self.slide_ids)

    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        # X = load_slide_deep(sid)  # (N,D)
        X, coords = load_slide_deep_with_coords(sid) # save attn heatmaps
        N = X.shape[0]

        # cap
        if self.bag_cap is not None and N > self.bag_cap:
            sel = self.rng.choice(N, size=self.bag_cap, replace=False)
            X = X[sel]
            coords = coords[sel]

        # topk by deep norm
        if self.topk is not None and X.shape[0] > self.topk:
            score = np.linalg.norm(X, axis=1)
            keep = np.argsort(-score)[: self.topk]
            X = X[keep]
            coords = coords[keep]

        y = label_from_slide_id(sid)
        return (torch.from_numpy(X), torch.tensor(y, dtype=torch.long), sid,torch.from_numpy(coords).int(),) 


def collate_bag(batch):
    return batch  # batch_size=1 -> [ (X,y,sid) ]

from torch.utils.data import Dataset

class VisionConceptBagDataset(Dataset):
    def __init__(self, df_patch_concept, slide_ids, concept_cols, bag_cap=None, topk=None, seed=0):
        self.df = df_patch_concept[df_patch_concept[SLIDE_COL].isin(slide_ids)].copy()
        self.slide_ids = [s for s in slide_ids if s in set(self.df[SLIDE_COL].unique()) and s in y_map_global]
        self.concept_cols = concept_cols
        self.bag_cap = bag_cap
        self.topk = topk
        self.rng = np.random.default_rng(seed)
        self.by_slide = {sid: d for sid, d in self.df.groupby(SLIDE_COL)}

        # keep only slides with deep files
        keep = []
        for sid in self.slide_ids:
            try:
                _ = load_slide_deep_with_coords(sid)
                keep.append(sid)
            except Exception:
                pass
        self.slide_ids = keep

    def __len__(self): return len(self.slide_ids)

    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        Xd, coords = load_slide_deep_with_coords(sid)  # (N,D), (N,2)
        d = self.by_slide[sid]

        # map (x,y)->concept vec
        keys = list(zip(d[X_COL].astype(int).to_numpy(), d[Y_COL].astype(int).to_numpy()))
        C = d[self.concept_cols].to_numpy(dtype=np.float32)
        concept_map = {k: C[i] for i, k in enumerate(keys)}

        Xc = np.zeros((coords.shape[0], len(self.concept_cols)), dtype=np.float32)
        for i, (x, y) in enumerate(coords.astype(int)):
            v = concept_map.get((int(x), int(y)))
            if v is not None:
                Xc[i] = v

        # optional cap
        if self.bag_cap is not None and Xd.shape[0] > self.bag_cap:
            sel = self.rng.choice(Xd.shape[0], size=self.bag_cap, replace=False)
            # Xd = Xd[sel]; Xc = Xc[sel]
            Xd = Xd[sel]; Xc = Xc[sel]; coords = coords[sel]

        # optional topk by deep norm
        if self.topk is not None and Xd.shape[0] > self.topk:
            score = np.linalg.norm(Xd, axis=1)
            keep = np.argsort(-score)[: self.topk]
            # Xd = Xd[keep]; Xc = Xc[keep]
            Xd = Xd[keep]; Xc = Xc[keep]; coords = coords[keep]

        y = y_map_global[sid]
        return (torch.from_numpy(Xd), torch.from_numpy(Xc), torch.tensor(y, dtype=torch.long), sid,torch.from_numpy(coords).int(),   # added
        )
    
# ============================================================
# MODELS
# ============================================================

class PoolingClassifier(nn.Module):
    def __init__(self, in_dim, hid=128, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hid, 1),
        )
    def forward(self, X):  # (N,D)
        z = X.mean(dim=0)
        return self.net(z).squeeze(0)

class AttentionMIL(nn.Module):
    def __init__(self, in_dim, hid=128, drop=0.1):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop))
        self.attn = nn.Linear(hid, 1)
        self.cls = nn.Linear(hid, 1)
    def forward(self, X, return_details=False):
        H = self.phi(X)
        a = self.attn(H).squeeze(1)
        w = torch.softmax(a, dim=0)
        z = (w.unsqueeze(1) * H).sum(0)
        logit = self.cls(z).squeeze(0)

        # save attn heatmap
        if return_details:
            return logit, {"attn_final": w, "attn_logits": a}
        
        return logit, w

# gated attn class with attn heatmap weights 
class GatedAttentionMIL(nn.Module):
    def __init__(self, deep_dim, concept_dim, hid=128, drop=0.1):
        super().__init__()
        self.phi_d = nn.Sequential(nn.Linear(deep_dim, hid), nn.ReLU(), nn.Dropout(drop))
        self.phi_c = nn.Sequential(nn.Linear(concept_dim, hid), nn.ReLU(), nn.Dropout(drop))
        self.attn_d = nn.Linear(hid, 1)
        self.attn_c = nn.Linear(hid, 1)
        self.cls = nn.Linear(hid, 1)

        # set one of these in your experiments
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.alpha = 1.0

    def forward(self, Xd, Xc, return_details=False):
        Hd = self.phi_d(Xd)
        Hc = self.phi_c(Xc)

        a_d = self.attn_d(Hd).squeeze(1)
        a_c = self.attn_c(Hc).squeeze(1)

        # choose the version you are actually using:
        a = a_d + a_c
        # alpha = torch.sigmoid(self.alpha)
        # a = alpha * a_d + (1 - alpha) * a_c
        # a = self.alpha * a_d + (1 - self.alpha) * a_c

        w = torch.softmax(a, dim=0)
        z = (w.unsqueeze(1) * Hd).sum(0)
        logit = self.cls(z).squeeze(0)

        if return_details:
            return logit, {
                "attn_final": w,
                "attn_vision": a_d,
                "attn_concept": a_c,
                "attn_logits": a,
            }
        return logit, w    
# ============================================================
# TRAIN/EVAL
# ============================================================

# -----------------------------------------
#  save attn heatmap train one fold
# def train_one_fold(model, dl_tr, dl_te, mode: str):
def train_one_fold(model, dl_tr, dl_te, mode: str, fold_name: str):
    out_dir = os.path.join(OUT_DIR, mode, fold_name)
    os.makedirs(out_dir, exist_ok=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- train ----------------
    model.train()
    for ep in range(EPOCHS):
        for batch in dl_tr:
            item = batch[0]

            if mode == "vision_pool":
                (Xd, y, sid, coords) = item
                Xd = Xd.to(DEVICE)
                y = y.float().to(DEVICE)
                logit = model(Xd)

            elif mode == "vision_abmil":
                (Xd, y, sid, coords) = item
                Xd = Xd.to(DEVICE)
                y = y.float().to(DEVICE)
                logit, _ = model(Xd)

            elif mode == "vision_concept_gated":
                (Xd, Xc, y, sid, coords) = item
                Xd = Xd.to(DEVICE)
                Xc = Xc.to(DEVICE)
                y = y.float().to(DEVICE)
                logit, _ = model(Xd, Xc)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss = loss_fn(logit.view(1), y.view(1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # ---------------- eval ----------------
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in dl_te:
            item = batch[0]

            if mode == "vision_pool":
                (Xd, y, sid, coords) = item
                Xd = Xd.to(DEVICE)
                logit = model(Xd)

                prob = torch.sigmoid(logit.detach().float().cpu()).item()
                coords_np = coords.detach().cpu().numpy().astype(np.int32)

            elif mode == "vision_abmil":
                (Xd, y, sid, coords) = item
                Xd = Xd.to(DEVICE)
                logit, details = model(Xd, return_details=True)

                prob = torch.sigmoid(logit.detach().float().cpu()).item()
                coords_np = coords.detach().cpu().numpy().astype(np.int32)
                w_np = details["attn_final"].detach().cpu().numpy().astype(np.float32)
                a_np = details["attn_logits"].detach().cpu().numpy().astype(np.float32)

                np.savez_compressed(
                    # os.path.join(OUT_DIR, f"{sid}_eval.npz"),
                    os.path.join(out_dir, f"{sid}_eval.npz"),
                    slide_id=np.array([sid]),
                    y_true=np.array([int(y.item())], dtype=np.int32),
                    p_pt=np.array([prob], dtype=np.float32),
                    coords=coords_np,
                    attn=w_np,
                    attn_logits=a_np,
                )

            elif mode == "vision_concept_gated":
                (Xd, Xc, y, sid, coords) = item
                Xd = Xd.to(DEVICE)
                Xc = Xc.to(DEVICE)
                logit, details = model(Xd, Xc, return_details=True)

                prob = torch.sigmoid(logit.detach().float().cpu()).item()
                coords_np = coords.detach().cpu().numpy().astype(np.int32)
                Xc_np = Xc.detach().cpu().numpy().astype(np.float32)

                w_np = details["attn_final"].detach().cpu().numpy().astype(np.float32)
                a_d_np = details["attn_vision"].detach().cpu().numpy().astype(np.float32)
                a_c_np = details["attn_concept"].detach().cpu().numpy().astype(np.float32)
                a_np = details["attn_logits"].detach().cpu().numpy().astype(np.float32)

                concept_mean = Xc_np.mean(axis=0).astype(np.float32)
                concept_attn_weighted = (w_np[:, None] * Xc_np).sum(axis=0).astype(np.float32)

                np.savez_compressed(
                    # os.path.join(OUT_DIR, f"{sid}_eval.npz"),
                    os.path.join(out_dir, f"{sid}_eval.npz"),
                    slide_id=np.array([sid]),
                    y_true=np.array([int(y.item())], dtype=np.int32),
                    p_pt=np.array([prob], dtype=np.float32),
                    coords=coords_np,
                    Xc=Xc_np,
                    attn=w_np,
                    attn_vision=a_d_np,
                    attn_concept=a_c_np,
                    attn_logits=a_np,
                    concept_mean=concept_mean,
                    concept_attn_weighted=concept_attn_weighted,
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            rows.append({
                "slide_id": sid,
                "y_true": int(y.item()),
                "p_pt": float(prob),
                "n_patches": int(coords_np.shape[0]),
            })

    df_pred = pd.DataFrame(rows)
    metrics = eval_metrics(df_pred["y_true"].values, df_pred["p_pt"].values, thr=0.5)
    return metrics, df_pred

# ============================================================
# RUN CV
# ============================================================

fold_dirs = find_fold_dirs(CV_ROOT)
all_rows = []

for mode in EXPERIMENTS:
# for mode_idx, mode in enumerate(EXPERIMENTS): # FOLD SEED
    print(f"\n==================== {mode} ====================")
    fold_metrics = []
    all_alpha_histories = [] # NEW to store alpha histories

    for fold_dir in fold_dirs:
    # for fold_idx, fold_dir in enumerate(fold_dirs): # FOLD SEED
        fold_name = os.path.basename(fold_dir)
        train_slides = read_slide_list(os.path.join(fold_dir, "train.csv"))
        val_slides   = read_slide_list(os.path.join(fold_dir, "val.csv"))
        test_slides  = read_slide_list(os.path.join(fold_dir, "test.csv"))
        fit_slides = train_slides + val_slides

        # -------- datasets per mode --------
        if mode in ["vision_pool", "vision_abmil"]:
            ds_tr = VisionOnlyBagDataset(fit_slides, bag_cap=BAG_CAP, topk=TOPK, seed=SEED)
            ds_te = VisionOnlyBagDataset(test_slides, bag_cap=BAG_CAP, topk=TOPK, seed=SEED)

        elif mode == "vision_concept_gated":
            ds_tr = VisionConceptBagDataset(df_patch, fit_slides, concept_cols,
                                            bag_cap=BAG_CAP, topk=TOPK, seed=SEED)
            ds_te = VisionConceptBagDataset(df_patch, test_slides, concept_cols,
                                            bag_cap=BAG_CAP, topk=TOPK, seed=SEED)
            
        else:
            raise ValueError(mode)

        if len(ds_tr) == 0 or len(ds_te) == 0:
            print(f"[{fold_name}] skipped (empty train/test after filtering)")
            continue

        dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True, collate_fn=collate_bag)
        dl_te = DataLoader(ds_te, batch_size=1, shuffle=False, collate_fn=collate_bag)

        # -------- model init per mode --------
        if mode == "vision_pool":
            # X0, _, _ = ds_tr[0]
            X0, _, _, _ = ds_tr[0]
            in_dim = X0.shape[1]
            model = PoolingClassifier(in_dim=in_dim, hid=HID, drop=DROPOUT).to(DEVICE)

        elif mode == "vision_abmil":
            # X0, _, _ = ds_tr[0]
            X0, _, _, _ = ds_tr[0]
            in_dim = X0.shape[1]
            model = AttentionMIL(in_dim=in_dim, hid=HID, drop=DROPOUT).to(DEVICE)

        elif mode == "vision_concept_gated":
            # Xd0, Xc0, _, _ = ds_tr[0]
            Xd0, Xc0, _, _, _ = ds_tr[0]
            model = GatedAttentionMIL(deep_dim=Xd0.shape[1], concept_dim=Xc0.shape[1],
                                      hid=HID, drop=DROPOUT).to(DEVICE)
        else:
            raise ValueError(mode)

        # m = train_one_fold(model, dl_tr, dl_te, mode) # og
        # m, df_pred = train_one_fold(model, dl_tr, dl_te, mode)
        m, df_pred = train_one_fold(model, dl_tr, dl_te, mode, fold_name)
        
        
        fold_metrics.append({"fold": fold_name, **m, "n_test": len(ds_te)})
        print(f"[{fold_name}] auc={m['auc']:.3f} balacc={m['balacc']:.3f} acc={m['acc']:.3f} (n_test={len(ds_te)})")

    dfm = pd.DataFrame(fold_metrics)
    if len(dfm) == 0:
        print(f"No folds ran for {mode}")
        continue

    summary = summarize_percent(dfm, cols=("auc", "balacc", "acc"))
    all_rows.append({
        "experiment": mode,
        "AUC (mean±std %)": f"{summary['auc'][0]:.2f} ± {summary['auc'][1]:.2f}",
        "BAC (mean±std %)": f"{summary['balacc'][0]:.2f} ± {summary['balacc'][1]:.2f}",
        "ACC (mean±std %)": f"{summary['acc'][0]:.2f} ± {summary['acc'][1]:.2f}",
        "n_folds": len(dfm),
    })

results_table = pd.DataFrame(all_rows)
print("\n==================== FINAL SUMMARY TABLE ====================")
print(results_table.to_string(index=False))
