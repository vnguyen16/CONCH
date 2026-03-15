# import os, re
# import numpy as np
# import pandas as pd
# import h5py
# import torch

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

# # ==============================
# # EDIT THESE
# # ==============================
# FEATURE_ROOT = r"C:\Users\Vivian\Documents\CONCH\conch_img_feats\40x\pt"  # folder of slide_id.pt or .h5
# FEATURE_EXT  = ".pt"  # ".pt" or ".h5"

# LABEL_CSV = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\10x_noNORM_conch_zeroshot_slides_fullprompts_40x.csv"

# TEST_SIZE = 0.2
# N_SPLITS = 5
# SEED = 2348

# TOPK = 64          # try 16, 32, 64, 128
# MAX_PATCHES = 0    # 0 = use all patches; else cap per slide for speed (e.g., 2000)

# # ==============================
# def slide_to_patient(slide_id: str) -> str:
#     return re.sub(r"\s+B\d+\s*$", "", str(slide_id).strip())

# def load_patch_feats(slide_id: str):
#     p = os.path.join(FEATURE_ROOT, slide_id + FEATURE_EXT)
#     if not os.path.exists(p):
#         return None

#     if FEATURE_EXT == ".pt":
#         obj = torch.load(p, map_location="cpu")
#         feats = obj["features"].numpy()
#     else:
#         with h5py.File(p, "r") as f:
#             feats = f["features"][:]

#     if feats.ndim != 2 or feats.shape[0] == 0:
#         return None

#     if MAX_PATCHES > 0 and feats.shape[0] > MAX_PATCHES:
#         feats = feats[:MAX_PATCHES]

#     return feats.astype(np.float32)

# def metrics(y, p):
#     yhat = (p >= 0.5).astype(int)
#     return {
#         "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
#         "balacc": balanced_accuracy_score(y, yhat),
#         "acc": accuracy_score(y, yhat),
#     }

# def mean_pool(feats: np.ndarray) -> np.ndarray:
#     return feats.mean(axis=0)

# def topk_pool_by_direction(feats: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
#     """
#     Score each patch by dot(feat, w). Take top-k and mean pool.
#     w should be shape (D,).
#     """
#     k = min(k, feats.shape[0])
#     scores = feats @ w  # (N,)
#     idx = np.argpartition(scores, -k)[-k:]
#     return feats[idx].mean(axis=0)

# # ==============================
# # Build slide table for slides that actually have features
# # ==============================
# df = pd.read_csv(LABEL_CSV)
# df["slide_id"] = df["slide_id"].astype(str).str.strip()
# df["true_label_str"] = df["true_label_str"].astype(str).str.upper().str.strip()
# df["y"] = df["true_label_str"].map({"FA": 0, "PT": 1})
# df["patient_id"] = df["slide_id"].apply(slide_to_patient)

# # keep only slides with feature file present
# mask = df["slide_id"].apply(lambda s: os.path.exists(os.path.join(FEATURE_ROOT, s + FEATURE_EXT)))
# df = df[mask].copy()

# print(f"Slides with feature files: {len(df)}")
# print("Label counts:\n", df["true_label_str"].value_counts())
# print("Patients with features:", df["patient_id"].nunique())

# # Load patch feats for all available slides once (so we don't reread disk every split)
# slide_ids = df["slide_id"].tolist()
# y_all = df["y"].astype(int).to_numpy()
# groups = df["patient_id"].astype(str).to_numpy()

# feat_dict = {}
# keep_idx = []
# for i, sid in enumerate(slide_ids):
#     feats = load_patch_feats(sid)
#     if feats is None:
#         continue
#     feat_dict[sid] = feats
#     keep_idx.append(i)

# keep_idx = np.array(keep_idx, dtype=int)
# slide_ids = [slide_ids[i] for i in keep_idx]
# y_all = y_all[keep_idx]
# groups = groups[keep_idx]

# print(f"Usable slides (loaded): {len(slide_ids)}")
# print(f"Example patch feat shape: {next(iter(feat_dict.values())).shape}")

# # Precompute mean pooled vectors (for the initial direction fit per split)
# X_mean = np.vstack([mean_pool(feat_dict[sid]) for sid in slide_ids]).astype(np.float32)

# # ==============================
# # Holdout splits
# # ==============================
# gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

# rows = []
# for split_id, (tr_idx, te_idx) in enumerate(gss.split(X_mean, y_all, groups), start=1):
#     # 1) Fit LR on MEAN pooled slide vectors to get a direction w
#     clf_dir = LogisticRegression(
#         solver="liblinear",
#         class_weight="balanced",
#         random_state=SEED,
#         max_iter=5000,
#     )
#     clf_dir.fit(X_mean[tr_idx], y_all[tr_idx])
#     w = clf_dir.coef_.ravel().astype(np.float32)  # (D,)

#     # 2) Build TOP-K pooled slide vectors using that direction
#     Xtr = np.vstack([topk_pool_by_direction(feat_dict[slide_ids[i]], w, TOPK) for i in tr_idx]).astype(np.float32)
#     ytr = y_all[tr_idx]

#     Xte = np.vstack([topk_pool_by_direction(feat_dict[slide_ids[i]], w, TOPK) for i in te_idx]).astype(np.float32)
#     yte = y_all[te_idx]

#     # 3) Train LR on TOP-K pooled vectors and evaluate
#     clf = LogisticRegression(
#         solver="liblinear",
#         class_weight="balanced",
#         random_state=SEED,
#         max_iter=5000,
#     )
#     clf.fit(Xtr, ytr)
#     prob = clf.predict_proba(Xte)[:, 1]

#     m = metrics(yte, prob)
#     rows.append({
#         "split": split_id,
#         "auc": m["auc"],
#         "balacc": m["balacc"],
#         "acc": m["acc"],
#         "n_test_slides": len(te_idx),
#         "n_test_patients": len(np.unique(groups[te_idx])),
#     })

# dfm = pd.DataFrame(rows)
# print(f"\n=== CONCH image-only TOPK pooling (TOPK={TOPK}) | {N_SPLITS} patient holdouts ===")
# print(dfm[["split","auc","balacc","acc","n_test_slides","n_test_patients"]])

# print("\n=== Summary (mean ± std) ===")
# print(dfm[["auc","balacc","acc"]].agg(["mean","std"]).T)

# ==============================

import os, re
import numpy as np
import pandas as pd
import h5py
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix


# ============================================================
# EDIT THESE
# # ============================================================
# FEAT20_ROOT = r"C:\Users\Vivian\Documents\CONCH\conch_img_feats\2.5x_feats\20x\pt"
# FEAT40_ROOT = r"C:\Users\Vivian\Documents\CONCH\conch_img_feats\2.5x_feats\40x\pt"
FEAT_UNI_ROOT = r'C:\Users\Vivian\Documents\CLAM\CLAM\FEATURES_DIR_5x\FEATURES_DIR_2.5x\uniextracted_mag2x_patch224_fp\feats_pt'
FEAT_EXT = ".pt"   # ".pt" or ".h5" (must match your outputs)

CV_ROOT = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val"
FOLD_PREFIX = "FA_PT_k="

OUT_DIR = r"C:\Users\Vivian\Documents\CONCH\conch_img_feats\2.5x_UNI_eval_patient_predefined_20x40x"
os.makedirs(OUT_DIR, exist_ok=True)

USE_TRAIN_PLUS_VAL = True
SEED = 2348

TOPK = 64
MAX_PATCHES = 0    # 0 = all, else cap for speed


# ============================================================
# Helpers
# ============================================================
def normalize_slide_id(x: str) -> str:
    return os.path.splitext(str(x).strip())[0]

def slide_to_patient(slide_id: str) -> str:
    # "FA 47 B1" -> "FA 47"
    return re.sub(r"\s+B\d+\s*$", "", str(slide_id).strip())

def label_from_slide_id(slide_id: str) -> int:
    s = str(slide_id).strip().upper()
    if s.startswith("FA"):
        return 0
    if s.startswith("PT"):
        return 1
    raise ValueError(f"Cannot infer label from slide_id: {slide_id}")

def read_slide_list(csv_path: str):
    d = pd.read_csv(csv_path)
    if "slide" in d.columns:
        slides = d["slide"].astype(str).tolist()
    elif "slide_id" in d.columns:
        slides = d["slide_id"].astype(str).tolist()
    else:
        slides = d.iloc[:, 0].astype(str).tolist()
    return [normalize_slide_id(s) for s in slides]

def find_fold_dirs(cv_root: str):
    out = []
    for d in os.listdir(cv_root):
        p = os.path.join(cv_root, d)
        if os.path.isdir(p) and d.startswith(FOLD_PREFIX):
            out.append(p)
    return sorted(out)

def list_slides_in_feat_root(feat_root: str, ext: str):
    # filenames: "<slide_id>.pt" or ".h5"
    slides = []
    for fn in os.listdir(feat_root):
        if fn.endswith(ext):
            slides.append(os.path.splitext(fn)[0])
    return sorted(slides)

# def load_patch_feats(feat_root: str, slide_id: str):
#     p = os.path.join(feat_root, slide_id + FEAT_EXT)
#     if not os.path.exists(p):
#         return None

#     if FEAT_EXT == ".pt":
#         obj = torch.load(p, map_location="cpu")
#         feats = obj["features"].numpy()
#     else:
#         with h5py.File(p, "r") as f:
#             feats = f["features"][:]

#     if feats.ndim != 2 or feats.shape[0] == 0:
#         return None

#     if MAX_PATCHES > 0 and feats.shape[0] > MAX_PATCHES:
#         feats = feats[:MAX_PATCHES]

#     return feats.astype(np.float32)

def load_patch_feats(feat_root: str, slide_id: str):
    p = os.path.join(feat_root, slide_id + FEAT_EXT)
    if not os.path.exists(p):
        return None

    if FEAT_EXT == ".pt":
        obj = torch.load(p, map_location="cpu")

        # UNI: tensor (N,D)
        if torch.is_tensor(obj):
            feats = obj

        # CONCH: dict with "features"
        elif isinstance(obj, dict):
            feats = obj["features"]  # you confirmed this key exists

        else:
            raise TypeError(f"{p}: unexpected .pt type {type(obj)}")

        feats = feats.detach().cpu().numpy()

    else:
        with h5py.File(p, "r") as f:
            feats = f["features"][:] if "features" in f else f["feats"][:]

    if feats.ndim != 2 or feats.shape[0] == 0:
        return None

    if MAX_PATCHES > 0 and feats.shape[0] > MAX_PATCHES:
        feats = feats[:MAX_PATCHES]

    return feats.astype(np.float32)


def mean_pool(feats: np.ndarray) -> np.ndarray:
    return feats.mean(axis=0)

def topk_pool_by_direction(feats: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    k = min(k, feats.shape[0])
    scores = feats @ w
    idx = np.argpartition(scores, -k)[-k:]
    return feats[idx].mean(axis=0)

def eval_prob(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "balacc": balanced_accuracy_score(y_true, pred),
        "acc": accuracy_score(y_true, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def agg_patient_probs(probs: np.ndarray, mode: str = "noisy_or") -> float:
    """
    probs: array of slide probabilities for one patient (from any scale)
    """
    probs = np.asarray(probs, dtype=np.float32)
    if len(probs) == 0:
        return np.nan
    if mode == "max":
        return float(np.max(probs))
    if mode == "mean":
        return float(np.mean(probs))
    if mode == "noisy_or":
        # 1 - Π(1 - p_i)
        return float(1.0 - np.prod(1.0 - probs))
    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Train + predict per scale (slide-level), then aggregate to patient-level
# ============================================================
def run_scale_for_fold(scale_name: str, feat_root: str, train_patients: set, test_patients: set):
    """
    Returns:
      slide_pred_df columns: slide_id, patient_id, y, prob_pt, scale
    """
    # discover slides we actually have for this scale
    all_slides = list_slides_in_feat_root(feat_root, FEAT_EXT)

    # partition slides by patient membership (from fold split)
    train_slides = []
    test_slides = []
    for sid in all_slides:
        pid = slide_to_patient(sid)
        if pid in train_patients:
            train_slides.append(sid)
        elif pid in test_patients:
            test_slides.append(sid)

    # load patch feats once for speed
    train_cache = {}
    test_cache = {}

    def load_cache(slides, cache):
        kept = []
        y = []
        Xmean = []
        for sid in slides:
            feats = load_patch_feats(feat_root, sid)
            if feats is None:
                continue
            cache[sid] = feats
            kept.append(sid)
            y.append(label_from_slide_id(sid))
            Xmean.append(mean_pool(feats))
        if len(kept) == 0:
            return None, None, []
        return np.vstack(Xmean).astype(np.float32), np.array(y, dtype=int), kept

    Xtr_mean, ytr, train_kept = load_cache(train_slides, train_cache)
    Xte_mean, yte, test_kept = load_cache(test_slides, test_cache)

    if Xtr_mean is None or Xte_mean is None:
        return pd.DataFrame()

    # 1) direction model on mean pooled
    clf_dir = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )
    clf_dir.fit(Xtr_mean, ytr)
    w = clf_dir.coef_.ravel().astype(np.float32)

    # 2) top-k pooled vectors
    Xtr = np.vstack([topk_pool_by_direction(train_cache[sid], w, TOPK) for sid in train_kept]).astype(np.float32)
    Xte = np.vstack([topk_pool_by_direction(test_cache[sid],  w, TOPK) for sid in test_kept ]).astype(np.float32)

    # 3) slide-level classifier on top-k pooled
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )
    clf.fit(Xtr, ytr)
    prob_pt = clf.predict_proba(Xte)[:, 1]

    df_pred = pd.DataFrame({
        "slide_id": test_kept,
        "patient_id": [slide_to_patient(s) for s in test_kept],
        "y": yte,
        "prob_pt": prob_pt,
        "scale": scale_name,
    })
    return df_pred

def run_uni_for_fold(feat_root, train_patients, test_patients):
    all_slides = list_slides_in_feat_root(feat_root, FEAT_EXT)

    train_slides, test_slides = [], []
    for sid in all_slides:
        pid = slide_to_patient(sid)
        if pid in train_patients:
            train_slides.append(sid)
        elif pid in test_patients:
            test_slides.append(sid)

    train_cache, test_cache = {}, {}

    def load_cache(slides, cache):
        kept, y, Xmean = [], [], []
        for sid in slides:
            feats = load_patch_feats(feat_root, sid)
            if feats is None:
                continue
            cache[sid] = feats
            kept.append(sid)
            y.append(label_from_slide_id(sid))
            Xmean.append(mean_pool(feats))
        if len(kept) == 0:
            return None, None, []
        return np.vstack(Xmean).astype(np.float32), np.array(y), kept

    Xtr_mean, ytr, train_kept = load_cache(train_slides, train_cache)
    Xte_mean, yte, test_kept = load_cache(test_slides, test_cache)
    if Xtr_mean is None or Xte_mean is None:
        return pd.DataFrame()

    # direction from mean pooled
    clf_dir = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )
    clf_dir.fit(Xtr_mean, ytr)
    w = clf_dir.coef_.ravel()

    # top-K pooling
    Xtr = np.vstack([topk_pool_by_direction(train_cache[s], w, TOPK) for s in train_kept])
    Xte = np.vstack([topk_pool_by_direction(test_cache[s],  w, TOPK) for s in test_kept])

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]

    return pd.DataFrame({
        "slide_id": test_kept,
        "patient_id": [slide_to_patient(s) for s in test_kept],
        "y": yte,
        "prob_pt": prob,
    })

def main():
    fold_dirs = find_fold_dirs(CV_ROOT)
    if not fold_dirs:
        raise ValueError(f"No fold dirs found under {CV_ROOT} with prefix {FOLD_PREFIX}")

    fold_rows = []
    all_slide_preds = []
    all_patient_preds = []

    # Choose patient aggregation mode(s)
    PAT_AGG_MODES = ["max", "noisy_or", "mean"]

    for fold_dir in fold_dirs:
        fold = os.path.basename(fold_dir)
        train_sl = read_slide_list(os.path.join(fold_dir, "train.csv"))
        val_sl   = read_slide_list(os.path.join(fold_dir, "val.csv"))
        test_sl  = read_slide_list(os.path.join(fold_dir, "test.csv"))

        fit_slides = train_sl + (val_sl if USE_TRAIN_PLUS_VAL else [])
        fit_slides = list(dict.fromkeys(fit_slides))

        train_patients = set(slide_to_patient(s) for s in fit_slides)
        test_patients  = set(slide_to_patient(s) for s in test_sl)

        # run per scale slide prediction ------------------ conch feats
        # pred20 = run_scale_for_fold("20x", FEAT20_ROOT, train_patients, test_patients)
        # pred40 = run_scale_for_fold("40x", FEAT40_ROOT, train_patients, test_patients)

        # slide_pred = pd.concat([pred20, pred40], ignore_index=True)
        # -------------------------- conch feats

        # -------------------------- uni feats
        slide_pred = run_uni_for_fold(FEAT_UNI_ROOT, train_patients, test_patients)
        slide_pred["scale"] = "UNI"
        # --------------------------------- 

        slide_pred["fold"] = fold

        if len(slide_pred) == 0:
            print(f"[{fold}] WARNING: no slide preds produced (missing features?)")
            continue

        all_slide_preds.append(slide_pred)
        slide_pred.to_csv(os.path.join(OUT_DIR, f"{fold}_slide_preds_20x40x_topk{TOPK}.csv"), index=False)

        # slide-level metrics per scale (optional but useful)
        for scale in ["20x", "40x"]:
            d = slide_pred[slide_pred["scale"] == scale]
            if len(d) >= 5 and len(np.unique(d["y"])) == 2:
                m = eval_prob(d["y"].values, d["prob_pt"].values)
                fold_rows.append({"fold": fold, "level": "slide", "model": f"img_{scale}_topk{TOPK}", "n": len(d), **m})

        # patient-level aggregation across ALL slides (both scales) for test patients
        # group by patient_id
        for mode in PAT_AGG_MODES:
            patient_rows = []
            for pid, g in slide_pred.groupby("patient_id"):
                # patient label from pid prefix (FA/PT)
                y_pat = label_from_slide_id(pid)
                p_pat = agg_patient_probs(g["prob_pt"].values, mode=mode)
                patient_rows.append({"patient_id": pid, "y": y_pat, "prob_pt": p_pat})

            df_pat = pd.DataFrame(patient_rows)
            df_pat["fold"] = fold
            df_pat["agg"] = mode

            # metrics
            if len(df_pat) >= 5 and len(np.unique(df_pat["y"])) == 2:
                mpat = eval_prob(df_pat["y"].values, df_pat["prob_pt"].values)
                fold_rows.append({"fold": fold, "level": "patient", "model": f"patient_{mode}_20x40x_topk{TOPK}", "n": len(df_pat), **mpat})

            all_patient_preds.append(df_pat)
            df_pat.to_csv(os.path.join(OUT_DIR, f"{fold}_patient_preds_{mode}_topk{TOPK}.csv"), index=False)

        # print(f"[{fold}] slides: {len(slide_pred)} | patients: {slide_pred['patient_id'].nunique()} | (20x slides={len(pred20)} | 40x slides={len(pred40)})")

    results = pd.DataFrame(fold_rows)
    results.to_csv(os.path.join(OUT_DIR, f"fold_metrics_patient_20x40x_topk{TOPK}.csv"), index=False)

    print("\n=== Fold metrics ===")
    if len(results):
        print(results[["fold","level","model","n","auc","balacc","acc","tn","fp","fn","tp"]])
        print("\n=== Summary (mean ± std) ===")
        print(results.groupby(["level","model"])[["auc","balacc","acc"]].agg(["mean","std"]))
    else:
        print("No results produced (check feature paths / folds / labels).")

    if all_slide_preds:
        pd.concat(all_slide_preds, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, f"all_slide_preds_20x40x_topk{TOPK}.csv"),
            index=False
        )
    if all_patient_preds:
        pd.concat(all_patient_preds, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, f"all_patient_preds_20x40x_topk{TOPK}.csv"),
            index=False
        )

    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
