# import os, re
# import numpy as np
# import pandas as pd

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix

# ABLATIONS = {
#     "ALL": [
#         "bucket_FA_lean",
#         "bucket_PT_lean",
#         "bucket_PT_benign_like",
#         "bucket_PT_borderline_like",
#         "bucket_PT_malignant_like",
#         "bucket_mitosis",
#         "bucket_NORMAL_lean",
#         "bucket_controls",
#     ],

#     "NO_MITOSIS": [
#         "bucket_FA_lean",
#         "bucket_PT_lean",
#         "bucket_PT_benign_like",
#         "bucket_PT_borderline_like",
#         "bucket_PT_malignant_like",
#         "bucket_NORMAL_lean",
#         "bucket_controls",
#     ],

#     "PT_ONLY": [
#         "bucket_PT_lean",
#         "bucket_PT_benign_like",
#         "bucket_PT_borderline_like",
#         "bucket_PT_malignant_like",
#         "bucket_mitosis",
#     ],

#     "FA_NORMAL_ONLY": [
#         "bucket_FA_lean",
#         "bucket_NORMAL_lean",
#         "bucket_controls",
#     ],

#     "MITOSIS_ONLY": [
#         "bucket_mitosis",
#     ],
# }


# # -------------------------
# # Paths
# # -------------------------
# CONCH_CSVS = [
#     r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\10x_noNORM_conch_zeroshot_slides_fullprompts_20x.csv",
#     r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\10x_noNORM_conch_zeroshot_slides_fullprompts_40x.csv",
# ]


# CV_ROOT = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val"
# # If your folds are directly under CV_ROOT like ...\cross-val\FA_PT_k=0\{train,val,test}.csv
# # and you only want FA_PT folds, we’ll filter for that prefix.

# OUT_DIR = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\eval_predefined_splits"
# os.makedirs(OUT_DIR, exist_ok=True)

# # -------------------------
# # Columns
# # -------------------------
# SLIDE_COL = "slide_id"
# LABEL_COL = "true_label_str"  # FA/PT in your CONCH csv
# SEED = 2348

# # If True: train on train+val (recommended for final reporting)
# USE_TRAIN_PLUS_VAL = True


# # -------------------------
# # Split CSV reading (your snippet + minor safety)
# # -------------------------
# def read_slide_list(csv_path: str):
#     d = pd.read_csv(csv_path)
#     if "slide" in d.columns:
#         slides = d["slide"].astype(str).tolist()
#     elif "slide_id" in d.columns:
#         slides = d["slide_id"].astype(str).tolist()
#     else:
#         slides = d.iloc[:, 0].astype(str).tolist()
#     slides = [os.path.splitext(str(s).strip())[0] for s in slides]
#     return slides

# def normalize_slide_id_series(s: pd.Series) -> pd.Series:
#     return s.astype(str).str.strip().map(lambda x: os.path.splitext(x)[0])

# def find_fold_dirs(cv_root: str):
#     # Find directories like FA_PT_k=0, FA_PT_k=1, ...
#     fold_dirs = []
#     for d in os.listdir(cv_root):
#         p = os.path.join(cv_root, d)
#         if os.path.isdir(p) and d.startswith("FA_PT_k="):
#             fold_dirs.append(p)
#     return sorted(fold_dirs)


# # -------------------------
# # Helpers
# # -------------------------
# def labels_to_int(y_str: pd.Series) -> np.ndarray:
#     y = y_str.astype(str).str.upper().str.strip().map({"FA": 0, "PT": 1})
#     if y.isna().any():
#         bad = y_str[y.isna()].unique()
#         raise ValueError(f"Unexpected labels in {LABEL_COL}: {bad} (expected FA/PT)")
#     return y.astype(int).to_numpy()

# def find_bucket_cols(df: pd.DataFrame):
#     df = df.copy()
#     df.columns = [str(c).strip() for c in df.columns]
#     cols = [c for c in df.columns if c.lower().startswith("bucket_")]
#     if not cols:
#         raise ValueError(f"No bucket_ columns found. Columns seen: {list(df.columns)[:30]}")
#     return cols

# def slide_level_aggregate(df_raw: pd.DataFrame):
#     df = df_raw.copy()
#     df.columns = [str(c).strip() for c in df.columns]

#     if SLIDE_COL not in df.columns:
#         raise ValueError(f"Missing {SLIDE_COL}")
#     if LABEL_COL not in df.columns:
#         raise ValueError(f"Missing {LABEL_COL}")

#     df[SLIDE_COL] = normalize_slide_id_series(df[SLIDE_COL])

#     bucket_cols = find_bucket_cols(df)
#     for c in bucket_cols:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     agg = {c: "mean" for c in bucket_cols}
#     agg[LABEL_COL] = "first"
#     if "mag" in df.columns:
#         agg["mag"] = "first"

#     df_slide = df.groupby(SLIDE_COL, as_index=False).agg(agg)

#     # clean
#     df_slide[bucket_cols] = df_slide[bucket_cols].replace([np.inf, -np.inf], np.nan)
#     df_slide[bucket_cols] = df_slide[bucket_cols].fillna(df_slide[bucket_cols].median(numeric_only=True))

#     return df_slide, bucket_cols

# def eval_prob(y_true, prob, thr=0.5):
#     pred = (prob >= thr).astype(int)
#     tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
#     return {
#         "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
#         "balacc": balanced_accuracy_score(y_true, pred),
#         "acc": accuracy_score(y_true, pred),
#         "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
#     }


# # -------------------------
# # Main eval
# # -------------------------
# def main():
#     # Load and combine conch csvs
#     dfs = []
#     for p in CONCH_CSVS:
#         d = pd.read_csv(p)
#         dfs.append(d)
#     df_all = pd.concat(dfs, ignore_index=True)

#     # slide-level table
#     df_slide, bucket_cols = slide_level_aggregate(df_all)
#     print(f"CONCH slide rows: {len(df_slide)}")
#     print(f"Bucket features: {len(bucket_cols)}")

#     # index for fast lookup
#     df_slide = df_slide.set_index(SLIDE_COL, drop=False)

#     fold_dirs = find_fold_dirs(CV_ROOT)
#     if not fold_dirs:
#         raise ValueError(f"No fold dirs found under {CV_ROOT} with prefix FA_PT_k=")

#     fold_rows = []
#     all_test_preds = []

#     for fold_idx, fold_dir in enumerate(fold_dirs):
#         train_csv = os.path.join(fold_dir, "train.csv")
#         val_csv   = os.path.join(fold_dir, "val.csv")
#         test_csv  = os.path.join(fold_dir, "test.csv")

#         train_slides = read_slide_list(train_csv)
#         val_slides   = read_slide_list(val_csv)
#         test_slides  = read_slide_list(test_csv)

#         # normalize to match df_slide index
#         train_slides = [os.path.splitext(s.strip())[0] for s in train_slides]
#         val_slides   = [os.path.splitext(s.strip())[0] for s in val_slides]
#         test_slides  = [os.path.splitext(s.strip())[0] for s in test_slides]

#         fit_slides = train_slides + (val_slides if USE_TRAIN_PLUS_VAL else [])
#         fit_slides = list(dict.fromkeys(fit_slides))  # unique preserve order

#         # Keep only slides that exist in df_slide
#         fit_slides_in = [s for s in fit_slides if s in df_slide.index]
#         test_slides_in = [s for s in test_slides if s in df_slide.index]

#         missing_fit = len(fit_slides) - len(fit_slides_in)
#         missing_test = len(test_slides) - len(test_slides_in)

#         if len(test_slides_in) == 0:
#             print(f"[fold {fold_idx}] WARNING: no test slides matched; skipping")
#             continue

#         for ab_name, feat_cols in ABLATIONS.items():
#             feat_cols = [c for c in feat_cols if c in df_slide.columns]
#             if len(feat_cols) == 0:
#                 continue

#             Xtr = df_slide.loc[fit_slides_in, feat_cols]
#             ytr = labels_to_int(df_slide.loc[fit_slides_in, LABEL_COL])

#             Xte = df_slide.loc[test_slides_in, feat_cols]
#             yte = labels_to_int(df_slide.loc[test_slides_in, LABEL_COL])

#             clf = LogisticRegression(
#                 solver="liblinear",
#                 class_weight="balanced",
#                 random_state=SEED,
#                 max_iter=5000,
#             )
#             clf.fit(Xtr, ytr)
#             prob = clf.predict_proba(Xte)[:, 1]

#             m = eval_prob(yte, prob)

#             fold_rows.append({
#                 "fold": os.path.basename(fold_dir),
#                 "ablation": ab_name,
#                 "n_features": len(feat_cols),
#                 **m,
#             })

#         # Xtr = df_slide.loc[fit_slides_in, bucket_cols]
#         # ytr = labels_to_int(df_slide.loc[fit_slides_in, LABEL_COL])

#         # Xte = df_slide.loc[test_slides_in, bucket_cols]
#         # yte = labels_to_int(df_slide.loc[test_slides_in, LABEL_COL])

#         # clf = LogisticRegression(
#         #     solver="liblinear",
#         #     class_weight="balanced",
#         #     random_state=SEED,
#         #     max_iter=5000,
#         # )
#         # clf.fit(Xtr, ytr)
#         # prob = clf.predict_proba(Xte)[:, 1]

#         # m = eval_prob(yte, prob, thr=0.5)
#         # fold_rows.append({
#         #     "fold": os.path.basename(fold_dir),
#         #     "n_train": len(fit_slides_in),
#         #     "n_test": len(test_slides_in),
#         #     "missing_fit": missing_fit,
#         #     "missing_test": missing_test,
#         #     **m
#         # })

#         # save per-slide test preds
#         df_pred = pd.DataFrame({
#             "fold": os.path.basename(fold_dir),
#             "slide_id": test_slides_in,
#             "true_label": df_slide.loc[test_slides_in, LABEL_COL].values,
#             "prob_pt": prob,
#             "pred_label": np.where(prob >= 0.5, "PT", "FA"),
#         })
#         all_test_preds.append(df_pred)

#         df_pred.to_csv(os.path.join(OUT_DIR, f"{os.path.basename(fold_dir)}_test_preds.csv"), index=False)

#         print(f"[{os.path.basename(fold_dir)}] n_test={len(test_slides_in)} | AUC={m['auc']:.3f} | BalAcc={m['balacc']:.3f} | Acc={m['acc']:.3f}")

#     # results = pd.DataFrame(fold_rows)

#     results = pd.DataFrame(fold_rows)

#     summary = (
#         results
#         .groupby("ablation")[["auc", "balacc", "acc"]]
#         .agg(["mean", "std"])
#     )

#     print("\n=== Bucket ablation summary (mean ± std) ===")
#     print(summary)
#     results.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False)


#     print("\n=== Fold metrics ===")
#     print(results[["fold","n_train","n_test","auc","balacc","acc","tn","fp","fn","tp","missing_fit","missing_test"]])

#     if len(results) > 0:
#         summary = results[["auc","balacc","acc"]].agg(["mean","std"]).T
#         print("\n=== Summary (mean ± std) ===")
#         print(summary)

#     if all_test_preds:
#         pd.concat(all_test_preds, ignore_index=True).to_csv(os.path.join(OUT_DIR, "all_test_preds.csv"), index=False)
#         print(f"\nSaved outputs to: {OUT_DIR}")

# if __name__ == "__main__":
#     main()


# -------------
import os
import re
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
)

# ============================================================
# Ablations (feature subsets)
# ============================================================
ABLATIONS = {
    "ALL": [
        "bucket_FA_lean",
        "bucket_PT_lean",
        "bucket_PT_benign_like",
        "bucket_PT_borderline_like",
        "bucket_PT_malignant_like",
        "bucket_mitosis",
        "bucket_NORMAL_lean",
        "bucket_controls",
    ],
    "NO_MITOSIS": [
        "bucket_FA_lean",
        "bucket_PT_lean",
        "bucket_PT_benign_like",
        "bucket_PT_borderline_like",
        "bucket_PT_malignant_like",
        "bucket_NORMAL_lean",
        "bucket_controls",
    ],
    "PT_ONLY": [
        "bucket_PT_lean",
        "bucket_PT_benign_like",
        "bucket_PT_borderline_like",
        "bucket_PT_malignant_like",
        "bucket_mitosis",
    ],
    "FA_NORMAL_ONLY": [
        "bucket_FA_lean",
        "bucket_NORMAL_lean",
        "bucket_controls",
    ],
    "MITOSIS_ONLY": [
        "bucket_mitosis",
    ],
}

# ============================================================
# Paths
# ============================================================
CONCH_CSVS = [
    r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\10x_noNORM_conch_zeroshot_slides_fullprompts_20x.csv",
    r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\10x_noNORM_conch_zeroshot_slides_fullprompts_40x.csv",
]

CV_ROOT = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val"
OUT_DIR = r"C:\Users\Vivian\Documents\CONCH\test_text_encoder\eval_predefined_splits2"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Columns / config
# ============================================================
SLIDE_COL = "slide_id"
LABEL_COL = "true_label_str"  # FA/PT in your CONCH csv
SEED = 2348
USE_TRAIN_PLUS_VAL = True  # train on train+val for each fold


# ============================================================
# Split CSV reading (your snippet + minor safety)
# ============================================================
def read_slide_list(csv_path: str):
    d = pd.read_csv(csv_path)
    if "slide" in d.columns:
        slides = d["slide"].astype(str).tolist()
    elif "slide_id" in d.columns:
        slides = d["slide_id"].astype(str).tolist()
    else:
        slides = d.iloc[:, 0].astype(str).tolist()
    slides = [os.path.splitext(str(s).strip())[0] for s in slides]
    return slides


def normalize_slide_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().map(lambda x: os.path.splitext(x)[0])


def find_fold_dirs(cv_root: str):
    # Find directories like FA_PT_k=0, FA_PT_k=1, ...
    fold_dirs = []
    for d in os.listdir(cv_root):
        p = os.path.join(cv_root, d)
        if os.path.isdir(p) and d.startswith("FA_PT_k="):
            fold_dirs.append(p)
    return sorted(fold_dirs)


# ============================================================
# Helpers
# ============================================================
def labels_to_int(y_str: pd.Series) -> np.ndarray:
    y = y_str.astype(str).str.upper().str.strip().map({"FA": 0, "PT": 1})
    if y.isna().any():
        bad = y_str[y.isna()].unique()
        raise ValueError(f"Unexpected labels in {LABEL_COL}: {bad} (expected FA/PT)")
    return y.astype(int).to_numpy()


def find_bucket_cols(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = [c for c in df.columns if c.lower().startswith("bucket_")]
    if not cols:
        raise ValueError(f"No bucket_ columns found. Columns seen: {list(df.columns)[:30]}")
    return cols


def slide_level_aggregate(df_raw: pd.DataFrame):
    """
    Ensures one row per slide_id by aggregating bucket_* columns with mean.
    Works for both:
      - "wide" files (already one row/slide)
      - "long" files (multiple rows/slide, e.g., top-concepts long format)
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if SLIDE_COL not in df.columns:
        raise ValueError(f"Missing {SLIDE_COL}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing {LABEL_COL}")

    df[SLIDE_COL] = normalize_slide_id_series(df[SLIDE_COL])

    bucket_cols = find_bucket_cols(df)
    for c in bucket_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = {c: "mean" for c in bucket_cols}
    agg[LABEL_COL] = "first"
    if "mag" in df.columns:
        agg["mag"] = "first"

    df_slide = df.groupby(SLIDE_COL, as_index=False).agg(agg)

    # clean
    df_slide[bucket_cols] = df_slide[bucket_cols].replace([np.inf, -np.inf], np.nan)
    df_slide[bucket_cols] = df_slide[bucket_cols].fillna(df_slide[bucket_cols].median(numeric_only=True))

    return df_slide, bucket_cols


def eval_prob(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "balacc": balanced_accuracy_score(y_true, pred),
        "acc": accuracy_score(y_true, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# Main eval
# ============================================================
def main():
    # ----------------------------
    # Load and combine CONCH CSVs
    # ----------------------------
    dfs = []
    for p in CONCH_CSVS:
        dfs.append(pd.read_csv(p))
    df_all = pd.concat(dfs, ignore_index=True)

    # Slide-level table (one row per slide)
    df_slide, bucket_cols = slide_level_aggregate(df_all)
    print(f"CONCH slide rows: {len(df_slide)}")
    print(f"Bucket features discovered: {len(bucket_cols)}")
    # Index for fast lookup
    df_slide = df_slide.set_index(SLIDE_COL, drop=False)

    # ----------------------------
    # Folds
    # ----------------------------
    fold_dirs = find_fold_dirs(CV_ROOT)
    if not fold_dirs:
        raise ValueError(f"No fold dirs found under {CV_ROOT} with prefix FA_PT_k=")

    # ----------------------------
    # Storage
    # ----------------------------
    fold_rows = []
    coef_rows = []  # coefficient table (for importance)
    all_test_preds = []  # store test preds for ALL ablation only (by default)

    # ----------------------------
    # Run per fold
    # ----------------------------
    for fold_idx, fold_dir in enumerate(fold_dirs):
        fold_name = os.path.basename(fold_dir)

        train_csv = os.path.join(fold_dir, "train.csv")
        val_csv = os.path.join(fold_dir, "val.csv")
        test_csv = os.path.join(fold_dir, "test.csv")

        train_slides = read_slide_list(train_csv)
        val_slides = read_slide_list(val_csv)
        test_slides = read_slide_list(test_csv)

        # normalize to match df_slide index
        train_slides = [os.path.splitext(s.strip())[0] for s in train_slides]
        val_slides = [os.path.splitext(s.strip())[0] for s in val_slides]
        test_slides = [os.path.splitext(s.strip())[0] for s in test_slides]

        fit_slides = train_slides + (val_slides if USE_TRAIN_PLUS_VAL else [])
        fit_slides = list(dict.fromkeys(fit_slides))  # unique preserve order

        # Keep only slides that exist in df_slide
        fit_slides_in = [s for s in fit_slides if s in df_slide.index]
        test_slides_in = [s for s in test_slides if s in df_slide.index]

        missing_fit = len(fit_slides) - len(fit_slides_in)
        missing_test = len(test_slides) - len(test_slides_in)

        if len(test_slides_in) == 0:
            print(f"[{fold_name}] WARNING: no test slides matched; skipping")
            continue

        # ----------------------------
        # Run ablations for this fold
        # ----------------------------
        for ab_name, feat_cols in ABLATIONS.items():
            # Keep only cols present in df_slide
            feat_cols = [c for c in feat_cols if c in df_slide.columns]
            if len(feat_cols) == 0:
                continue

            Xtr = df_slide.loc[fit_slides_in, feat_cols]
            ytr = labels_to_int(df_slide.loc[fit_slides_in, LABEL_COL])

            Xte = df_slide.loc[test_slides_in, feat_cols]
            yte = labels_to_int(df_slide.loc[test_slides_in, LABEL_COL])

            clf = LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                random_state=SEED,
                max_iter=5000,
            )
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)[:, 1]

            m = eval_prob(yte, prob, thr=0.5)

            fold_rows.append(
                {
                    "fold": fold_name,
                    "ablation": ab_name,
                    "n_train": len(fit_slides_in),
                    "n_test": len(test_slides_in),
                    "n_features": len(feat_cols),
                    "missing_fit": missing_fit,
                    "missing_test": missing_test,
                    **m,
                }
            )

            # Save coefficients for ALL (and optionally for other ablations too)
            # Here: save for ALL only, to keep it clean
            if ab_name == "ALL":
                coef_row = {"fold": fold_name, "ablation": ab_name}
                for c, w in zip(feat_cols, clf.coef_.ravel()):
                    coef_row[c] = float(w)
                coef_rows.append(coef_row)

                # Save per-slide test preds for ALL
                df_pred = pd.DataFrame(
                    {
                        "fold": fold_name,
                        "ablation": ab_name,
                        "slide_id": test_slides_in,
                        "true_label": df_slide.loc[test_slides_in, LABEL_COL].values,
                        "prob_pt": prob,
                        "pred_label": np.where(prob >= 0.5, "PT", "FA"),
                    }
                )
                all_test_preds.append(df_pred)

        print(f"[{fold_name}] done. n_test={len(test_slides_in)} missing_fit={missing_fit} missing_test={missing_test}")

    # ----------------------------
    # Results tables
    # ----------------------------
    results = pd.DataFrame(fold_rows)
    results_path = os.path.join(OUT_DIR, "ablation_fold_metrics.csv")
    results.to_csv(results_path, index=False)
    print(f"\nSaved fold metrics: {results_path}")

    # Ablation summary
    ab_summary = (
        results.groupby("ablation")[["auc", "balacc", "acc"]]
        .agg(["mean", "std"])
        .sort_index()
    )
    print("\n=== Bucket ablation summary (mean ± std) ===")
    print(ab_summary)

    ab_summary_path = os.path.join(OUT_DIR, "ablation_summary.csv")
    # flatten MultiIndex columns for CSV
    ab_summary_out = ab_summary.copy()
    ab_summary_out.columns = [f"{a}_{b}" for a, b in ab_summary_out.columns]
    ab_summary_out.to_csv(ab_summary_path)
    print(f"Saved ablation summary: {ab_summary_path}")

    # ----------------------------
    # Coefficient importance (ALL only)
    # ----------------------------
    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        coef_path = os.path.join(OUT_DIR, "coefficients_all_folds.csv")
        coef_df.to_csv(coef_path, index=False)
        print(f"\nSaved coefficients (ALL): {coef_path}")

        coef_only = coef_df.drop(columns=["fold", "ablation"], errors="ignore")
        coef_summary = coef_only.agg(["mean", "std"]).T
        coef_summary["abs_mean"] = coef_summary["mean"].abs()
        coef_summary = coef_summary.sort_values("abs_mean", ascending=False)

        print("\n=== Logistic coefficient importance (ALL buckets) ===")
        print(coef_summary[["mean", "std", "abs_mean"]])

        coef_summary_path = os.path.join(OUT_DIR, "coef_importance_all.csv")
        coef_summary.to_csv(coef_summary_path)
        print(f"Saved coef importance: {coef_summary_path}")

    # ----------------------------
    # Save per-slide test predictions for ALL
    # ----------------------------
    if all_test_preds:
        all_preds = pd.concat(all_test_preds, ignore_index=True)
        preds_path = os.path.join(OUT_DIR, "all_test_preds_ALL.csv")
        all_preds.to_csv(preds_path, index=False)
        print(f"\nSaved per-slide test preds (ALL): {preds_path}")

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
