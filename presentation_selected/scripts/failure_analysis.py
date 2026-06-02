from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "failure_analysis_outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
IMAGES = OUT / "images"
PRESENTATION = OUT / "presentation_selected"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
PROB_COLS = [f"prob_{c}" for c in CLASSES]
RISKY_ERRORS = {("mel", "nv"), ("mel", "bkl"), ("akiec", "bkl"), ("bcc", "bkl")}
HIGH_CONF_THRESHOLD = 0.90

TRAD_DIR = ROOT / "传统方法" / "outputs"
TRAD_BEST = TRAD_DIR / "artifacts_best_practice" / "A_single_stage_control"
DEEP_DIR = ROOT / "深度学习方法" / "experiments" / "ensemble_line1_line2_stepA"
INTERP_DIR = ROOT / "interpretability"
TRAD_CODE_DIR = ROOT / "传统方法"
TRAD_VAL_SOURCE = "preprocessing_metadata_origin_images"


@dataclass
class ModelBundle:
    name: str
    family: str
    metrics: dict
    per_class: pd.DataFrame


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    for path in [FIG, TABLES, REPORTS, IMAGES, PRESENTATION]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    legacy_dir = OUT / "high_confidence_error_images"
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir)
    for path in OUT.glob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
            path.unlink()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.rename(columns={first: "class"})
    elif first != "class":
        df = df.rename(columns={first: "class"})
    for col in ["precision", "recall", "f1-score", "support"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["class"].isin(CLASSES)].copy()


def load_model_bundles() -> tuple[ModelBundle, ModelBundle]:
    trad_metrics = read_json(TRAD_BEST / "metrics.json")
    trad_per_class = read_report(TRAD_BEST / "classification_report.csv")
    trad_per_class.insert(0, "model", "Traditional best")
    trad_per_class.insert(1, "family", "Traditional")

    deep_metrics = pd.read_csv(DEEP_DIR / "tta_ensemble_alpha_0.50_metrics.csv").iloc[0].to_dict()
    deep_metrics = {k: float(v) for k, v in deep_metrics.items()}
    deep_per_class = read_report(DEEP_DIR / "tta_ensemble_alpha_0.50_per_class_metrics.csv")
    deep_per_class.insert(0, "model", "TTA+Ensemble")
    deep_per_class.insert(1, "family", "Deep")

    return (
        ModelBundle("Traditional best", "Traditional", trad_metrics, trad_per_class),
        ModelBundle("TTA+Ensemble", "Deep", deep_metrics, deep_per_class),
    )


def model_summary(trad: ModelBundle, deep: ModelBundle) -> pd.DataFrame:
    rows = []
    for bundle in [trad, deep]:
        row = {"model": bundle.name, "family": bundle.family}
        for col in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            row[col] = float(bundle.metrics[col])
        rows.append(row)
    return pd.DataFrame(rows)


def load_metadata() -> pd.DataFrame:
    parts = []
    for folder in ["akiec", "bcc", "bkl", "df", "mel", "output nv2", "vasc"]:
        p = ROOT / "preprocessing" / folder / "metadata.csv"
        if p.exists():
            part = pd.read_csv(p)
            parts.append(part)
    meta = pd.concat(parts, ignore_index=True)
    meta = meta.drop_duplicates("image_id", keep="first")
    for col in ["hair_ratio", "age"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    if "is_valid" in meta.columns:
        meta["is_valid"] = meta["is_valid"].astype(str).str.lower().map({"true": True, "false": False})
    return meta


def load_preprocessing_metadata_with_paths() -> pd.DataFrame:
    parts = []
    for folder in ["akiec", "bcc", "bkl", "df", "mel", "output nv2", "vasc"]:
        p = ROOT / "preprocessing" / folder / "metadata.csv"
        if not p.exists():
            continue
        part = pd.read_csv(p)
        label = "nv" if folder == "output nv2" else folder
        part["dx"] = label
        origin_folder = "output nv1" if folder == "output nv2" else folder
        origin = ROOT / "preprocessing" / origin_folder / "origin"
        enhanced = ROOT / "preprocessing" / folder / "enhanced"
        part["origin_image_path"] = part["image_id"].map(lambda image_id: origin / f"{image_id}.jpg")
        part["enhanced_image_path"] = part["image_id"].map(lambda image_id: enhanced / f"{image_id}.jpg")
        parts.append(part)
    if not parts:
        raise FileNotFoundError("No preprocessing metadata.csv files found.")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("image_id", keep="first")
    df["image_path"] = df["origin_image_path"].where(df["origin_image_path"].map(Path.exists), df["enhanced_image_path"])
    df = df[df["image_path"].map(Path.exists)].copy()
    return df.reset_index(drop=True)


def load_traditional_config():
    if str(TRAD_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(TRAD_CODE_DIR))

    from ham_pipeline.config import PipelineConfig

    cfg = read_json(TRAD_BEST / "run_config.json")
    if "minority_classes" in cfg and isinstance(cfg["minority_classes"], list):
        cfg["minority_classes"] = tuple(cfg["minority_classes"])
    for key in ["dataset_root", "output_dir"]:
        cfg[key] = Path(cfg[key])
    for key in ["clahe_tile_grid_size", "hsv_bins", "glcm_distances", "glcm_angles", "ann_hidden_layer_sizes"]:
        if key in cfg and isinstance(cfg[key], list):
            cfg[key] = tuple(cfg[key])
    return PipelineConfig(**cfg)


def traditional_validation_split(config, meta_df: pd.DataFrame) -> pd.DataFrame:
    if config.split_mode != "stratified_ratio":
        raise ValueError(f"Unsupported traditional split_mode for reconstruction: {config.split_mode}")
    _, val_df = train_test_split(
        meta_df,
        test_size=config.val_ratio,
        random_state=config.random_state,
        stratify=meta_df["dx"],
    )
    return val_df.reset_index(drop=True)


def traditional_predictions_from_artifact(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import joblib

    if str(TRAD_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(TRAD_CODE_DIR))
    from ham_pipeline.features import extract_features_for_row
    from ham_pipeline import preprocess as trad_preprocess

    def unicode_safe_read_and_resize(image_path: str, image_size: int) -> np.ndarray:
        import cv2

        raw = np.fromfile(image_path, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    trad_preprocess.read_and_resize = unicode_safe_read_and_resize

    config = load_traditional_config()
    meta_all = load_preprocessing_metadata_with_paths()
    val_df = traditional_validation_split(config, meta_all)

    model = joblib.load(TRAD_BEST / "model.joblib")
    rows = []
    for idx, row in val_df.iterrows():
        x = extract_features_for_row(row, config)
        pred_label = str(model.predict(np.expand_dims(x, axis=0))[0])
        out = {
            "image_id": row["image_id"],
            "true_label": row["dx"],
            "pred_label": pred_label,
            "correct": pred_label == row["dx"],
            "image_path": str(row["image_path"]),
            "origin_image_path": str(row["origin_image_path"]),
            "enhanced_image_path": str(row["enhanced_image_path"]),
            "split_source": TRAD_VAL_SOURCE,
        }
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(np.expand_dims(x, axis=0))[0], dtype=float)
            if scores.ndim == 0:
                scores = np.array([float(scores)])
            if len(scores) == len(model.classes_):
                order = np.argsort(-scores)
                out["decision_score"] = float(scores[order[0]])
                out["second_label"] = str(model.classes_[order[1]]) if len(order) > 1 else ""
                out["decision_margin"] = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else np.nan
            else:
                out["decision_score"] = float(np.max(scores))
                out["second_label"] = ""
                out["decision_margin"] = np.nan
        rows.append(out)

    pred = pd.DataFrame(rows)
    pred = pred.merge(meta.drop(columns=[c for c in ["dx"] if c in meta.columns]), on="image_id", how="left", suffixes=("", "_meta"))
    pred["error_type"] = np.where(pred["correct"], "correct", pred["true_label"] + "->" + pred["pred_label"])
    pred["is_risky_error"] = [
        (true_label, pred_label) in RISKY_ERRORS
        for true_label, pred_label in zip(pred["true_label"], pred["pred_label"])
    ]
    pred["hair_bin"] = pd.cut(
        pred["hair_ratio"],
        bins=[-0.001, 0.01, 0.05, 0.10, 0.20, 1.0],
        labels=["0-1%", "1-5%", "5-10%", "10-20%", ">20%"],
        include_lowest=True,
    )
    pred["age_bin"] = pd.cut(
        pred["age"],
        bins=[0, 30, 45, 60, 75, 120],
        labels=["<=30", "31-45", "46-60", "61-75", ">75"],
        include_lowest=True,
    )

    saved_metrics = read_json(TRAD_BEST / "metrics.json")
    y_true = pred["true_label"].to_numpy()
    y_pred = pred["pred_label"].to_numpy()
    recomputed = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    validation = {
        "source": TRAD_VAL_SOURCE,
        "n_predictions": int(len(pred)),
        "model_has_predict_proba": bool(hasattr(model, "predict_proba")),
        "model_has_decision_function": bool(hasattr(model, "decision_function")),
        "saved_metrics": saved_metrics,
        "recomputed_metrics": recomputed,
        "metric_abs_diff": {k: float(abs(recomputed[k] - saved_metrics.get(k, np.nan))) for k in recomputed},
        "exact_metric_match": bool(all(abs(recomputed[k] - saved_metrics.get(k, np.nan)) < 1e-10 for k in recomputed)),
    }
    validation["max_metric_abs_diff"] = float(max(validation["metric_abs_diff"].values()))
    validation["close_metric_match_0p01"] = bool(validation["max_metric_abs_diff"] < 0.01)
    return pred, validation


def enrich_deep_predictions(meta: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(DEEP_DIR / "tta_ensemble_alpha_0.50_predictions.csv")
    probs = pred[PROB_COLS].astype(float).to_numpy()
    order = np.argsort(-probs, axis=1)
    pred["confidence"] = probs[np.arange(len(pred)), order[:, 0]]
    pred["second_label"] = [CLASSES[i] for i in order[:, 1]]
    pred["second_prob"] = probs[np.arange(len(pred)), order[:, 1]]
    pred["margin"] = pred["confidence"] - pred["second_prob"]
    pred["correct"] = pred["correct"].astype(str).str.lower().map({"true": True, "false": False})
    pred["error_type"] = np.where(pred["correct"], "correct", pred["true_label"] + "->" + pred["pred_label"])
    pred["is_risky_error"] = [
        (true_label, pred_label) in RISKY_ERRORS
        for true_label, pred_label in zip(pred["true_label"], pred["pred_label"])
    ]
    pred["high_conf_error"] = (~pred["correct"]) & (pred["confidence"] >= HIGH_CONF_THRESHOLD)
    merged = pred.merge(meta, on="image_id", how="left", suffixes=("", "_meta"))
    merged["age_bin"] = pd.cut(
        merged["age"],
        bins=[0, 30, 45, 60, 75, 120],
        labels=["<=30", "31-45", "46-60", "61-75", ">75"],
        include_lowest=True,
    )
    merged["hair_bin"] = pd.cut(
        merged["hair_ratio"],
        bins=[-0.001, 0.01, 0.05, 0.10, 0.20, 1.0],
        labels=["0-1%", "1-5%", "5-10%", "10-20%", ">20%"],
        include_lowest=True,
    )
    return merged


def traditional_experiment_ranking() -> pd.DataFrame:
    ranking_path = TRAD_DIR / "analysis_results.csv"
    if not ranking_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(ranking_path)
    rename = {
        "文件夹": "experiment",
        "准确率": "accuracy",
        "加权精准率": "weighted_precision",
        "加权召回率": "weighted_recall",
        "加权F1": "weighted_f1",
    }
    df = df.rename(columns=rename)
    for col in ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["weighted_f1", "accuracy"], ascending=False)


def traditional_error_profile(trad_per_class: pd.DataFrame) -> pd.DataFrame:
    df = trad_per_class.copy()
    df["missed"] = df["support"] * (1 - df["recall"])
    df["error_rate"] = 1 - df["recall"]
    df["false_positive_pressure"] = np.where(df["precision"] > 0, 1 - df["precision"], np.nan)
    return df[["class", "precision", "recall", "f1-score", "support", "missed", "error_rate", "false_positive_pressure"]]


def deep_error_flow(deep_pred: pd.DataFrame) -> pd.DataFrame:
    wrong = deep_pred[~deep_pred["correct"]].copy()
    flow = (
        wrong.groupby(["true_label", "pred_label"], dropna=False)
        .agg(
            count=("image_id", "count"),
            avg_confidence=("confidence", "mean"),
            avg_margin=("margin", "mean"),
            avg_hair_ratio=("hair_ratio", "mean"),
            high_conf_errors=("high_conf_error", "sum"),
            risky_errors=("is_risky_error", "sum"),
        )
        .reset_index()
    )
    support = deep_pred.groupby("true_label").size().rename("support").reset_index()
    flow = flow.merge(support, on="true_label", how="left")
    flow["pct_of_true"] = flow["count"] / flow["support"]
    return flow.sort_values(["count", "avg_confidence"], ascending=False)


def traditional_error_flow(trad_pred: pd.DataFrame) -> pd.DataFrame:
    wrong = trad_pred[~trad_pred["correct"]].copy()
    flow = (
        wrong.groupby(["true_label", "pred_label"], dropna=False)
        .agg(
            count=("image_id", "count"),
            avg_decision_margin=("decision_margin", "mean") if "decision_margin" in wrong.columns else ("image_id", "count"),
            avg_hair_ratio=("hair_ratio", "mean"),
            risky_errors=("is_risky_error", "sum"),
        )
        .reset_index()
    )
    support = trad_pred.groupby("true_label").size().rename("support").reset_index()
    flow = flow.merge(support, on="true_label", how="left")
    flow["pct_of_true"] = flow["count"] / flow["support"]
    return flow.sort_values(["count", "pct_of_true"], ascending=False)


def model_error_overlap(trad_pred: pd.DataFrame, deep_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deep_cols = [
        "image_id",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "error_type",
        "hair_ratio",
        "age",
        "sex",
        "localization",
    ]
    deep_cols = [col for col in deep_cols if col in deep_pred.columns]
    merged = trad_pred[["image_id", "true_label", "pred_label", "correct", "error_type"]].merge(
        deep_pred[deep_cols],
        on="image_id",
        how="inner",
        suffixes=("_traditional", "_deep"),
    )
    merged = merged.rename(columns={"true_label_traditional": "true_label"})
    if "true_label_deep" in merged.columns:
        merged = merged.drop(columns=["true_label_deep"])
    conditions = [
        merged["correct_traditional"] & merged["correct_deep"],
        (~merged["correct_traditional"]) & merged["correct_deep"],
        merged["correct_traditional"] & (~merged["correct_deep"]),
        (~merged["correct_traditional"]) & (~merged["correct_deep"]),
    ]
    labels = ["both_correct", "traditional_wrong_deep_correct", "traditional_correct_deep_wrong", "both_wrong"]
    merged["overlap_group"] = np.select(conditions, labels, default="unknown")
    summary = (
        merged.groupby("overlap_group")
        .agg(
            n=("image_id", "count"),
            avg_deep_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    summary["pct_of_common"] = summary["n"] / len(merged) if len(merged) else 0.0
    order = {label: i for i, label in enumerate(labels)}
    summary["order"] = summary["overlap_group"].map(order).fillna(99)
    summary = summary.sort_values("order").drop(columns="order")
    return merged, summary


def overlap_focus_summary(overlap_detail: pd.DataFrame) -> pd.DataFrame:
    focus = overlap_detail[
        overlap_detail["overlap_group"].isin(["traditional_correct_deep_wrong", "both_wrong"])
    ].copy()
    if not len(focus):
        return pd.DataFrame()
    summary = (
        focus.groupby(["overlap_group", "true_label", "pred_label_traditional", "pred_label_deep"], dropna=False)
        .agg(
            count=("image_id", "count"),
            avg_deep_confidence=("confidence", "mean"),
            avg_hair_ratio=("hair_ratio", "mean") if "hair_ratio" in focus.columns else ("image_id", "count"),
        )
        .reset_index()
        .sort_values(["overlap_group", "count", "avg_deep_confidence"], ascending=[True, False, False])
    )
    return summary


def metadata_error_summary(deep_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in ["true_label", "pred_label", "error_type", "hair_bin", "is_valid", "age_bin", "sex", "localization", "dataset", "dx_type"]:
        if factor not in deep_pred.columns:
            continue
        g = (
            deep_pred.groupby(factor, dropna=False)
            .agg(
                n=("image_id", "count"),
                errors=("correct", lambda x: int((~x).sum())),
                error_rate=("correct", lambda x: float((~x).mean())),
                avg_confidence=("confidence", "mean"),
                high_conf_errors=("high_conf_error", "sum"),
                risky_errors=("is_risky_error", "sum"),
            )
            .reset_index()
            .rename(columns={factor: "group"})
        )
        g.insert(0, "factor", factor)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def traditional_metadata_error_summary(trad_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in ["true_label", "pred_label", "error_type", "hair_bin", "age_bin", "sex", "localization", "dataset", "dx_type"]:
        if factor not in trad_pred.columns:
            continue
        g = (
            trad_pred.groupby(factor, dropna=False)
            .agg(
                n=("image_id", "count"),
                errors=("correct", lambda x: int((~x).sum())),
                error_rate=("correct", lambda x: float((~x).mean())),
                risky_errors=("is_risky_error", "sum"),
            )
            .reset_index()
            .rename(columns={factor: "group"})
        )
        g.insert(0, "factor", factor)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def class_quality_summary(deep_pred: pd.DataFrame) -> pd.DataFrame:
    return (
        deep_pred.groupby("true_label")
        .agg(
            n=("image_id", "count"),
            error_rate=("correct", lambda x: float((~x).mean())),
            avg_hair_ratio=("hair_ratio", "mean"),
            invalid_rate=("is_valid", lambda x: float((x == False).mean())),
            high_conf_error_rate=("high_conf_error", "mean"),
            risky_error_count=("is_risky_error", "sum"),
        )
        .reset_index()
        .sort_values("error_rate", ascending=False)
    )


def feature_group_summary() -> pd.DataFrame:
    feature_path = TRAD_BEST / "feature_columns.json"
    if not feature_path.exists():
        return pd.DataFrame()
    features = read_json(feature_path)
    rows = []
    for feature in features:
        group = classify_feature(feature)
        rows.append({"feature": feature, "group": group})
    df = pd.DataFrame(rows)
    summary = df.groupby("group").size().rename("n_features").reset_index()
    summary = summary.sort_values("n_features", ascending=False)
    return summary


def classify_feature(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("lbp_roi"):
        return "LBP_ROI"
    if lowered.startswith("lbp_bimf"):
        return "LBP_BIMF"
    if lowered.startswith("glcm"):
        return "GLCM"
    if lowered.startswith("hsv"):
        return "HSV"
    if lowered.startswith("abcd"):
        return "ABCD"
    return "Other"


def read_svm_support_vectors() -> pd.DataFrame:
    path = INTERP_DIR / "outputs_interpretability_svm" / "svm_support_vectors.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_interpretability_index(deep_pred: pd.DataFrame) -> pd.DataFrame:
    base = INTERP_DIR / "outputs_interpretability_deep"
    attn_files = list((base / "attention_maps").glob("*.png"))
    grad_files = list((base / "gradcam").glob("*.png"))

    def match(files: list[Path], image_id: str) -> str:
        for p in files:
            if image_id in p.name:
                return str(p.relative_to(ROOT))
        return ""

    rows = []
    candidates = deep_pred.copy()
    candidates["sort_correct"] = candidates["correct"].astype(int)
    candidates = candidates.sort_values(["sort_correct", "confidence"], ascending=[True, False])
    for _, row in candidates.iterrows():
        attn = match(attn_files, row["image_id"])
        grad = match(grad_files, row["image_id"])
        if attn or grad:
            rows.append(
                {
                    "image_id": row["image_id"],
                    "true_label": row["true_label"],
                    "pred_label": row["pred_label"],
                    "correct": bool(row["correct"]),
                    "confidence": float(row["confidence"]),
                    "attention_file": attn,
                    "gradcam_file": grad,
                }
            )
    return pd.DataFrame(rows)


def image_path_for(row: pd.Series, enhanced: bool = True) -> Path | None:
    label = row.get("true_label") or row.get("dx")
    folder = "output nv2" if label == "nv" else str(label)
    sub = "enhanced" if enhanced else "origin"
    path = ROOT / "preprocessing" / folder / sub / f"{row['image_id']}.jpg"
    if path.exists():
        return path
    if label == "nv" and not enhanced:
        path = ROOT / "preprocessing" / "output nv1" / "origin" / f"{row['image_id']}.jpg"
        if path.exists():
            return path
    alt = ROOT / "preprocessing" / folder / "origin" / f"{row['image_id']}.jpg"
    if alt.exists():
        return alt
    return None


def visual_reason(row: pd.Series) -> str:
    true_label = row["true_label"]
    pred_label = row["pred_label"]
    hair = float(row["hair_ratio"]) if pd.notna(row.get("hair_ratio")) else 0.0
    loc = str(row.get("localization", "unknown"))
    pair = (true_label, pred_label)
    reasons = {
        ("mel", "nv"): "黑色素瘤被判为痣，通常说明病灶整体色调/形态接近良性色素痣；若边界过渡弱或局部恶性线索不突出，模型会偏向多数类 nv。",
        ("mel", "bkl"): "黑色素瘤被判为 bkl，通常与棕黑色斑片、角化样纹理或颜色不均有关；模型可能把恶性异质性解释成脂溢性角化病纹理。",
        ("nv", "mel"): "普通痣被判为黑色素瘤，常见于颜色较深、边界不够规则或局部色素斑明显的样本；这是偏保守的假阳性。",
        ("nv", "bkl"): "普通痣被判为 bkl，通常是局部斑点/粗糙纹理强于整体痣结构，模型被局部纹理牵引。",
        ("bkl", "nv"): "bkl 被判为 nv，常见于病灶较小、颜色集中、角化纹理不明显的样本；模型更容易把它当作普通色素痣。",
        ("bkl", "mel"): "bkl 被判为 mel，说明病灶存在深浅不均、边界复杂或斑片状结构，触发了模型的恶性类别响应。",
        ("bkl", "akiec"): "bkl 被判为 akiec，通常与红褐色、粗糙或角化样区域有关，两类都可能呈现表面角化/鳞屑样视觉线索。",
        ("akiec", "mel"): "akiec 被判为 mel，可能因为红褐色混合、局部不均匀和边界复杂，被模型解释为恶性黑色素瘤线索。",
        ("akiec", "bcc"): "akiec 被判为 bcc，可能因为面部样本、粉红/红色区域或浅色病灶结构与基底细胞癌相似。",
        ("nv", "vasc"): "nv 被判为 vasc，通常是局部蓝紫/红色或圆形高对比区域触发了血管性病变响应。",
    }
    text = reasons.get(pair, "该错误属于视觉相似类别之间的混淆，可能与颜色、边界、局部纹理或病灶尺度有关。")
    if hair >= 0.05:
        text += " 该图 hair_ratio 较高，毛发或线状背景纹理可能额外干扰边界和局部纹理判断。"
    if loc in {"scalp", "face"}:
        text += f" 病灶位于 {loc}，该部位更容易受到毛发、光照或背景纹理影响。"
    return text


def risk_level(row: pd.Series) -> str:
    pair = (row["true_label"], row["pred_label"])
    if pair == ("mel", "nv"):
        return "high"
    if row.get("high_conf_error", False):
        return "medium-high"
    if pair in RISKY_ERRORS:
        return "medium-high"
    return "medium"


def build_error_case_notes(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in errors.iterrows():
        rows.append(
            {
                "image_id": row["image_id"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "confidence": row["confidence"],
                "second_label": row["second_label"],
                "margin": row["margin"],
                "hair_ratio": row.get("hair_ratio", np.nan),
                "age": row.get("age", np.nan),
                "sex": row.get("sex", ""),
                "localization": row.get("localization", ""),
                "risk_level": risk_level(row),
                "visual_error_note": visual_reason(row),
            }
        )
    return pd.DataFrame(rows)


def overlap_error_reason(row: pd.Series) -> tuple[str, str, str]:
    group = row["overlap_group"]
    true_label = row["true_label"]
    trad_pred = row["pred_label_traditional"]
    deep_pred = row["pred_label_deep"]
    confidence = float(row.get("confidence", 0.0))
    hair = float(row["hair_ratio"]) if pd.notna(row.get("hair_ratio")) else 0.0
    loc = str(row.get("localization", "unknown"))

    pair = (true_label, deep_pred)
    if group == "traditional_correct_deep_wrong":
        if true_label == "nv" and deep_pred in {"mel", "bkl", "bcc", "vasc"}:
            category = "deep_false_alarm_on_nv"
            reason = "传统模型判断正确而深度模型误报，主要是普通痣中局部深色、红紫色或纹理斑块触发了深度模型的高敏感恶性/血管类响应。"
        elif true_label == "mel" and deep_pred in {"nv", "bkl", "akiec"}:
            category = "deep_missed_or_shifted_mel"
            reason = "传统模型保住了 mel，但深度模型把它推向视觉相近类别，说明该图的恶性线索可能较弱或被局部良性纹理主导。"
        elif true_label == "bkl" and deep_pred in {"akiec", "mel", "bcc"}:
            category = "deep_overreacts_to_bkl_texture"
            reason = "bkl 的角化/粗糙纹理被深度模型解释成 akiec、mel 或 bcc 线索，属于纹理过度响应。"
        else:
            category = "deep_specific_failure"
            reason = "传统模型在该样本上保留了正确的全局颜色/纹理特征，但深度模型被局部区域或类别相似性牵引。"
    else:
        if pair == ("mel", "nv"):
            category = "shared_mel_to_nv_high_risk"
            reason = "两个模型都把 mel 判成 nv，是最高风险方向；这通常说明病灶整体外观接近普通痣，边界/颜色不均等恶性线索不够突出。"
        elif true_label == "mel" and deep_pred == "bkl":
            category = "shared_mel_bkl_overlap"
            reason = "两个模型都没有识别出 mel，且深度模型偏向 bkl，说明棕黑斑片、角化样纹理或颜色不均容易被解释为脂溢性角化病。"
        elif true_label == "bkl" and deep_pred in {"nv", "mel", "akiec"}:
            category = "shared_bkl_boundary_case"
            reason = "bkl 同时难倒传统和深度模型，说明其颜色集中、角化线索弱或纹理与 nv/mel/akiec 边界重叠。"
        elif true_label == "akiec":
            category = "shared_akiec_boundary_case"
            reason = "akiec 少数类样本同时失败，常见原因是红褐色、鳞屑/角化样纹理与 mel、bcc、bkl 或 nv 的局部表现相似。"
        elif true_label == "nv":
            category = "shared_nv_false_positive"
            reason = "普通痣被两个模型同时误判，说明该 nv 样本可能具有较强局部色素、边界不规则或红紫色区域，触发了非 nv 响应。"
        else:
            category = "shared_hard_case"
            reason = "两个模型都失败，说明该样本处在类别视觉边界附近，仅靠当前图像特征难以稳定区分。"

    if confidence >= 0.90:
        reason += " 深度模型置信度很高，提示这是需要重点复核的过度自信错误。"
    if hair >= 0.05:
        reason += " hair_ratio 较高，毛发或线状纹理可能干扰边界和局部纹理判断。"
    if loc in {"face", "scalp", "hand", "foot"}:
        reason += f" localization={loc}，该部位更容易出现光照、毛发、角质或背景纹理干扰。"

    priority = "high" if (true_label == "mel" and deep_pred == "nv") else "medium-high" if confidence >= 0.90 or true_label in {"mel", "akiec", "bcc"} else "medium"
    return category, priority, reason


def build_overlap_case_notes(overlap_detail: pd.DataFrame) -> pd.DataFrame:
    focus = overlap_detail[
        overlap_detail["overlap_group"].isin(["traditional_correct_deep_wrong", "both_wrong"])
    ].copy()
    focus = focus.sort_values(["overlap_group", "confidence", "image_id"], ascending=[True, False, True])
    rows = []
    for _, row in focus.iterrows():
        category, priority, reason = overlap_error_reason(row)
        rows.append(
            {
                "image_id": row["image_id"],
                "overlap_group": row["overlap_group"],
                "true_label": row["true_label"],
                "traditional_pred": row["pred_label_traditional"],
                "deep_pred": row["pred_label_deep"],
                "deep_confidence": row["confidence"],
                "hair_ratio": row.get("hair_ratio", np.nan),
                "age": row.get("age", np.nan),
                "sex": row.get("sex", ""),
                "localization": row.get("localization", ""),
                "reason_category": category,
                "review_priority": priority,
                "error_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def export_high_confidence_images(high_conf: pd.DataFrame) -> pd.DataFrame:
    out_dir = IMAGES / "high_confidence_errors"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (_, row) in enumerate(high_conf.iterrows(), start=1):
        image_id = row["image_id"]
        prefix = f"{idx:02d}_{image_id}_{row['true_label']}_to_{row['pred_label']}"
        origin = image_path_for(row, enhanced=False)
        enhanced = image_path_for(row, enhanced=True)
        origin_target = ""
        enhanced_target = ""
        if origin and origin.exists():
            target = out_dir / f"{prefix}_origin{origin.suffix.lower()}"
            shutil.copy2(origin, target)
            origin_target = str(target.relative_to(ROOT))
        if enhanced and enhanced.exists():
            target = out_dir / f"{prefix}_enhanced{enhanced.suffix.lower()}"
            shutil.copy2(enhanced, target)
            enhanced_target = str(target.relative_to(ROOT))
        rows.append(
            {
                "image_id": image_id,
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "confidence": row["confidence"],
                "origin_export": origin_target,
                "enhanced_export": enhanced_target,
                "visual_error_note": visual_reason(row),
            }
        )
    return pd.DataFrame(rows)


def export_overlap_group_images(overlap_detail: pd.DataFrame, group: str) -> pd.DataFrame:
    out_dir = IMAGES / group
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = overlap_detail[overlap_detail["overlap_group"] == group].copy()
    selected = selected.sort_values(["confidence", "image_id"], ascending=[False, True])
    rows = []
    for idx, (_, row) in enumerate(selected.iterrows(), start=1):
        image_id = row["image_id"]
        prefix = (
            f"{idx:02d}_{image_id}_{row['true_label']}"
            f"_trad-{row['pred_label_traditional']}_deep-{row['pred_label_deep']}"
        )
        origin = image_path_for(row, enhanced=False)
        enhanced = image_path_for(row, enhanced=True)
        origin_target = ""
        enhanced_target = ""
        if origin and origin.exists():
            target = out_dir / f"{prefix}_origin{origin.suffix.lower()}"
            shutil.copy2(origin, target)
            origin_target = str(target.relative_to(ROOT))
        if enhanced and enhanced.exists():
            target = out_dir / f"{prefix}_enhanced{enhanced.suffix.lower()}"
            shutil.copy2(enhanced, target)
            enhanced_target = str(target.relative_to(ROOT))
        rows.append(
            {
                "image_id": image_id,
                "overlap_group": group,
                "true_label": row["true_label"],
                "traditional_pred": row["pred_label_traditional"],
                "deep_pred": row["pred_label_deep"],
                "deep_confidence": row["confidence"],
                "origin_export": origin_target,
                "enhanced_export": enhanced_target,
                "visual_error_note": visual_reason(
                    pd.Series(
                        {
                            "image_id": image_id,
                            "true_label": row["true_label"],
                            "pred_label": row["pred_label_deep"],
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def annotate_bars(ax, bars, fmt: str = "{:.3f}", rotation: int = 0, fontsize: int = 8) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=fontsize,
        )


def save_bar(
    path: Path,
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    rotate: int = 0,
    annotate: bool = False,
    annotation_fmt: str = "{:.3f}",
) -> None:
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    bars = ax.bar(df[x].astype(str), df[y].astype(float), color="#386fa4")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    plt.ylim(0, max(1.0, float(df[y].max()) * 1.15 if len(df) else 1.0))
    if annotate:
        annotate_bars(ax, bars, annotation_fmt, rotation=90 if rotate else 0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_grouped_metric_bar(path: Path, summary: pd.DataFrame) -> None:
    metrics = ["accuracy", "balanced_accuracy", "f1_macro"]
    labels = summary["model"].tolist()
    x = np.arange(len(labels))
    width = 0.23
    plt.figure(figsize=(8.5, 5.2))
    ax = plt.gca()
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * width, summary[metric], width, label=metric)
        annotate_bars(ax, bars, "{:.3f}", fontsize=8)
    plt.xticks(x, labels)
    plt.ylim(0, 1.12)
    plt.ylabel("Score")
    plt.title("Best Traditional vs Best Deep Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_per_class_compare(path: Path, trad_pc: pd.DataFrame, deep_pc: pd.DataFrame, metric: str) -> None:
    merged = trad_pc[["class", metric]].merge(deep_pc[["class", metric]], on="class", suffixes=("_traditional", "_deep"))
    x = np.arange(len(merged))
    width = 0.36
    plt.figure(figsize=(9.5, 5.4))
    ax = plt.gca()
    bars_trad = ax.bar(x - width / 2, merged[f"{metric}_traditional"], width, label="Traditional", color="#b45f4d")
    bars_deep = ax.bar(x + width / 2, merged[f"{metric}_deep"], width, label="TTA+Ensemble", color="#386fa4")
    annotate_bars(ax, bars_trad, "{:.2f}", rotation=90, fontsize=7)
    annotate_bars(ax, bars_deep, "{:.2f}", rotation=90, fontsize=7)
    plt.xticks(x, merged["class"])
    plt.ylim(0, 1.14)
    plt.ylabel(metric)
    plt.title(f"Per-class {metric}: Traditional vs Deep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_localization_error_compare(path: Path, trad_meta: pd.DataFrame, deep_meta: pd.DataFrame) -> pd.DataFrame:
    trad_loc = trad_meta[trad_meta["factor"] == "localization"][["group", "n", "errors", "error_rate"]].copy()
    deep_loc = deep_meta[deep_meta["factor"] == "localization"][["group", "n", "errors", "error_rate"]].copy()
    trad_loc = trad_loc.rename(
        columns={
            "n": "n_traditional",
            "errors": "errors_traditional",
            "error_rate": "error_rate_traditional",
        }
    )
    deep_loc = deep_loc.rename(
        columns={
            "n": "n_deep",
            "errors": "errors_deep",
            "error_rate": "error_rate_deep",
        }
    )
    merged = trad_loc.merge(deep_loc, on="group", how="outer")
    merged["group"] = merged["group"].astype(str)
    merged["mean_error_rate"] = merged[["error_rate_traditional", "error_rate_deep"]].mean(axis=1, skipna=True)
    merged = merged.sort_values(["mean_error_rate", "group"], ascending=[False, True]).head(14)

    x = np.arange(len(merged))
    width = 0.36
    plt.figure(figsize=(11, 5.8))
    ax = plt.gca()
    trad_vals = merged["error_rate_traditional"].astype(float).to_numpy()
    deep_vals = merged["error_rate_deep"].astype(float).to_numpy()
    bars_trad = ax.bar(x - width / 2, trad_vals, width, label="Traditional", color="#b45f4d")
    bars_deep = ax.bar(x + width / 2, deep_vals, width, label="TTA+Ensemble", color="#386fa4")
    annotate_bars(ax, bars_trad, "{:.1%}", rotation=90, fontsize=7)
    annotate_bars(ax, bars_deep, "{:.1%}", rotation=90, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(merged["group"], rotation=35, ha="right")
    ymax = np.nanmax([np.nanmax(trad_vals) if len(trad_vals) else 0, np.nanmax(deep_vals) if len(deep_vals) else 0])
    ax.set_ylim(0, min(1.0, max(0.2, float(ymax) * 1.28)))
    ax.set_ylabel("Error rate")
    ax.set_title("Error rate by localization: Traditional vs TTA+Ensemble")
    ax.legend()
    ax.text(
        0.0,
        -0.28,
        "Grouped by each model's available evaluation split; labels show error rate within the localization group.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return merged


def save_deep_confusion(path: Path) -> None:
    cm = pd.read_csv(DEEP_DIR / "tta_ensemble_alpha_0.50_confusion_matrix.csv", index_col=0)
    data = cm.to_numpy(dtype=float)
    row_sum = data.sum(axis=1, keepdims=True)
    norm = np.divide(data, row_sum, out=np.zeros_like(data), where=row_sum != 0)
    plt.figure(figsize=(7, 6))
    plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Row-normalized")
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    plt.yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, f"{int(data[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("TTA+Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_deep_error_gallery(path: Path, deep_pred: pd.DataFrame) -> pd.DataFrame:
    wrong = deep_pred[~deep_pred["correct"]].copy()
    priority = [
        ("mel", "nv"),
        ("nv", "mel"),
        ("bkl", "mel"),
        ("bkl", "nv"),
        ("akiec", "mel"),
        ("bkl", "akiec"),
        ("nv", "bkl"),
        ("mel", "bkl"),
        ("nv", "vasc"),
        ("akiec", "bcc"),
    ]
    chosen = []
    seen = set()
    for true_label, pred_label in priority:
        sub = wrong[(wrong["true_label"] == true_label) & (wrong["pred_label"] == pred_label)]
        if len(sub):
            row = sub.sort_values(["confidence", "margin"], ascending=False).iloc[0]
            chosen.append(row)
            seen.add(row["image_id"])
    for _, row in wrong.sort_values(["confidence", "margin"], ascending=False).iterrows():
        if row["image_id"] not in seen:
            chosen.append(row)
            seen.add(row["image_id"])
        if len(chosen) >= 12:
            break
    case_df = pd.DataFrame(chosen)

    cols, cell_w, cell_h = 4, 330, 365
    rows_n = math.ceil(len(case_df) / cols)
    sheet = Image.new("RGB", (cols * cell_w, 65 + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), "Deep model representative error images", fill=(20, 30, 45), font=font(24))
    small = font(16)
    for idx, (_, row) in enumerate(case_df.iterrows()):
        x = (idx % cols) * cell_w
        y = 65 + (idx // cols) * cell_h
        p = image_path_for(row)
        if p:
            sheet.paste(fit_image(p, (cell_w - 24, 238)), (x + 12, y + 8))
        draw.text((x + 12, y + 258), f"{row['true_label']} -> {row['pred_label']}  conf={row['confidence']:.3f}", fill=(145, 50, 45), font=small)
        draw.text((x + 12, y + 285), f"{row['image_id']}  hair={row['hair_ratio']:.3f}", fill=(20, 30, 45), font=small)
        draw.text((x + 12, y + 312), f"age={row.get('age', '')}, {row.get('sex', '')}, {row.get('localization', '')}"[:38], fill=(20, 30, 45), font=small)
    sheet.save(path, quality=94)
    return case_df


def save_traditional_error_gallery(path: Path, trad_pred: pd.DataFrame) -> pd.DataFrame:
    wrong = trad_pred[~trad_pred["correct"]].copy()
    priority = [
        ("mel", "nv"),
        ("bkl", "nv"),
        ("bcc", "nv"),
        ("akiec", "nv"),
        ("df", "nv"),
        ("vasc", "nv"),
        ("mel", "bkl"),
        ("bkl", "mel"),
        ("nv", "mel"),
        ("nv", "bkl"),
    ]
    chosen = []
    seen = set()
    for true_label, pred_label in priority:
        sub = wrong[(wrong["true_label"] == true_label) & (wrong["pred_label"] == pred_label)]
        if len(sub):
            sort_cols = ["decision_margin"] if "decision_margin" in sub.columns else ["image_id"]
            row = sub.sort_values(sort_cols, ascending=False).iloc[0]
            chosen.append(row)
            seen.add(row["image_id"])
    sort_cols = ["decision_margin"] if "decision_margin" in wrong.columns else ["image_id"]
    for _, row in wrong.sort_values(sort_cols, ascending=False).iterrows():
        if row["image_id"] not in seen:
            chosen.append(row)
            seen.add(row["image_id"])
        if len(chosen) >= 12:
            break
    case_df = pd.DataFrame(chosen)

    cols, cell_w, cell_h = 4, 330, 365
    rows_n = math.ceil(len(case_df) / cols) if len(case_df) else 1
    sheet = Image.new("RGB", (cols * cell_w, 65 + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), "Traditional model representative error images", fill=(20, 30, 45), font=font(24))
    small = font(16)
    for idx, (_, row) in enumerate(case_df.iterrows()):
        x = (idx % cols) * cell_w
        y = 65 + (idx // cols) * cell_h
        p = Path(row.get("origin_image_path", ""))
        if not p.exists():
            p = image_path_for(row, enhanced=False) or image_path_for(row, enhanced=True)
        if p and p.exists():
            sheet.paste(fit_image(p, (cell_w - 24, 238)), (x + 12, y + 8))
        draw.text((x + 12, y + 258), f"{row['true_label']} -> {row['pred_label']}", fill=(145, 50, 45), font=small)
        margin = row.get("decision_margin", np.nan)
        margin_text = f"margin={margin:.3f}" if pd.notna(margin) else "margin=n/a"
        draw.text((x + 12, y + 285), f"{row['image_id']}  {margin_text}", fill=(20, 30, 45), font=small)
        draw.text((x + 12, y + 312), f"hair={row.get('hair_ratio', np.nan):.3f}, {row.get('localization', '')}"[:38], fill=(20, 30, 45), font=small)
    sheet.save(path, quality=94)
    return case_df


def save_overlap_group_gallery(path: Path, overlap_detail: pd.DataFrame, group: str, title: str) -> pd.DataFrame:
    selected = overlap_detail[overlap_detail["overlap_group"] == group].copy()
    selected = selected.sort_values(["confidence", "image_id"], ascending=[False, True]).head(16)
    cols, cell_w, cell_h = 4, 360, 385
    rows_n = math.ceil(len(selected) / cols) if len(selected) else 1
    sheet = Image.new("RGB", (cols * cell_w, 72 + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 20), title, fill=(20, 30, 45), font=font(24))
    small = font(15)
    for idx, (_, row) in enumerate(selected.iterrows()):
        x = (idx % cols) * cell_w
        y = 72 + (idx // cols) * cell_h
        p = image_path_for(row, enhanced=True) or image_path_for(row, enhanced=False)
        if p and p.exists():
            sheet.paste(fit_image(p, (cell_w - 24, 240)), (x + 12, y + 8))
        draw.text((x + 12, y + 258), f"true={row['true_label']}", fill=(20, 30, 45), font=small)
        draw.text((x + 12, y + 282), f"traditional={row['pred_label_traditional']}", fill=(145, 50, 45), font=small)
        draw.text((x + 12, y + 306), f"deep={row['pred_label_deep']} conf={row['confidence']:.3f}", fill=(145, 50, 45), font=small)
        draw.text((x + 12, y + 330), f"{row['image_id']}", fill=(20, 30, 45), font=small)
    sheet.save(path, quality=94)
    return selected


def save_overlap_bar(path: Path, overlap_summary: pd.DataFrame) -> None:
    if not len(overlap_summary):
        return
    labels = overlap_summary["overlap_group"].astype(str)
    values = overlap_summary["n"].astype(int)
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values, color=["#5b8e7d", "#386fa4", "#b45f4d", "#7a5195"][: len(values)])
    plt.ylabel("Number of common images")
    plt.title("Traditional vs Deep Error Overlap")
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(values):
        plt.text(i, int(v) + max(values) * 0.02, str(int(v)), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_error_notes_panel(path: Path, notes: pd.DataFrame) -> None:
    notes = notes.head(8).copy()
    cols, cell_w, cell_h = 2, 700, 300
    rows_n = math.ceil(len(notes) / cols) if len(notes) else 1
    sheet = Image.new("RGB", (cols * cell_w, 70 + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 20), "Representative deep error case notes", fill=(20, 30, 45), font=font(24))
    title_font = font(17)
    body_font = font(14)
    for idx, (_, row) in enumerate(notes.iterrows()):
        x = (idx % cols) * cell_w + 15
        y = 70 + (idx // cols) * cell_h + 10
        draw.rectangle((x, y, x + cell_w - 30, y + cell_h - 20), outline=(210, 216, 224), width=1)
        draw.text((x + 14, y + 12), f"{row['image_id']}  {row['true_label']} -> {row['pred_label']}", fill=(130, 45, 42), font=title_font)
        draw.text((x + 14, y + 38), f"conf={row['confidence']:.3f}  risk={row['risk_level']}", fill=(130, 45, 42), font=title_font)
        note = str(row["visual_error_note"])
        wrapped = wrap_text(note, 44)
        draw.multiline_text((x + 14, y + 72), wrapped, fill=(20, 30, 45), font=body_font, spacing=5)
    sheet.save(path, quality=94)


def save_overlap_reason_panel(path: Path, notes: pd.DataFrame) -> None:
    if not len(notes):
        return
    chosen = pd.concat(
        [
            notes[notes["overlap_group"] == "traditional_correct_deep_wrong"].head(4),
            notes[notes["overlap_group"] == "both_wrong"].head(4),
        ],
        ignore_index=True,
    )
    cols, cell_w, cell_h = 2, 760, 320
    rows_n = math.ceil(len(chosen) / cols) if len(chosen) else 1
    sheet = Image.new("RGB", (cols * cell_w, 72 + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 20), "Focused overlap error reasons", fill=(20, 30, 45), font=font(24))
    title_font = font(16)
    body_font = font(13)
    for idx, (_, row) in enumerate(chosen.iterrows()):
        x = (idx % cols) * cell_w + 15
        y = 72 + (idx // cols) * cell_h + 10
        draw.rectangle((x, y, x + cell_w - 30, y + cell_h - 22), outline=(210, 216, 224), width=1)
        header = (
            f"{row['overlap_group']} | {row['image_id']} | "
            f"true={row['true_label']} trad={row['traditional_pred']} deep={row['deep_pred']}"
        )
        draw.text((x + 14, y + 12), header[:82], fill=(130, 45, 42), font=title_font)
        draw.text(
            (x + 14, y + 38),
            f"deep_conf={row['deep_confidence']:.3f} | priority={row['review_priority']} | {row['reason_category']}",
            fill=(130, 45, 42),
            font=title_font,
        )
        wrapped = wrap_text(str(row["error_reason"]), 52)
        draw.multiline_text((x + 14, y + 72), wrapped, fill=(20, 30, 45), font=body_font, spacing=5)
    sheet.save(path, quality=94)


def wrap_text(text: str, width: int) -> str:
    import re

    def token_width(token: str) -> float:
        return sum(0.55 if ord(char) < 128 else 1.0 for char in token)

    tokens = re.findall(r"[A-Za-z0-9_.>=+\-/]+|\s+|[^\sA-Za-z0-9_.>=+\-/]", str(text))
    lines = []
    current = ""
    current_width = 0.0
    for token in tokens:
        if token.isspace():
            if current:
                current += " "
                current_width += 0.55
            continue
        tw = token_width(token)
        if current and current_width + tw > width:
            lines.append(current.rstrip())
            current = token
            current_width = tw
        else:
            current += token
            current_width += tw
    if current:
        lines.append(current.rstrip())
    return "\n".join(lines)


def draw_ppt_case_card(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    reason: str,
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=(248, 250, 252), outline=(210, 216, 224), width=2)
    draw.text((x + 26, y + 20), title, fill=(0, 63, 136), font=font(26))

    img_w, img_h = 260, 235
    origin = image_path_for(row, enhanced=False)
    enhanced = image_path_for(row, enhanced=True)
    image_y = y + 75
    for label, path, image_x in [
        ("原图", origin, x + 26),
        ("预处理后", enhanced, x + 306),
    ]:
        draw.text((image_x, image_y - 25), label, fill=(51, 65, 85), font=font(16))
        draw.rectangle((image_x, image_y, image_x + img_w, image_y + img_h), fill=(255, 255, 255), outline=(226, 232, 240), width=1)
        if path and path.exists():
            sheet.paste(fit_image(path, (img_w, img_h)), (image_x, image_y))

    confidence = row.get("confidence", np.nan)
    confidence_text = f"{float(confidence):.4f}" if pd.notna(confidence) else "n/a"
    metadata = [
        f"image_id：{row.get('image_id', '')}",
        f"真实标签：{row.get('true_label', '')}",
        f"传统预测：{row.get('pred_label_traditional', row.get('pred_label', ''))}",
        f"深度预测：{row.get('pred_label_deep', row.get('pred_label', ''))}",
        f"置信度：{confidence_text}",
        f"部位：{row.get('localization', 'unknown')}",
    ]
    text_x = x + 26
    text_y = image_y + img_h + 24
    for idx, line in enumerate(metadata):
        draw.text((text_x, text_y + idx * 28), line, fill=(30, 41, 59), font=font(18))

    reason_y = text_y + 184
    draw.rounded_rectangle((x + 26, reason_y, x + w - 26, y + h - 24), radius=12, fill=(255, 247, 237), outline=(253, 186, 116), width=1)
    draw.text((x + 44, reason_y + 14), "错误原因", fill=(154, 52, 18), font=font(18))
    draw.multiline_text(
        (x + 44, reason_y + 44),
        wrap_text(reason, 31),
        fill=(71, 85, 105),
        font=font(16),
        spacing=5,
    )


def save_ppt_case_panel(path: Path, cases: list[tuple[pd.Series, str, str]], title: str, subtitle: str) -> None:
    width, height = 1600, 900
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((42, 30), title, fill=(0, 63, 136), font=font(34))
    draw.text((42, 78), subtitle, fill=(71, 85, 105), font=font(20))

    card_w, card_h = 744, 760
    positions = [(42, 130), (814, 130)]
    for (row, case_title, reason), (x, y) in zip(cases, positions):
        draw_ppt_case_card(sheet, draw, row, x, y, card_w, card_h, case_title, reason)
    sheet.save(path, quality=95)


def require_case(overlap_detail: pd.DataFrame, image_id: str) -> pd.Series:
    matches = overlap_detail[overlap_detail["image_id"] == image_id]
    if not len(matches):
        raise ValueError(f"Required PPT case image_id not found in overlap detail: {image_id}")
    return matches.iloc[0]


def save_ppt_general_error_panel(path: Path, overlap_detail: pd.DataFrame) -> None:
    cases = [
        (
            require_case(overlap_detail, "ISIC_0032046"),
            "传统漏判：mel -> nv",
            "真实为黑色素瘤，传统手工特征模型判为 nv；深度模型判对。该例用于说明 mel/nv 视觉相似时，颜色与纹理手工特征容易把恶性样本推向多数类。",
        ),
        (
            require_case(overlap_detail, "ISIC_0028611"),
            "高置信错误：bkl -> akiec",
            "真实为 bkl，传统模型判为 nv，深度模型以接近 1 的置信度判为 akiec。该例说明 softmax 高置信并不等于医学可靠，需要错误样本复核。",
        ),
    ]
    save_ppt_case_panel(
        path,
        cases,
        "代表性错误样例",
        "每个卡片同时展示原图、预处理图、标签、预测结果和简短错误原因。",
    )


def save_ppt_overlap_panel(path: Path, overlap_detail: pd.DataFrame) -> None:
    cases = [
        (
            require_case(overlap_detail, "ISIC_0034205"),
            "深度回退：传统对、深度错",
            "传统模型判对 mel，深度模型高置信判为 nv，属于深度模型相对传统方法的回退；重点提示深度模型仍会漏掉少数 mel 样本。",
        ),
        (
            require_case(overlap_detail, "ISIC_0030107"),
            "共同困难：两个模型都错",
            "传统模型和深度模型都把 mel 判为 nv，且深度置信度接近 1。该类样本是共同困难样本，应优先进入人工复核和模型校准。",
        ),
    ]
    save_ppt_case_panel(
        path,
        cases,
        "Overlap 重点错误样例",
        "两类样本分别对应深度模型回退和两个模型共同困难。",
    )


def save_interpretability_gallery(path: Path, evidence: pd.DataFrame) -> None:
    wrong = evidence[evidence["correct"] == False].head(4)
    correct = evidence[evidence["correct"] == True].head(4)
    rows = pd.concat([wrong, correct], ignore_index=True)
    cols, cell_w, cell_h = 2, 560, 360
    rows_n = math.ceil(len(rows) / cols) if len(rows) else 1
    sheet = Image.new("RGB", (cols * cell_w, 65 + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), "Deep interpretability cases: Attention / Grad-CAM", fill=(20, 30, 45), font=font(24))
    small = font(15)
    for idx, (_, row) in enumerate(rows.iterrows()):
        x = (idx % cols) * cell_w
        y = 65 + (idx // cols) * cell_h
        rel = row["gradcam_file"] or row["attention_file"]
        p = ROOT / rel if rel else None
        if p and p.exists():
            sheet.paste(fit_image(p, (cell_w - 24, 260)), (x + 12, y + 8))
        draw.text((x + 12, y + 282), f"{row['image_id']}  {row['true_label']} -> {row['pred_label']}", fill=(20, 30, 45), font=small)
        draw.text((x + 12, y + 307), f"correct={row['correct']}  confidence={row['confidence']:.3f}", fill=(20, 30, 45), font=small)
    sheet.save(path, quality=94)


def save_traditional_evidence_panel(path: Path) -> None:
    items = [
        ("SVM support vectors", INTERP_DIR / "outputs_interpretability_svm" / "svm_support_vectors.png"),
        ("Feature ablation", INTERP_DIR / "outputs_interpretability_traditional" / "feature_ablation_full.png"),
        ("ANN feature groups", INTERP_DIR / "outputs_interpretability_traditional" / "ann_weight_by_group.png"),
        ("ANN top weighted features", INTERP_DIR / "outputs_interpretability_traditional" / "ann_top20_features.png"),
        ("Traditional confusion matrix", TRAD_BEST / "confusion_matrix.png"),
    ]
    cols, cell_w, cell_h = 2, 620, 405
    rows_n = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell_w, 70 + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 20), "Traditional model evidence: handcrafted feature failure modes", fill=(20, 30, 45), font=font(24))
    for idx, (label, p) in enumerate(items):
        x = (idx % cols) * cell_w
        y = 70 + (idx // cols) * cell_h
        if p.exists():
            sheet.paste(fit_image(p, (cell_w - 28, cell_h - 58)), (x + 14, y + 10))
        draw.text((x + 16, y + cell_h - 34), label, fill=(20, 30, 45), font=font(18))
    sheet.save(path, quality=94)


def copy_for_presentation() -> None:
    groups = {
        "01_metrics": [
            FIG / "01_overall_metrics.png",
            FIG / "02_per_class_recall_compare.png",
            FIG / "03_per_class_f1_compare.png",
            FIG / "11_error_rate_by_localization.png",
            FIG / "21_error_rate_by_localization_compare.png",
            TABLES / "model_summary_best_traditional_vs_deep.csv",
            TABLES / "traditional_metadata_error_summary.csv",
            TABLES / "localization_error_rate_compare.csv",
        ],
        "02_error_galleries": [
            FIG / "07_deep_error_gallery.jpg",
            FIG / "14_traditional_error_gallery.jpg",
            FIG / "13_deep_error_case_notes.png",
            FIG / "19_ppt_general_error_panel.jpg",
            TABLES / "deep_error_flow_summary.csv",
            TABLES / "traditional_error_flow_summary.csv",
        ],
        "03_overlap_focus": [
            FIG / "15_traditional_deep_error_overlap.png",
            FIG / "16_overlap_traditional_correct_deep_wrong_gallery.jpg",
            FIG / "17_overlap_both_wrong_gallery.jpg",
            FIG / "18_overlap_reason_panel.jpg",
            FIG / "20_ppt_overlap_panel.jpg",
            TABLES / "traditional_deep_error_overlap.csv",
            TABLES / "overlap_focus_summary.csv",
            TABLES / "overlap_case_notes.csv",
            TABLES / "overlap_traditional_correct_deep_wrong.csv",
            TABLES / "overlap_both_wrong.csv",
        ],
        "04_high_confidence": [
            TABLES / "deep_high_confidence_errors.csv",
            TABLES / "high_confidence_error_image_index.csv",
        ],
        "05_reports": [
            REPORTS / "failure_analysis_report.md",
            REPORTS / "manifest.json",
        ],
    }
    for folder, files in groups.items():
        target_dir = PRESENTATION / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            if src.exists():
                shutil.copy2(src, target_dir / src.name)

    image_targets = {
        IMAGES / "traditional_correct_deep_wrong": PRESENTATION / "03_overlap_focus" / "traditional_correct_deep_wrong_images",
        IMAGES / "both_wrong": PRESENTATION / "03_overlap_focus" / "both_wrong_images",
        IMAGES / "high_confidence_errors": PRESENTATION / "04_high_confidence" / "high_confidence_error_images",
    }
    for src_dir, dst_dir in image_targets.items():
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def markdown_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    view = df[columns].copy()
    if n is not None:
        view = view.head(n)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in view.iterrows():
        values = []
        for c in columns:
            v = row[c]
            if pd.isna(v):
                values.append("")
            elif isinstance(v, (float, np.floating)):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    summary: pd.DataFrame,
    trad_pc: pd.DataFrame,
    deep_pc: pd.DataFrame,
    trad_profile: pd.DataFrame,
    trad_rank: pd.DataFrame,
    feature_groups: pd.DataFrame,
    svm_sv: pd.DataFrame,
    deep_flow: pd.DataFrame,
    trad_pred: pd.DataFrame,
    trad_flow: pd.DataFrame,
    trad_case_df: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    overlap_focus: pd.DataFrame,
    overlap_case_notes: pd.DataFrame,
    traditional_validation: dict,
    high_conf: pd.DataFrame,
    all_errors: pd.DataFrame,
    case_df: pd.DataFrame,
    case_notes: pd.DataFrame,
    class_quality: pd.DataFrame,
    metadata_summary: pd.DataFrame,
) -> str:
    trad = summary[summary["family"] == "Traditional"].iloc[0]
    deep = summary[summary["family"] == "Deep"].iloc[0]
    total_errors = int(deep_flow["count"].sum())
    high_conf_errors = int(len(high_conf))
    risky_errors = int(deep_flow["risky_errors"].sum())

    delta = {
        "accuracy": deep["accuracy"] - trad["accuracy"],
        "balanced_accuracy": deep["balanced_accuracy"] - trad["balanced_accuracy"],
        "f1_macro": deep["f1_macro"] - trad["f1_macro"],
    }

    pc_delta = deep_pc[["class", "recall", "f1-score"]].merge(
        trad_pc[["class", "recall", "f1-score"]],
        on="class",
        suffixes=("_deep", "_traditional"),
    )
    pc_delta["recall_gain"] = pc_delta["recall_deep"] - pc_delta["recall_traditional"]
    pc_delta["f1_gain"] = pc_delta["f1-score_deep"] - pc_delta["f1-score_traditional"]
    pc_delta = pc_delta.rename(columns={"f1-score_deep": "f1_deep", "f1-score_traditional": "f1_traditional"})
    pc_delta = pc_delta.sort_values("f1_gain", ascending=False)

    best_trad_name = trad_rank.iloc[0]["experiment"] if len(trad_rank) else "A_single_stage_control"
    if traditional_validation.get("exact_metric_match"):
        trad_metric_match = "完全对齐"
    elif traditional_validation.get("close_metric_match_0p01"):
        trad_metric_match = "基本对齐，但非逐位一致"
    else:
        trad_metric_match = "未完全对齐"
    trad_metric_diff = pd.DataFrame(
        [
            {
                "metric": metric,
                "saved": traditional_validation.get("saved_metrics", {}).get(metric, np.nan),
                "recomputed": traditional_validation.get("recomputed_metrics", {}).get(metric, np.nan),
                "abs_diff": traditional_validation.get("metric_abs_diff", {}).get(metric, np.nan),
            }
            for metric in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
        ]
    )

    lines = [
        "# 模型评估与错误分析报告",
        "",
        "## 1. 本模块做了什么",
        "",
        "本次重新实现的 `failure_analysis.py` 会从团队已有结果中重新读取数据、重新计算指标、重新生成图表和报告。分析只深入比较两个代表模型：传统方法中表现最好的 `Traditional best`，以及深度学习中表现最好的 `TTA+Ensemble`。这样可以把篇幅集中在错误机制上，而不是罗列所有实验。",
        "",
        f"报告重点放在图像分析上：深度模型部分使用逐样本预测表定位错误图像；传统模型部分从最佳 SVM artifact 重新构建验证集逐图预测表，并结合分类报告、特征组、SVM 支持向量和特征消融解释为什么手工特征会失败。高置信错误定义为预测错误，并且模型对预测类别的最大 softmax probability >= {HIGH_CONF_THRESHOLD:.2f}。",
        "",
        "## 2. 总体指标对比",
        "",
        markdown_table(summary, ["model", "family", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]),
        "",
        f"`TTA+Ensemble` 相比传统最佳模型 Accuracy 提升 {delta['accuracy']:.4f}，Balanced Accuracy 提升 {delta['balanced_accuracy']:.4f}，Macro F1 提升 {delta['f1_macro']:.4f}。传统模型 Accuracy 达到 {trad['accuracy']:.4f}，但 Balanced Accuracy 只有 {trad['balanced_accuracy']:.4f}，说明它主要依赖多数类 `nv`；深度模型 Balanced Accuracy 达到 {deep['balanced_accuracy']:.4f}，说明少数类也被明显改善。",
        "",
        "对应图表：`figures/01_overall_metrics.png`、`figures/02_per_class_recall_compare.png`、`figures/03_per_class_f1_compare.png`。",
        "",
        "## 3. 传统模型深入分析",
        "",
        f"传统方法扫描结果中，按加权 F1/Accuracy 排序的最佳实验为 `{best_trad_name}`。本报告对其进行深入分析。传统模型的主要问题不是完全不能识别图像，而是把图像压缩成 LBP、GLCM、HSV、ABCD 等手工特征后，很多空间细节被丢失。",
        "",
        f"本次已使用传统方法代码中的 `run_config.json`、`model.joblib`、`features.py` 重新生成验证集逐图预测表，共 {len(trad_pred)} 张图。当前工作区没有原始 `HAM10000/` 目录，因此验证集由 `preprocessing/*/metadata.csv` 与 origin 图像按同一 `random_state` 和 `stratified_ratio` 重建；重新计算指标与原 artifact 中 `metrics.json` 的关系为：{trad_metric_match}。校验细节见 `reports/traditional_prediction_validation.json`。",
        "",
        "复现差异的主要原因是当前只保留了预处理目录下的 metadata/图像组织，而不是传统模型训练时使用的完整原始 `HAM10000/` 数据目录；此外，当前环境读取 `model.joblib` 时 sklearn 版本与训练版本不同。因此逐图预测表可用于错误分析，但总体指标对比仍以原 artifact 的 `metrics.json` 和 `classification_report.csv` 为准。",
        "",
        markdown_table(trad_metric_diff, ["metric", "saved", "recomputed", "abs_diff"]),
        "",
        "### 3.1 逐类别失败",
        "",
        markdown_table(trad_profile, ["class", "precision", "recall", "f1-score", "support", "missed", "error_rate"]),
        "",
        "传统模型对 `nv` 的 recall 很高，但 `df` 和 `vasc` 的 recall 为 0，`mel` recall 也只有 0.3094。这说明模型更倾向于学习多数类边界，在小类和高风险类别上召回不足。尤其是 `mel`，漏判数量高、临床风险大，是传统模型最需要解释的失败类别之一。",
        "",
        "### 3.2 传统模型逐图错误",
        "",
        markdown_table(trad_flow, ["true_label", "pred_label", "count", "pct_of_true", "avg_decision_margin", "avg_hair_ratio", "risky_errors"], n=12),
        "",
        "传统模型逐图预测表见 `tables/traditional_predictions_reconstructed.csv`，错误样例见 `figures/14_traditional_error_gallery.jpg`。这些样例把传统模型失败落实到具体图像：它经常把 `mel/bkl/bcc/akiec/df/vasc` 等少数类推向 `nv`，这与逐类 recall 结果一致。传统模型的 `decision_margin` 来自 SVM decision function，只能表示分类边界距离，不等同于深度学习 softmax confidence。",
        "",
        markdown_table(trad_case_df, ["image_id", "true_label", "pred_label", "decision_margin", "hair_ratio", "age", "sex", "localization"], n=8),
        "",
        "### 3.3 手工特征覆盖",
        "",
        markdown_table(feature_groups, ["group", "n_features"]),
        "",
        "特征组说明传统模型确实利用了颜色、纹理和形态信息：HSV 对应颜色，LBP/GLCM 对应局部纹理，ABCD 对应皮损形态与边界。但是这些特征多是全局或局部摘要，很难保留病灶内部不同区域之间的空间关系。",
        "",
        "`figures/05_traditional_feature_groups.png` 只说明传统最佳模型输入特征的组成和数量，不代表特征重要性。它显示 LBP 纹理特征数量最多，因此传统模型输入被纹理描述主导；真正的“哪些特征更重要”需要结合 `figures/09_traditional_evidence_panel.jpg` 中 ANN 权重和特征消融图解释。",
        "",
        "### 3.4 可解释性证据",
        "",
        "传统模型证据面板见 `figures/09_traditional_evidence_panel.jpg`。SVM 支持向量、ANN 权重和特征消融共同指向同一个结论：`mel/nv/bkl` 在手工特征空间内边界纠缠，而少数类需要大量边界样本才能被区分。",
        "",
    ]
    if len(svm_sv):
        lines.extend(
            [
                markdown_table(svm_sv, ["class", "n_support_vectors", "sv_distribution_pct", "n_train_samples", "sv_ratio_vs_train_pct"]),
                "",
                "`bkl`、`mel`、`nv` 的支持向量数量占比最高，说明这些类别经常落在分类边界附近。`df` 和 `vasc` 的支持向量/训练样本比例极高，说明小类边界不稳定。换句话说，传统模型不是没有诊断信息，而是手工特征空间无法把这些视觉相近类别稳定分开。",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. 深度模型深入分析",
            "",
            f"`TTA+Ensemble` 在验证集上共有 {total_errors} 个错误，其中 confidence>=0.90 的高置信错误有 {high_conf_errors} 个，高风险错误方向有 {risky_errors} 个。深度模型虽然整体表现好，但错误集中在视觉相似类别，尤其是 `mel/nv/bkl`。",
            "",
            "### 4.1 错误流向",
            "",
            markdown_table(deep_flow, ["true_label", "pred_label", "count", "pct_of_true", "avg_confidence", "avg_margin", "high_conf_errors"], n=12),
            "",
            "最大错误方向是 `mel -> nv`，共 24 例，占真实 `mel` 的 21.62%。这是医学风险最高的方向，因为黑色素瘤被判断为良性色素痣。第二大错误方向是 `nv -> mel`，属于假阳性，反映出模型对深色、不规则或局部色素密集的普通痣较敏感。",
            "",
            "### 4.2 错误图像案例",
            "",
            markdown_table(case_df, ["image_id", "true_label", "pred_label", "confidence", "second_label", "margin", "hair_ratio", "age", "sex", "localization"], n=10),
            "",
            "错误图像画廊见 `figures/07_deep_error_gallery.jpg`。这些案例显示，深度模型的失败通常与以下视觉因素相关：病灶面积小、边界过渡弱、棕黑色调与 `nv/bkl` 接近、局部纹理不典型，以及毛发或背景纹理干扰。为了避免只看图但不知道为什么错，本脚本还生成了逐图解释表 `tables/deep_error_case_notes.csv` 和可视化说明图 `figures/13_deep_error_case_notes.png`。",
            "",
            markdown_table(case_notes, ["image_id", "true_label", "pred_label", "confidence", "risk_level", "visual_error_note"], n=8),
            "",
            "### 4.3 高置信错误",
            "",
            markdown_table(high_conf, ["image_id", "true_label", "pred_label", "confidence", "second_label", "margin", "hair_ratio", "age", "sex", "localization"], n=12),
            "",
            f"高置信错误说明 softmax confidence 不能直接等同于医学可靠性。本报告将 confidence >= {HIGH_CONF_THRESHOLD:.2f} 且预测错误的样本定义为高置信错误。例如 `ISIC_0030107` 是 `mel -> nv` 且 confidence 接近 1，这类样本在真实应用中必须进入人工复核或模型校准流程。所有高置信错误的原图和预处理后图像已导出到 `images/high_confidence_errors/`，索引见 `tables/high_confidence_error_image_index.csv`。",
            "",
            "### 4.4 Attention/Grad-CAM 证据",
            "",
            "深度可解释性画廊见 `figures/08_deep_interpretability_gallery.jpg`。正确样本中，热力图多集中在病灶主体；错误样本中，热力图也常覆盖病灶区域。这说明很多错误不是因为模型完全看错位置，而是因为病灶内部的颜色、纹理、边界特征本身与其他类别高度重叠。",
            "",
            "## 5. 图像质量与元数据交叉分析",
            "",
            markdown_table(class_quality, ["true_label", "n", "error_rate", "avg_hair_ratio", "invalid_rate", "high_conf_error_rate", "risky_error_count"]),
            "",
            "`hair_ratio` 是预处理阶段估计的毛发/线状遮挡比例，`hair_bin` 是把它离散成 `0-1%`、`1-5%`、`5-10%`、`10-20%`、`>20%` 后得到的分组。这里的错误率并没有随 `hair_bin` 单调升高，甚至高毛发组错误率更低。这个现象不能解释为“毛发越多越容易分类”，更合理的解释是：预处理已经有效去除了相当多毛发；高 hair_ratio 组样本量较小；不同 hair_bin 的类别组成不同，例如大量容易识别的 `nv` 可能落在高 hair_ratio 组。因此 hair_bin 只能作为图像质量辅助线索，不能单独作为错误原因。",
            "",
            "`localization` 是 HAM10000 元数据中的病灶身体部位，例如 back、face、lower extremity、scalp、trunk 等。按 localization 看错误率，目的是检查模型是否在某些身体部位更容易失败，或是否存在类别/部位分布偏差。元数据分组图见 `figures/10_error_rate_by_hair_bin.png`、`figures/11_error_rate_by_localization.png`。",
            "",
            "## 6. 传统模型与深度模型错误重叠",
            "",
            markdown_table(overlap_summary, ["overlap_group", "n", "pct_of_common", "avg_deep_confidence"]),
            "",
            "重叠分析只在两个模型共同出现的 image_id 上进行，结果见 `tables/traditional_deep_error_overlap.csv` 和 `figures/15_traditional_deep_error_overlap.png`。如果样本落在 `traditional_wrong_deep_correct`，说明深度模型确实修正了传统手工特征的失败；如果落在 `both_wrong`，说明该图可能属于更困难的视觉相似或图像质量问题，需要结合原图、预处理图和可解释性热力图进一步检查。",
            "",
            "### 6.1 重点关注的两组 overlap 错误",
            "",
            markdown_table(overlap_focus, ["overlap_group", "true_label", "pred_label_traditional", "pred_label_deep", "count", "avg_deep_confidence", "avg_hair_ratio"], n=16),
            "",
            "`traditional_correct_deep_wrong` 是深度模型相对传统模型退步的样本，已导出到 `images/traditional_correct_deep_wrong/`，索引见 `tables/overlap_traditional_correct_deep_wrong_image_index.csv`，画廊见 `figures/16_overlap_traditional_correct_deep_wrong_gallery.jpg`。`both_wrong` 是两个模型都失败的困难样本，已导出到 `images/both_wrong/`，索引见 `tables/overlap_both_wrong_image_index.csv`，画廊见 `figures/17_overlap_both_wrong_gallery.jpg`。",
            "",
            "### 6.2 两组 overlap 的错误原因",
            "",
            markdown_table(overlap_case_notes, ["image_id", "overlap_group", "true_label", "traditional_pred", "deep_pred", "deep_confidence", "reason_category", "review_priority", "error_reason"], n=10),
            "",
            "`traditional_correct_deep_wrong` 主要说明深度模型存在局部过度敏感：一些 `nv` 被深度模型误报为 `mel/vasc/bkl/bcc`，以及少量 `mel/bkl` 被推向相邻类别。`both_wrong` 更像真正的困难样本：最多的是 `mel -> nv` 和 `mel -> bkl`，说明这些图像的恶性线索弱、颜色/纹理更接近良性痣或 bkl；这类样本应优先用于人工复核、后续模型校准或针对性数据增强。逐样本原因表见 `tables/overlap_case_notes.csv`，说明图见 `figures/18_overlap_reason_panel.jpg`。",
            "",
            "## 7. 传统模型与深度模型的对等结论",
            "",
            markdown_table(pc_delta, ["class", "recall_traditional", "recall_deep", "recall_gain", "f1_traditional", "f1_deep", "f1_gain"]),
            "",
            "传统模型的失败重点是表达能力不足：手工特征能描述颜色、纹理和形态，但难以保留细粒度空间关系，因此在 `mel/nv/bkl` 和少数类上边界不稳定。深度模型的失败重点是高相似类别的细粒度误判：它能看到病灶，也能显著改善少数类，但仍会在黑色素瘤、普通痣和脂溢性角化病之间产生高风险、高置信错误。",
            "",
            "最终建议表述：传统方法提供了可解释的医学特征基线，但在 HAM10000 的不平衡和视觉相似条件下不足以稳定区分关键类别；深度学习通过多层视觉特征、TTA 和 ensemble 显著提升 Macro F1 与 Balanced Accuracy，但仍需要结合错误图像案例、可解释性热力图、图像质量控制、模型校准和人工复核来降低 `mel -> nv` 等高风险错误。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs() -> None:
    ensure_dirs()
    trad, deep = load_model_bundles()
    summary = model_summary(trad, deep)
    meta = load_metadata()
    deep_pred = enrich_deep_predictions(meta)
    trad_pred, traditional_validation = traditional_predictions_from_artifact(meta)
    trad_rank = traditional_experiment_ranking()
    trad_profile = traditional_error_profile(trad.per_class)
    deep_flow = deep_error_flow(deep_pred)
    trad_flow = traditional_error_flow(trad_pred)
    overlap_detail, overlap_summary = model_error_overlap(trad_pred, deep_pred)
    overlap_focus = overlap_focus_summary(overlap_detail)
    overlap_case_notes = build_overlap_case_notes(overlap_detail)
    all_errors = deep_pred[~deep_pred["correct"]].sort_values(["confidence", "margin"], ascending=False)
    high_conf = all_errors[all_errors["confidence"] >= HIGH_CONF_THRESHOLD].copy()
    meta_summary = metadata_error_summary(deep_pred)
    trad_meta_summary = traditional_metadata_error_summary(trad_pred)
    class_quality = class_quality_summary(deep_pred)
    feature_groups = feature_group_summary()
    svm_sv = read_svm_support_vectors()
    evidence = read_interpretability_index(deep_pred)
    error_case_notes = build_error_case_notes(all_errors)
    high_conf_image_index = export_high_confidence_images(high_conf)
    overlap_trad_correct_deep_wrong = overlap_detail[overlap_detail["overlap_group"] == "traditional_correct_deep_wrong"].copy()
    overlap_both_wrong = overlap_detail[overlap_detail["overlap_group"] == "both_wrong"].copy()
    overlap_trad_correct_deep_wrong_index = export_overlap_group_images(overlap_detail, "traditional_correct_deep_wrong")
    overlap_both_wrong_index = export_overlap_group_images(overlap_detail, "both_wrong")

    summary.to_csv(TABLES / "model_summary_best_traditional_vs_deep.csv", index=False, encoding="utf-8-sig")
    trad_rank.to_csv(TABLES / "traditional_experiment_ranking.csv", index=False, encoding="utf-8-sig")
    trad.per_class.to_csv(TABLES / "traditional_best_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    trad_profile.to_csv(TABLES / "traditional_error_profile.csv", index=False, encoding="utf-8-sig")
    trad_pred.to_csv(TABLES / "traditional_predictions_reconstructed.csv", index=False, encoding="utf-8-sig")
    trad_flow.to_csv(TABLES / "traditional_error_flow_summary.csv", index=False, encoding="utf-8-sig")
    trad_meta_summary.to_csv(TABLES / "traditional_metadata_error_summary.csv", index=False, encoding="utf-8-sig")
    overlap_detail.to_csv(TABLES / "traditional_deep_error_overlap_detail.csv", index=False, encoding="utf-8-sig")
    overlap_summary.to_csv(TABLES / "traditional_deep_error_overlap.csv", index=False, encoding="utf-8-sig")
    overlap_focus.to_csv(TABLES / "overlap_focus_summary.csv", index=False, encoding="utf-8-sig")
    overlap_case_notes.to_csv(TABLES / "overlap_case_notes.csv", index=False, encoding="utf-8-sig")
    overlap_trad_correct_deep_wrong.to_csv(TABLES / "overlap_traditional_correct_deep_wrong.csv", index=False, encoding="utf-8-sig")
    overlap_both_wrong.to_csv(TABLES / "overlap_both_wrong.csv", index=False, encoding="utf-8-sig")
    overlap_trad_correct_deep_wrong_index.to_csv(TABLES / "overlap_traditional_correct_deep_wrong_image_index.csv", index=False, encoding="utf-8-sig")
    overlap_both_wrong_index.to_csv(TABLES / "overlap_both_wrong_image_index.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "traditional_prediction_validation.json").write_text(
        json.dumps(traditional_validation, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )
    feature_groups.to_csv(TABLES / "traditional_feature_group_summary.csv", index=False, encoding="utf-8-sig")
    svm_sv.to_csv(TABLES / "traditional_svm_support_vectors.csv", index=False, encoding="utf-8-sig")
    deep.per_class.to_csv(TABLES / "deep_best_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    deep_pred.to_csv(TABLES / "deep_predictions_enriched.csv", index=False, encoding="utf-8-sig")
    deep_flow.to_csv(TABLES / "deep_error_flow_summary.csv", index=False, encoding="utf-8-sig")
    all_errors.to_csv(TABLES / "all_deep_errors.csv", index=False, encoding="utf-8-sig")
    high_conf.to_csv(TABLES / "deep_high_confidence_errors.csv", index=False, encoding="utf-8-sig")
    error_case_notes.to_csv(TABLES / "deep_error_case_notes.csv", index=False, encoding="utf-8-sig")
    high_conf_image_index.to_csv(TABLES / "high_confidence_error_image_index.csv", index=False, encoding="utf-8-sig")
    meta_summary.to_csv(TABLES / "metadata_error_summary.csv", index=False, encoding="utf-8-sig")
    class_quality.to_csv(TABLES / "class_quality_error_summary.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(TABLES / "deep_interpretability_evidence_index.csv", index=False, encoding="utf-8-sig")

    save_grouped_metric_bar(FIG / "01_overall_metrics.png", summary)
    save_per_class_compare(FIG / "02_per_class_recall_compare.png", trad.per_class, deep.per_class, "recall")
    save_per_class_compare(FIG / "03_per_class_f1_compare.png", trad.per_class, deep.per_class, "f1-score")
    save_bar(FIG / "04_traditional_error_rate_by_class.png", trad_profile.sort_values("error_rate", ascending=False), "class", "error_rate", "Traditional model error rate by class", "Error rate")
    if len(feature_groups):
        save_bar(FIG / "05_traditional_feature_groups.png", feature_groups, "group", "n_features", "Traditional handcrafted feature groups", "Number of features", rotate=30)
    if len(svm_sv):
        save_bar(FIG / "06_traditional_svm_sv_ratio.png", svm_sv, "class", "sv_ratio_vs_train_pct", "SVM support vector ratio per class", "SV / train samples (%)")
    case_df = save_deep_error_gallery(FIG / "07_deep_error_gallery.jpg", deep_pred)
    save_interpretability_gallery(FIG / "08_deep_interpretability_gallery.jpg", evidence)
    save_traditional_evidence_panel(FIG / "09_traditional_evidence_panel.jpg")
    save_deep_confusion(FIG / "12_deep_confusion_matrix.png")
    save_error_notes_panel(FIG / "13_deep_error_case_notes.png", error_case_notes)
    trad_case_df = save_traditional_error_gallery(FIG / "14_traditional_error_gallery.jpg", trad_pred)
    save_overlap_bar(FIG / "15_traditional_deep_error_overlap.png", overlap_summary)
    save_overlap_group_gallery(
        FIG / "16_overlap_traditional_correct_deep_wrong_gallery.jpg",
        overlap_detail,
        "traditional_correct_deep_wrong",
        "Traditional correct, deep wrong cases",
    )
    save_overlap_group_gallery(
        FIG / "17_overlap_both_wrong_gallery.jpg",
        overlap_detail,
        "both_wrong",
        "Both traditional and deep wrong cases",
    )
    save_overlap_reason_panel(FIG / "18_overlap_reason_panel.jpg", overlap_case_notes)
    save_ppt_general_error_panel(FIG / "19_ppt_general_error_panel.jpg", overlap_detail)
    save_ppt_overlap_panel(FIG / "20_ppt_overlap_panel.jpg", overlap_detail)
    localization_compare = save_localization_error_compare(
        FIG / "21_error_rate_by_localization_compare.png",
        trad_meta_summary,
        meta_summary,
    )
    localization_compare.to_csv(TABLES / "localization_error_rate_compare.csv", index=False, encoding="utf-8-sig")

    for factor, fig_name in [("hair_bin", "10_error_rate_by_hair_bin.png"), ("localization", "11_error_rate_by_localization.png")]:
        sub = meta_summary[meta_summary["factor"] == factor].copy()
        if len(sub):
            sub = sub.sort_values("error_rate", ascending=False).head(14)
            save_bar(
                FIG / fig_name,
                sub,
                "group",
                "error_rate",
                f"Deep model error rate by {factor}",
                "Error rate",
                rotate=35,
                annotate=True,
                annotation_fmt="{:.1%}",
            )

    report = build_report(
        summary,
        trad.per_class,
        deep.per_class,
        trad_profile,
        trad_rank,
        feature_groups,
        svm_sv,
        deep_flow,
        trad_pred,
        trad_flow,
        trad_case_df,
        overlap_summary,
        overlap_focus,
        overlap_case_notes,
        traditional_validation,
        high_conf,
        all_errors,
        case_df,
        error_case_notes,
        class_quality,
        meta_summary,
    )
    (REPORTS / "failure_analysis_report.md").write_text(report, encoding="utf-8-sig")

    manifest = {
        "selected_traditional_model": str(TRAD_BEST.relative_to(ROOT)),
        "selected_deep_model": str(DEEP_DIR.relative_to(ROOT) / "tta_ensemble_alpha_0.50_predictions.csv"),
        "n_deep_predictions": int(len(deep_pred)),
        "n_deep_errors": int((~deep_pred["correct"]).sum()),
        "n_traditional_predictions_reconstructed": int(len(trad_pred)),
        "n_traditional_errors_reconstructed": int((~trad_pred["correct"]).sum()),
        "traditional_prediction_metrics_exact_match": bool(traditional_validation["exact_metric_match"]),
        "traditional_prediction_metrics_close_match_0p01": bool(traditional_validation["close_metric_match_0p01"]),
        "traditional_prediction_max_metric_abs_diff": float(traditional_validation["max_metric_abs_diff"]),
        "traditional_prediction_source": traditional_validation["source"],
        "n_common_images_for_overlap": int(overlap_summary["n"].sum()) if len(overlap_summary) else 0,
        "n_overlap_traditional_correct_deep_wrong": int(len(overlap_trad_correct_deep_wrong)),
        "n_overlap_both_wrong": int(len(overlap_both_wrong)),
        "n_overlap_case_notes": int(len(overlap_case_notes)),
        "high_confidence_threshold": HIGH_CONF_THRESHOLD,
        "n_high_conf_errors": int(len(high_conf)),
        "n_high_conf_origin_images_exported": int((high_conf_image_index["origin_export"] != "").sum()) if len(high_conf_image_index) else 0,
        "n_high_conf_enhanced_images_exported": int((high_conf_image_index["enhanced_export"] != "").sum()) if len(high_conf_image_index) else 0,
        "table_outputs": sorted(p.name for p in TABLES.glob("*.csv")),
        "figure_outputs": sorted(p.name for p in FIG.glob("*")),
        "image_output_dirs": sorted(p.name for p in IMAGES.iterdir() if p.is_dir()),
    }
    (REPORTS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    copy_for_presentation()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    write_outputs()


if __name__ == "__main__":
    main()
