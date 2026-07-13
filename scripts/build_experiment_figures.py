# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "reports" / "figures" / "experiment_results"

DETECTOR_JSON = ROOT / "outputs" / "metrics" / "football" / "yolo26m" / "yolo26m_val.json"
TRACKING_COMPARE_JSON = (
    ROOT
    / "outputs"
    / "metrics"
    / "experiments"
    / "yolo26m_botsort_all"
    / "sort_vs_deepsort_overall.json"
)
TRACKING_STABLE_JSON = (
    ROOT
    / "outputs"
    / "metrics"
    / "experiments"
    / "yolo26m_botsort_identity_stable_all"
    / "sort_vs_deepsort_overall.json"
)
TRACKER_GRID_JSON = (
    ROOT
    / "outputs"
    / "metrics"
    / "experiments"
    / "tracker_grid"
    / "botsort_reid_recall_identity_overall.json"
)
IDSW_SUMMARY_JSON = ROOT / "outputs" / "reports" / "idsw_taxonomy" / "idsw_taxonomy_summary.json"
IDSW_PER_SEQUENCE_CSV = (
    ROOT / "outputs" / "reports" / "idsw_taxonomy" / "idsw_taxonomy_per_sequence.csv"
)
SEMANTIC_EVAL_JSON = (
    ROOT / "outputs" / "semantic_video_experiments" / "1" / "evaluation" / "evaluation_results.json"
)
CONFUSION_DIR = ROOT / "outputs" / "semantic_video_experiments" / "1" / "evaluation"

TRACKER_LABELS = {
    "sort": "SORT",
    "deepsort": "DeepSORT",
    "bytetrack": "ByteTrack",
    "botsort_reid": "BoT-SORT ReID",
    "botsort_no_reid": "BoT-SORT no ReID",
    "botsort_reid_balanced": "BoT-SORT ReID balanced",
    "botsort_reid_identity_stable": "BoT-SORT ReID stable",
}

PIPELINE_LABELS = {
    "A": "A: Qwen3-VL 4B",
    "B": "B: LocateAnything 3B",
    "C": "C: Locate + Qwen",
}

COLORS = {
    "green": "#5B7537",
    "dark_green": "#385823",
    "blue": "#3C7DA6",
    "red": "#C00000",
    "gold": "#C9A227",
    "gray": "#6F6F6F",
    "light": "#E9F0E2",
    "dark": "#404040",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.edgecolor": "#BBBBBB",
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
        }
    )


def save_figure(
    fig: plt.Figure, path: Path, generated: list[dict], title: str, source: Path | str
) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    generated.append({"title": title, "path": str(path.relative_to(ROOT)), "source": str(source)})


def annotate_bars(ax, fmt: str = "{:.1f}", rotation: int = 0) -> None:
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.annotate(
                fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=rotation,
            )


def label_tracker(name: str) -> str:
    return TRACKER_LABELS.get(name, name.replace("_", " "))


def tracking_rows() -> pd.DataFrame:
    rows = []
    if TRACKING_COMPARE_JSON.exists():
        rows.extend(read_json(TRACKING_COMPARE_JSON))
    if TRACKING_STABLE_JSON.exists():
        stable_rows = read_json(TRACKING_STABLE_JSON)
        for row in stable_rows:
            row = dict(row)
            row["tracker"] = "botsort_reid_identity_stable"
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["tracker_label"] = df["tracker"].map(label_tracker)
    return df


def figure_detector(out_dir: Path, generated: list[dict]) -> None:
    if not DETECTOR_JSON.exists():
        return
    data = read_json(DETECTOR_JSON)
    metrics = {
        "Precision": data.get("precision"),
        "Recall": data.get("recall"),
        "mAP50": data.get("map50"),
        "mAP50-95": data.get("map50_95"),
        "mAP75": data.get("map75"),
    }
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(
        metrics.keys(),
        metrics.values(),
        color=[COLORS["green"], COLORS["green"], COLORS["blue"], COLORS["blue"], COLORS["gold"]],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("YOLO26m detector validation on SportsMOT football")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    note = (
        f"checkpoint: {Path(data.get('checkpoint', '')).name} | "
        f"imgsz={data.get('image_size')} | batch={data.get('batch')}"
    )
    ax.text(0.5, -0.18, note, transform=ax.transAxes, ha="center", fontsize=9, color=COLORS["gray"])
    save_figure(
        fig,
        out_dir / "01_detector_validation_metrics.png",
        generated,
        "Detector validation metrics",
        DETECTOR_JSON,
    )


def figure_tracking_comparison(out_dir: Path, generated: list[dict]) -> None:
    df = tracking_rows()
    if df.empty:
        return
    metrics = ["HOTA", "DetA", "AssA", "MOTA", "IDF1"]
    labels = df["tracker_label"].tolist()
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    palette = [COLORS["green"], COLORS["blue"], COLORS["gold"], COLORS["red"], COLORS["gray"]]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, df[metric].astype(float), width, label=metric, color=palette[i])
    ax.set_title("Tracking accuracy comparison on SportsMOT football")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend(ncols=len(metrics), loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        out_dir / "02_tracking_accuracy_comparison.png",
        generated,
        "Tracking accuracy comparison",
        f"{TRACKING_COMPARE_JSON}; {TRACKING_STABLE_JSON}",
    )

    error_metrics = [
        m for m in ["IDSW", "Frag", "FP", "FN", "unique_predicted_ids"] if m in df.columns
    ]
    if error_metrics:
        fig, axes = plt.subplots(
            1, len(error_metrics), figsize=(4.2 * len(error_metrics), 5.2), sharex=False
        )
        if len(error_metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, error_metrics, strict=True):
            vals = df[metric].astype(float).to_numpy()
            ax.barh(
                labels,
                vals,
                color=COLORS["green"] if metric in {"IDSW", "Frag"} else COLORS["blue"],
            )
            ax.set_title(metric)
            ax.grid(axis="x", alpha=0.25)
            for j, v in enumerate(vals):
                ax.text(v + max(vals) * 0.015, j, f"{v:.0f}", va="center", fontsize=8)
        fig.suptitle("Tracking error/count comparison", y=1.02, fontsize=16)
        save_figure(
            fig,
            out_dir / "03_tracking_error_counts.png",
            generated,
            "Tracking error counts",
            f"{TRACKING_COMPARE_JSON}; {TRACKING_STABLE_JSON}",
        )

    if "tracker_fps" in df.columns:
        fig, ax = plt.subplots(figsize=(10.5, 5.4))
        vals = df["tracker_fps"].astype(float)
        ax.bar(df["tracker_label"], vals, color=COLORS["dark_green"])
        ax.set_title("Tracker speed")
        ax.set_ylabel("FPS")
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df["tracker_label"], rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        annotate_bars(ax, "{:.1f}")
        save_figure(
            fig,
            out_dir / "04_tracker_fps.png",
            generated,
            "Tracker FPS",
            f"{TRACKING_COMPARE_JSON}; {TRACKING_STABLE_JSON}",
        )


def idsw_dataframe() -> pd.DataFrame:
    if not IDSW_SUMMARY_JSON.exists():
        return pd.DataFrame()
    data = read_json(IDSW_SUMMARY_JSON)
    rows = [row for row in data.get("summaries", []) if row.get("sequence") == "__overall__"]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["tracker_label"] = df["tracker"].map(label_tracker)
    return df


def figure_idsw(out_dir: Path, generated: list[dict]) -> None:
    df = idsw_dataframe()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    vals = df["total_id_switches_recomputed"].astype(float)
    ax.bar(df["tracker_label"], vals, color=COLORS["red"])
    ax.set_title("Total ID switches by tracker")
    ax.set_ylabel("IDSW count")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["tracker_label"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, "{:.0f}")
    save_figure(
        fig,
        out_dir / "05_idsw_total_by_tracker.png",
        generated,
        "IDSW total by tracker",
        IDSW_SUMMARY_JSON,
    )

    categories = [
        ("fragmentation", "Fragmentation", COLORS["green"]),
        ("identity_swap", "Identity swap", COLORS["blue"]),
        ("re_identification_failure", "ReID failure", COLORS["gold"]),
        ("association_error", "Association error", COLORS["red"]),
        ("appearance_confusion", "Appearance confusion", COLORS["gray"]),
    ]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    left = np.zeros(len(df))
    for key, label, color in categories:
        vals = df[f"{key}_percent"].astype(float).to_numpy()
        ax.barh(y, vals, left=left, label=label, color=color)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(df["tracker_label"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of IDSW")
    ax.set_title("IDSW taxonomy percentage")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="x", alpha=0.2)
    save_figure(
        fig,
        out_dir / "06_idsw_taxonomy_percent_stacked.png",
        generated,
        "IDSW taxonomy percentage",
        IDSW_SUMMARY_JSON,
    )

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    left = np.zeros(len(df))
    for key, label, color in categories:
        vals = df[f"{key}_count"].astype(float).to_numpy()
        ax.barh(y, vals, left=left, label=label, color=color)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(df["tracker_label"])
    ax.set_xlabel("Count")
    ax.set_title("IDSW taxonomy counts")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="x", alpha=0.2)
    save_figure(
        fig,
        out_dir / "07_idsw_taxonomy_count_stacked.png",
        generated,
        "IDSW taxonomy counts",
        IDSW_SUMMARY_JSON,
    )

    if IDSW_PER_SEQUENCE_CSV.exists():
        per_seq = pd.read_csv(IDSW_PER_SEQUENCE_CSV)
        if not per_seq.empty:
            pivot = per_seq.pivot_table(
                index="sequence",
                columns="tracker",
                values="total_id_switches_recomputed",
                aggfunc="sum",
                fill_value=0,
            )
            top = (
                pivot.assign(total=pivot.sum(axis=1)).sort_values("total", ascending=False).head(10)
            )
            fig, ax = plt.subplots(figsize=(13, 6.0))
            top.drop(columns=["total"]).plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set_title("Top 10 sequences by recomputed IDSW")
            ax.set_ylabel("IDSW count")
            ax.set_xlabel("Sequence")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="Tracker", ncols=3, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            save_figure(
                fig,
                out_dir / "08_idsw_top_sequences.png",
                generated,
                "Top IDSW sequences",
                IDSW_PER_SEQUENCE_CSV,
            )


def figure_tracker_grid(out_dir: Path, generated: list[dict]) -> None:
    if not TRACKER_GRID_JSON.exists():
        return
    rows = read_json(TRACKER_GRID_JSON)
    if not rows:
        return
    df = pd.DataFrame(rows)
    required = {"HOTA", "IDF1", "IDSW", "tracker_fps", "variant"}
    if not required.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    sizes = np.clip(df["tracker_fps"].astype(float), 4, None) * 14
    scatter = ax.scatter(
        df["HOTA"].astype(float),
        df["IDF1"].astype(float),
        c=df["IDSW"].astype(float),
        s=sizes,
        cmap="viridis_r",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title("BoT-SORT ReID grid: HOTA vs IDF1")
    ax.set_xlabel("HOTA")
    ax.set_ylabel("IDF1")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("IDSW")
    best = df.sort_values(["HOTA", "IDF1"], ascending=False).head(3)
    for _, row in best.iterrows():
        ax.annotate(
            str(row["variant"]).replace("variant_", "v"),
            (row["HOTA"], row["IDF1"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.text(
        0.02,
        -0.16,
        "Bubble size ~ tracker FPS. Lower color value means fewer ID switches.",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["gray"],
    )
    save_figure(
        fig,
        out_dir / "09_tracker_grid_hota_idf1_tradeoff.png",
        generated,
        "Tracker grid HOTA-IDF1 tradeoff",
        TRACKER_GRID_JSON,
    )

    top = df.sort_values("HOTA", ascending=False).head(10).copy()
    fig, ax1 = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(top))
    ax1.bar(x, top["HOTA"].astype(float), color=COLORS["green"], label="HOTA")
    ax1.set_ylabel("HOTA")
    ax1.set_ylim(max(0, top["HOTA"].min() - 4), min(100, top["HOTA"].max() + 4))
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        top["variant"].str.replace("variant_", "v", regex=False), rotation=25, ha="right"
    )
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, top["IDSW"].astype(float), color=COLORS["red"], marker="o", label="IDSW")
    ax2.set_ylabel("IDSW")
    ax1.set_title("Top tracker-grid variants by HOTA")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    save_figure(
        fig,
        out_dir / "10_tracker_grid_top_variants.png",
        generated,
        "Top tracker-grid variants",
        TRACKER_GRID_JSON,
    )


def semantic_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SEMANTIC_EVAL_JSON.exists():
        return pd.DataFrame(), pd.DataFrame()
    data = read_json(SEMANTIC_EVAL_JSON)
    model_rows = []
    render_rows = []
    for pid, result in data.get("pipeline_results", {}).items():
        model = result.get("model_claim_metrics", {})
        render = result.get("render_output_metrics", {})
        model_rows.append(
            {
                "pipeline": pid,
                "pipeline_label": PIPELINE_LABELS.get(pid, pid),
                "coverage": model.get("coverage_rate", 0) * 100,
                "team_acc": model.get("team_accuracy_on_all_annotated", 0) * 100,
                "role_acc": model.get("role_accuracy_on_all_annotated", 0) * 100,
                "joint_acc": model.get("joint_accuracy_on_all_annotated", 0) * 100,
                "covered_tracks": model.get("covered_tracks", 0),
                "annotated_tracks": model.get(
                    "annotated_tracks", data.get("annotated_track_count", 0)
                ),
            }
        )
        render_rows.append(
            {
                "pipeline": pid,
                "pipeline_label": PIPELINE_LABELS.get(pid, pid),
                "coverage": render.get("coverage_rate", 0) * 100,
                "team_acc": render.get("team_accuracy_on_all_annotated", 0) * 100,
                "role_acc": render.get("role_accuracy_on_all_annotated", 0) * 100,
                "joint_acc": render.get("joint_accuracy_on_all_annotated", 0) * 100,
                "covered_tracks": render.get("covered_tracks", 0),
                "annotated_tracks": render.get(
                    "annotated_tracks", data.get("annotated_track_count", 0)
                ),
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(render_rows)


def figure_semantic(out_dir: Path, generated: list[dict]) -> None:
    model_df, render_df = semantic_rows()
    metrics = [
        ("coverage", "Coverage"),
        ("team_acc", "Team Acc"),
        ("role_acc", "Role Acc"),
        ("joint_acc", "Team+Role Acc"),
    ]
    for df, name, filename in [
        (model_df, "Semantic model-claim metrics", "11_semantic_model_claim_metrics.png"),
        (render_df, "Semantic render-output audit", "12_semantic_render_output_audit.png"),
    ]:
        if df.empty:
            continue
        x = np.arange(len(df))
        width = 0.18
        fig, ax = plt.subplots(figsize=(11, 5.7))
        for i, (col, label) in enumerate(metrics):
            ax.bar(x + (i - 1.5) * width, df[col].astype(float), width, label=label)
        ax.set_title(name)
        ax.set_ylabel("Percent")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(df["pipeline_label"], rotation=12, ha="right")
        ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, out_dir / filename, generated, name, SEMANTIC_EVAL_JSON)


def metadata_rows() -> pd.DataFrame:
    rows = []
    for path in Path("F:/videos").glob("*pipeline*.metadata.json"):
        try:
            data = read_json(path)
        except Exception:
            continue
        name = path.name
        pipeline = (
            "A"
            if "_pipeline_A_" in name
            else "B"
            if "_pipeline_B_" in name
            else "C"
            if "_pipeline_C_" in name
            else "?"
        )
        video = Path(str(data.get("source_video", ""))).stem or name.split("_pipeline_")[0]
        rows.append(
            {
                "path": path,
                "video": video,
                "pipeline": pipeline,
                "pipeline_label": PIPELINE_LABELS.get(pipeline, pipeline),
                "frame_count": data.get("frame_count", 0),
                "rendered_frames": data.get("rendered_frames", 0),
                "total_track_boxes": data.get("total_track_boxes", 0),
                "drawn_boxes": data.get("drawn_boxes", 0),
                "skipped_unlabeled_boxes": data.get("skipped_unlabeled_boxes", 0),
                "unlabeled_track_count": data.get("unlabeled_track_count", 0),
                "team_label_counts": data.get("team_label_counts", {}),
                "role_label_counts": data.get("role_label_counts", {}),
            }
        )
    return pd.DataFrame(rows)


def figure_metadata(out_dir: Path, generated: list[dict]) -> None:
    df = metadata_rows()
    if df.empty:
        return
    for count_col, title, filename in [
        (
            "team_label_counts",
            "Rendered team-label distribution",
            "13_render_team_label_counts.png",
        ),
        (
            "role_label_counts",
            "Rendered role-label distribution",
            "14_render_role_label_counts.png",
        ),
    ]:
        labels = sorted({label for counts in df[count_col] for label in counts.keys()})
        if not labels:
            continue
        plot_df = pd.DataFrame(
            [
                {label: row[count_col].get(label, 0) for label in labels}
                | {"run": f"{row['video']}-{row['pipeline']}"}
                for _, row in df.iterrows()
            ]
        ).set_index("run")
        fig, ax = plt.subplots(figsize=(13, 6.0))
        plot_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(title)
        ax.set_ylabel("Track count")
        ax.set_xlabel("Video / pipeline")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Label", ncols=3, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, out_dir / filename, generated, title, "F:/videos/*pipeline*.metadata.json")

    fig, ax = plt.subplots(figsize=(12, 5.8))
    labels = [f"{row.video}-{row.pipeline}" for row in df.itertuples()]
    x = np.arange(len(df))
    ax.bar(
        x,
        df["total_track_boxes"].astype(float),
        color=COLORS["light"],
        edgecolor=COLORS["green"],
        label="Total boxes",
    )
    ax.bar(
        x, df["drawn_boxes"].astype(float), color=COLORS["green"], alpha=0.7, label="Drawn boxes"
    )
    ax.plot(
        x,
        df["skipped_unlabeled_boxes"].astype(float),
        color=COLORS["red"],
        marker="o",
        label="Skipped unlabeled boxes",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Render completeness by video and pipeline")
    ax.set_ylabel("Box count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        out_dir / "15_render_completeness.png",
        generated,
        "Render completeness",
        "F:/videos/*pipeline*.metadata.json",
    )


def figure_confusions(out_dir: Path, generated: list[dict]) -> None:
    if not CONFUSION_DIR.exists():
        return
    confusion_out = out_dir / "confusion_matrices"
    ensure_dir(confusion_out)
    for csv_path in sorted(CONFUSION_DIR.glob("confusion_*.csv")):
        df = pd.read_csv(csv_path, index_col=0)
        if df.empty:
            continue
        matrix = df.to_numpy(dtype=float)
        fig_w = max(5.5, 0.7 * len(df.columns) + 3)
        fig_h = max(4.8, 0.55 * len(df.index) + 2.4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = max(1, float(matrix.max()))
        im = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=35, ha="right")
        ax.set_yticklabels(df.index)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("GT label")
        title = csv_path.stem.replace("confusion_", "").replace("_", " ")
        ax.set_title(f"Confusion matrix: {title}")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > vmax * 0.55 else COLORS["dark"]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_figure(
            fig,
            confusion_out / f"{csv_path.stem}.png",
            generated,
            f"Confusion matrix {title}",
            csv_path,
        )


def make_contact_sheet(out_dir: Path, generated: list[dict]) -> None:
    image_paths = [
        ROOT / item["path"]
        for item in generated
        if item["path"].endswith(".png") and "confusion_matrices" not in item["path"]
    ]
    image_paths = [p for p in image_paths if p.exists()]
    if not image_paths:
        return
    thumb_w, thumb_h = 360, 220
    cols = 3
    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w - 16, thumb_h - 12))
        col = idx % cols
        row = idx // cols
        x = col * thumb_w + (thumb_w - img.width) // 2
        y = row * (thumb_h + 34) + 8
        sheet.paste(img, (x, y))
        label = path.name
        if len(label) > 42:
            label = label[:39] + "..."
        draw.text(
            (col * thumb_w + 10, row * (thumb_h + 34) + thumb_h + 8),
            label,
            fill=COLORS["dark"],
            font=font,
        )
    sheet_path = out_dir / "00_all_main_figures_contact_sheet.png"
    sheet.save(sheet_path)
    generated.insert(
        0,
        {
            "title": "All main figures contact sheet",
            "path": str(sheet_path.relative_to(ROOT)),
            "source": "generated figures",
        },
    )


def write_index(out_dir: Path, generated: list[dict]) -> None:
    lines = ["# Experiment Figure Pack", ""]
    lines.append(
        "Generated Matplotlib figures for the current football tracking / "
        "semantic labeling experiments."
    )
    lines.append("")
    for item in generated:
        rel = item["path"].replace("\\", "/")
        source = str(item["source"]).replace("\\", "/")
        lines.append(f"- **{item['title']}**: `{rel}`")
        lines.append(f"  Source: `{source}`")
    (out_dir / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "figure_index.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_figures(output_dir: Path) -> list[dict]:
    setup_style()
    ensure_dir(output_dir)
    generated: list[dict] = []
    figure_detector(output_dir, generated)
    figure_tracking_comparison(output_dir, generated)
    figure_idsw(output_dir, generated)
    figure_tracker_grid(output_dir, generated)
    figure_semantic(output_dir, generated)
    figure_metadata(output_dir, generated)
    figure_confusions(output_dir, generated)
    make_contact_sheet(output_dir, generated)
    write_index(output_dir, generated)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Matplotlib figures from experiment metrics."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PNG figures and indexes will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    generated = build_figures(output_dir)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "figure_count": len(generated)},
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
