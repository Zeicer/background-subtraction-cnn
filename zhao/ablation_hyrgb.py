import argparse
import csv
import json
import os
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import test_new_HyRGB


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_SIZE = (320, 240)
UNLIMITED_ACTIVE_INDICES = 10 ** 9
PPT_ABLATION_GROUPS = {
    "channel_selection": {
        "title": "Channel Selection Ablation",
        "groups": ("channel",),
    },
    "foreground_threshold": {
        "title": "FOREGROUND_THRESHOLD Ablation",
        "groups": ("foreground_threshold",),
    },
    "varThreshold_auto_sensitivity": {
        "title": "varThreshold and Auto Sensitivity Ablation",
        "groups": ("varThreshold", "auto_varThreshold"),
    },
    "active_region_sampling": {
        "title": "Active Region Sampling Ablation",
        "groups": ("active_region",),
    },
    "inference_mode": {
        "title": "Full LeNet vs ROI CNN + MOG2 Ablation",
        "groups": ("inference_mode",),
    },
}


@dataclass
class Experiment:
    name: str
    group: str
    params: dict = field(default_factory=dict)
    channel_mode: str = "cache"
    inference_mode: str = "fusion"


BASELINE = {
    "update_interval": 1,
    "varThreshold": 16,
    "active_ratio_threshold": 0.35,
    "active_indices_threshold": 200,
    "active_indices_limit": 8000,
    "learningRate": 0.001,
    "batch_size": 2048,
    "auto_varThreshold": True,
    "varThreshold_min": 8,
    "varThreshold_max": 64,
    "varThreshold_step": 2,
    "active_ratio_spike": 0.20,
    "active_ratio_drop": 0.08,
    "active_ratio_smooth": 0.4,
    "foreground_threshold": 0.9,
}


def build_experiments():
    experiments = [
        Experiment("baseline", "baseline"),
        Experiment("channel_fixed_01", "channel", channel_mode="fixed_01"),
        Experiment("channel_cache", "channel", channel_mode="cache"),
        Experiment("channel_recalibrate", "channel", channel_mode="recalibrate"),
    ]

    for threshold in (0.5, 0.7, 0.9, 0.95):
        experiments.append(
            Experiment(
                f"fg_threshold_{str(threshold).replace('.', '_')}",
                "foreground_threshold",
                {"foreground_threshold": threshold},
            )
        )

    for value in (8, 16, 32, 64):
        experiments.append(
            Experiment(
                f"var_fixed_{value}",
                "varThreshold",
                {"varThreshold": value, "auto_varThreshold": False},
            )
        )

    experiments.extend(
        [
            Experiment("auto_var_on", "auto_varThreshold", {"auto_varThreshold": True}),
            Experiment("auto_var_off", "auto_varThreshold", {"auto_varThreshold": False}),
        ]
    )

    for threshold in (100, 200, 240):
        experiments.append(
            Experiment(
                f"active_threshold_{threshold}",
                "active_region",
                {"active_indices_threshold": threshold},
            )
        )

    for limit_name, limit in (
        ("2000", 2000),
        ("8000", 8000),
        ("unlimited", UNLIMITED_ACTIVE_INDICES),
    ):
        experiments.append(
            Experiment(
                f"active_limit_{limit_name}",
                "active_region",
                {"active_indices_limit": limit},
            )
        )

    for lr in (0, 0.001, 0.005, 0.01):
        experiments.append(
            Experiment(
                f"learningRate_{str(lr).replace('.', '_')}",
                "learningRate",
                {"learningRate": lr},
            )
        )

    experiments.extend(
        [
            Experiment("mode_full_lenet", "inference_mode", inference_mode="full_lenet"),
            Experiment(
                "mode_roi_only",
                "inference_mode",
                {"active_ratio_threshold": 2.0},
                inference_mode="roi_only",
            ),
            Experiment("mode_fusion", "inference_mode", inference_mode="fusion"),
        ]
    )

    return experiments


def list_images(folder, max_frames=None):
    files = [
        path for path in sorted(Path(folder).iterdir())
        if path.suffix.lower() in IMAGE_EXTS
    ]

    if max_frames is not None and max_frames > 0:
        files = files[:max_frames]

    return files


def load_pil_rgb(path):
    return Image.open(path).convert("RGB").resize(DEFAULT_SIZE)


def mask_to_np(mask_img):
    if isinstance(mask_img, Image.Image):
        mask = np.array(mask_img.convert("L"))
    else:
        mask = np.array(mask_img)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return cv2.resize(mask, DEFAULT_SIZE, interpolation=cv2.INTER_NEAREST)


def mask_stats(mask, previous_mask=None):
    foreground = mask > 0
    area = int(np.count_nonzero(foreground))
    area_ratio = area / float(mask.size)

    components = 0
    small_components = 0

    if area > 0:
        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8),
            connectivity=8
        )
        components = max(0, num_labels - 1)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] < 20:
                small_components += 1

    flicker_ratio = 0.0
    if previous_mask is not None:
        flicker_ratio = float(np.mean((mask > 0) != (previous_mask > 0)))

    return {
        "area": area,
        "area_ratio": area_ratio,
        "components": components,
        "small_components": small_components,
        "flicker_ratio": flicker_ratio,
    }


def overlay_mask(frame_bgr, mask):
    overlay = frame_bgr.copy()
    active = mask > 0
    if not np.any(active):
        return overlay

    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = 0.45
    blended = cv2.addWeighted(overlay, 1.0 - alpha, red, alpha, 0)
    overlay[active] = blended[active]
    return overlay


def write_sample_panel(path, frame_bgr, mask):
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel = np.hstack([frame_bgr, mask_bgr, overlay_mask(frame_bgr, mask)])
    cv2.imwrite(str(path), panel)


def draw_bar_chart(path, title, labels, values, ylabel="", width=1280, height=720):
    if not labels or not values:
        return

    values = [float(value) for value in values]
    max_value = max(values) if values else 0.0
    if max_value <= 0:
        max_value = 1.0

    margin_l = 110
    margin_r = 40
    margin_t = 82
    margin_b = 170
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_l, 24), title, fill=(20, 20, 20))
    if ylabel:
        draw.text((18, margin_t + chart_h // 2), ylabel, fill=(70, 70, 70))

    axis_color = (70, 70, 70)
    grid_color = (220, 225, 230)
    bar_color = (47, 111, 180)
    draw.line((margin_l, margin_t, margin_l, margin_t + chart_h), fill=axis_color, width=2)
    draw.line((margin_l, margin_t + chart_h, margin_l + chart_w, margin_t + chart_h), fill=axis_color, width=2)

    for step in range(1, 5):
        y = margin_t + chart_h - int(chart_h * step / 4)
        value = max_value * step / 4
        draw.line((margin_l, y, margin_l + chart_w, y), fill=grid_color)
        draw.text((margin_l - 86, y - 8), f"{value:.3g}", fill=(80, 80, 80))

    count = len(labels)
    slot = chart_w / max(count, 1)
    bar_w = max(12, int(slot * 0.62))
    for index, (label, value) in enumerate(zip(labels, values)):
        center_x = margin_l + slot * index + slot / 2
        x0 = int(center_x - bar_w / 2)
        x1 = int(center_x + bar_w / 2)
        bar_h = int(chart_h * (value / max_value))
        y0 = margin_t + chart_h - bar_h
        y1 = margin_t + chart_h
        draw.rectangle((x0, y0, x1, y1), fill=bar_color)
        draw.text((x0, max(margin_t, y0 - 22)), f"{value:.3g}", fill=(20, 20, 20))

        short_label = label[:24]
        text_x = int(center_x - min(90, len(short_label) * 3.5))
        draw.text((text_x, margin_t + chart_h + 16), short_label, fill=(40, 40, 40))

    image.save(path)


def write_experiment_metric_chart(exp_dir, summary):
    labels = [
        "FPS",
        "Avg ms",
        "Mask mean %",
        "Mask std %",
        "Flicker %",
        "Event frames",
        "Small comps",
    ]
    values = [
        summary["fps"],
        summary["avg_infer_ms"],
        summary["mask_area_mean"] * 100.0,
        summary["mask_area_std"] * 100.0,
        summary["flicker_mean"] * 100.0,
        summary["event_like_frames"],
        summary["small_component_mean"],
    ]
    draw_bar_chart(
        Path(exp_dir) / "average_metrics_bar.jpg",
        f"{summary['experiment']} average metrics",
        labels,
        values,
        "value",
    )


def write_summary_charts(output_root, summaries, chart_dir=None, title_prefix=""):
    if not summaries:
        return

    chart_dir = Path(chart_dir) if chart_dir is not None else Path(output_root) / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["experiment"] for row in summaries]
    chart_specs = [
        ("fps", "Average FPS", "fps", 1.0),
        ("avg_infer_ms", "Average Inference Time", "ms/frame", 1.0),
        ("mask_area_mean", "Average Mask Area Ratio", "percent", 100.0),
        ("mask_area_std", "Mask Area Std", "percent", 100.0),
        ("flicker_mean", "Mask Flicker", "percent", 100.0),
        ("event_like_frames", "Event-like Frame Count", "frames", 1.0),
        ("small_component_mean", "Small Component Mean", "count", 1.0),
    ]

    for key, title, ylabel, scale in chart_specs:
        values = [float(row.get(key, 0.0)) * scale for row in summaries]
        draw_bar_chart(chart_dir / f"{key}_bar.jpg", f"{title_prefix}{title}", labels, values, ylabel)


def make_comparison_grid(output_root, sample_records, max_samples=6, filename="comparison_grid.jpg"):
    if not sample_records:
        return

    sample_frame_names = [
        item[1]
        for item in sorted(
            {
                (record.get("frame_number", 0), record["frame_name"])
                for record in sample_records
            }
        )
    ][:max_samples]
    exp_names = []
    for record in sample_records:
        if record["experiment"] not in exp_names:
            exp_names.append(record["experiment"])
    lookup = {
        (record["experiment"], record["frame_name"]): record
        for record in sample_records
    }

    cell_w, cell_h = DEFAULT_SIZE
    header_h = 34
    label_w = 190
    grid = Image.new(
        "RGB",
        (label_w + cell_w * len(sample_frame_names), header_h + cell_h * len(exp_names)),
        "white"
    )
    draw = ImageDraw.Draw(grid)

    for col, frame_name in enumerate(sample_frame_names):
        draw.text((label_w + col * cell_w + 6, 8), frame_name[:30], fill=(0, 0, 0))

    for row, exp_name in enumerate(exp_names):
        y = header_h + row * cell_h
        draw.text((8, y + 8), exp_name[:28], fill=(0, 0, 0))

        for col, frame_name in enumerate(sample_frame_names):
            record = lookup.get((exp_name, frame_name))
            if record is None:
                continue

            mask = Image.open(record["mask_path"]).convert("L").resize(DEFAULT_SIZE)
            mask_rgb = Image.merge("RGB", (mask, mask, mask))
            grid.paste(mask_rgb, (label_w + col * cell_w, y))

    grid.save(Path(output_root) / filename)


def write_ppt_group_outputs(output_root, summaries, sample_records, max_samples=6):
    output_root = Path(output_root)
    group_root = output_root / "ppt_ablation_groups"
    group_root.mkdir(parents=True, exist_ok=True)

    for group_key, spec in PPT_ABLATION_GROUPS.items():
        wanted_groups = set(spec["groups"])
        group_summaries = [row for row in summaries if row.get("group") in wanted_groups]
        group_samples = [record for record in sample_records if record.get("group") in wanted_groups]
        if not group_summaries:
            continue

        group_dir = group_root / group_key
        chart_dir = group_dir / "charts"
        group_dir.mkdir(parents=True, exist_ok=True)

        write_summary_csv(group_dir / "ablation_results.csv", group_summaries)
        write_summary_charts(
            output_root,
            group_summaries,
            chart_dir=chart_dir,
            title_prefix=f"{spec['title']} - ",
        )
        make_comparison_grid(
            group_dir,
            group_samples,
            max_samples,
            filename=f"{group_key}_comparison_grid.jpg",
        )


def reset_scene_cache(base_dir):
    cache_path = Path(base_dir) / "scene_channel_cache.txt"
    if cache_path.exists():
        cache_path.unlink()

    if hasattr(test_new_HyRGB, "global_scene_cache"):
        test_new_HyRGB.global_scene_cache.cache.clear()


def init_system_for_experiment(exp, args, image_paths):
    params = deepcopy(BASELINE)
    params.update(exp.params)

    calibration_frames = None
    scene_id = args.scene_id

    if exp.channel_mode == "fixed_01":
        calibration_frames = None
        scene_id = f"{args.scene_id}_fixed_01"
    elif exp.channel_mode == "cache":
        calibration_frames = [load_pil_rgb(path) for path in image_paths[:args.calibration_frames]]
        scene_id = args.scene_id
    elif exp.channel_mode == "recalibrate":
        reset_scene_cache(args.base_dir)
        calibration_frames = [load_pil_rgb(path) for path in image_paths[:args.calibration_frames]]
        scene_id = f"{args.scene_id}_{exp.name}_{int(time.time() * 1000)}"

    start = time.perf_counter()
    system, bg_img = test_new_HyRGB.init_hyrgb_system(
        base_dir=args.base_dir,
        roi_model_dir=args.roi_model_dir,
        background_path=args.background_path,
        calibration_frames=calibration_frames,
        scene_id=scene_id,
        **params
    )
    init_seconds = time.perf_counter() - start

    if exp.channel_mode == "fixed_01":
        system.best_channels = [0, 1]
        system.is_calibrated = True

    return system, bg_img, params, init_seconds


def infer_mask(exp, system, bg_img, frame_img):
    if exp.inference_mode == "full_lenet":
        mask_img, active_count = system.infer_full_lenet(bg_img, frame_img)
    else:
        mask_img, active_count = system.infer(bg_img, frame_img)

    return mask_to_np(mask_img), int(active_count)


def run_experiment(exp, args, image_paths, sample_records):
    exp_dir = Path(args.output_dir) / exp.name
    mask_dir = exp_dir / "masks"
    sample_dir = exp_dir / "samples"
    sample_start_index = max(0, args.sample_start - 1)
    sample_end_index = sample_start_index + max(0, args.sample_count)

    mask_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    system, bg_img, params, init_seconds = init_system_for_experiment(exp, args, image_paths)

    frame_rows = []
    masks = []
    previous_mask = None
    total_infer_seconds = 0.0

    for frame_idx, image_path in enumerate(image_paths):
        frame_img = load_pil_rgb(image_path)
        frame_bgr = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)

        start = time.perf_counter()
        mask, active_count = infer_mask(exp, system, bg_img, frame_img)
        infer_seconds = time.perf_counter() - start
        total_infer_seconds += infer_seconds

        stats = mask_stats(mask, previous_mask)
        previous_mask = mask
        masks.append(mask)

        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        if sample_start_index <= frame_idx < sample_end_index:
            sample_path = sample_dir / f"{image_path.stem}_panel.jpg"
            write_sample_panel(sample_path, frame_bgr, mask)
            sample_records.append(
                {
                    "experiment": exp.name,
                    "group": exp.group,
                    "frame_number": frame_idx + 1,
                    "frame_name": image_path.name,
                    "mask_path": str(mask_path),
                }
            )

        frame_rows.append(
            {
                "experiment": exp.name,
                "group": exp.group,
                "frame_index": frame_idx,
                "frame_name": image_path.name,
                "infer_ms": infer_seconds * 1000.0,
                "active_count": active_count,
                **stats,
            }
        )

    infer_times = [row["infer_ms"] for row in frame_rows]
    area_ratios = [row["area_ratio"] for row in frame_rows]
    flickers = [row["flicker_ratio"] for row in frame_rows[1:]]
    components = [row["components"] for row in frame_rows]
    small_components = [row["small_components"] for row in frame_rows]
    event_like_frames = sum(1 for ratio in area_ratios if ratio >= args.event_area_ratio)
    empty_fp_frames = sum(
        1
        for row in frame_rows[:args.empty_frames]
        if row["area_ratio"] >= args.false_positive_area_ratio
    )

    summary = {
        "experiment": exp.name,
        "group": exp.group,
        "channel_mode": exp.channel_mode,
        "inference_mode": exp.inference_mode,
        "frames": len(frame_rows),
        "init_seconds": init_seconds,
        "avg_infer_ms": float(np.mean(infer_times)) if infer_times else 0.0,
        "fps": 1000.0 / float(np.mean(infer_times)) if infer_times and np.mean(infer_times) > 0 else 0.0,
        "mask_area_mean": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "mask_area_std": float(np.std(area_ratios)) if area_ratios else 0.0,
        "flicker_mean": float(np.mean(flickers)) if flickers else 0.0,
        "component_mean": float(np.mean(components)) if components else 0.0,
        "small_component_mean": float(np.mean(small_components)) if small_components else 0.0,
        "empty_false_positive_frames": empty_fp_frames,
        "event_like_frames": event_like_frames,
        "best_channels": json.dumps(getattr(system, "best_channels", None), ensure_ascii=False),
        "final_varThreshold": getattr(system, "varThreshold", None),
        "params": json.dumps(params, ensure_ascii=False),
    }

    with open(exp_dir / "frame_metrics.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    write_experiment_metric_chart(exp_dir, summary)

    return summary


def write_summary_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HyRGB ablation experiments and export masks/metrics."
    )
    parser.add_argument("--base-dir", required=True, help="Base folder containing background/cache output.")
    parser.add_argument("--data-dir", required=True, help="Image folder, absolute or relative to base-dir.")
    parser.add_argument("--roi-model-dir", required=True, help="Folder containing best_model_step*.pth.")
    parser.add_argument("--background-path", default=None, help="Background image path. Defaults to base-dir/background.jpg.")
    parser.add_argument("--output-dir", default=None, help="Ablation output folder.")
    parser.add_argument("--scene-id", default="ablation_scene", help="Scene id used for SceneCache experiments.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit test frames. 0 means all.")
    parser.add_argument("--calibration-frames", type=int, default=30, help="Calibration frames for channel experiments.")
    parser.add_argument("--sample-start", type=int, default=164, help="1-based input-frame number used as first sample panel.")
    parser.add_argument("--sample-count", type=int, default=6, help="Number of sample panels per experiment.")
    parser.add_argument("--empty-frames", type=int, default=20, help="First N frames treated as empty-scene weak check.")
    parser.add_argument("--false-positive-area-ratio", type=float, default=0.01)
    parser.add_argument("--event-area-ratio", type=float, default=0.02)
    parser.add_argument("--experiments", nargs="*", default=None, help="Optional experiment names to run.")
    parser.add_argument("--clear-output", action="store_true", help="Delete output folder before running.")
    return parser.parse_args()


def normalize_args(args):
    args.base_dir = os.path.abspath(args.base_dir)

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(args.base_dir, args.data_dir)

    if args.background_path is None:
        args.background_path = os.path.join(args.base_dir, "background.jpg")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, "ablation_hyrgb_output")

    args.data_dir = os.path.abspath(args.data_dir)
    args.roi_model_dir = os.path.abspath(args.roi_model_dir)
    args.background_path = os.path.abspath(args.background_path)
    args.output_dir = os.path.abspath(args.output_dir)
    args.max_frames = None if args.max_frames == 0 else args.max_frames

    return args


def main():
    args = normalize_args(parse_args())
    output_root = Path(args.output_dir)

    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(args.data_dir, args.max_frames)
    if not image_paths:
        raise FileNotFoundError(f"No images found: {args.data_dir}")

    if not os.path.exists(args.background_path):
        raise FileNotFoundError(f"Background image not found: {args.background_path}")

    experiments = build_experiments()
    if args.experiments:
        wanted = set(args.experiments)
        experiments = [exp for exp in experiments if exp.name in wanted]
        missing = wanted - {exp.name for exp in experiments}
        if missing:
            raise ValueError(f"Unknown experiments: {sorted(missing)}")

    metadata = {
        "base_dir": args.base_dir,
        "data_dir": args.data_dir,
        "roi_model_dir": args.roi_model_dir,
        "background_path": args.background_path,
        "output_dir": args.output_dir,
        "frame_count": len(image_paths),
        "sample_start": args.sample_start,
        "sample_count": args.sample_count,
        "sample_frame_numbers": [
            number
            for number in range(args.sample_start, args.sample_start + args.sample_count)
            if 1 <= number <= len(image_paths)
        ],
        "ppt_ablation_groups": {
            key: {
                "title": spec["title"],
                "groups": list(spec["groups"]),
            }
            for key, spec in PPT_ABLATION_GROUPS.items()
        },
        "experiments": [exp.name for exp in experiments],
        "notes": {
            "empty_false_positive_frames": "First --empty-frames frames are treated as a weak empty-scene check.",
            "event_like_frames": "Frames with mask area ratio >= --event-area-ratio.",
            "mode_roi_only": "Uses active_ratio_threshold=2.0 to avoid full-frame fallback.",
            "mode_fusion": "Uses the current HybridBGSSystem default decision path.",
            "sample_start": "--sample-start is 1-based. Default 164 exports frames 164-169 when --sample-count is 6.",
        },
    }
    with open(output_root / "ablation_config.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    summaries = []
    sample_records = []

    for index, exp in enumerate(experiments, 1):
        print(f"[{index}/{len(experiments)}] Running {exp.name}")
        summary = run_experiment(exp, args, image_paths, sample_records)
        summaries.append(summary)
        write_summary_csv(output_root / "ablation_results.csv", summaries)
        write_summary_charts(output_root, summaries)
        write_ppt_group_outputs(output_root, summaries, sample_records, args.sample_count)

    make_comparison_grid(output_root, sample_records, args.sample_count)
    write_summary_csv(output_root / "ablation_results.csv", summaries)
    write_summary_charts(output_root, summaries)
    write_ppt_group_outputs(output_root, summaries, sample_records, args.sample_count)
    print(f"Ablation finished: {output_root}")


if __name__ == "__main__":
    main()
