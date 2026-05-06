#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASETS = ["MH_01_easy", "V2_02_medium"]
TEMPLATE_BY_MODE = {
    "filter": "euroc_filter_stable.yaml",
    "optimizer": "euroc_optimizer_stable.yaml",
}
PROFILE_CANDIDATES = {
    "filter": [
        ("baseline", {}),
        (
            "robust_tracking",
            {
                "visual_observation_noise": 0.0035,
                "window_size": 12,
                "keyframe_parallax": 8.0,
                "feature_max_count": 260,
                "feature_min_distance": 18,
                "camera_tracking_rate_threshold": 0.65,
                "camera_state_prune_translation_threshold": 0.12,
            },
        ),
        (
            "conservative_motion",
            {
                "gyro_noise": 0.003,
                "acc_noise": 0.0065,
                "msckf_translation_threshold": 0.08,
                "camera_state_prune_rotation_threshold_deg": 8.0,
                "tbc": [-0.0216401454975, -0.064676986768, 0.00881073058949],
            },
        ),
    ],
    "optimizer": [
        ("baseline", {}),
        (
            "low_noise",
            {
                "visual_observation_noise": 0.0006,
                "window_size": 12,
                "keyframe_parallax": 7.0,
                "feature_max_count": 300,
                "msckf_huber_epsilon": 0.006,
            },
        ),
        (
            "conservative_window",
            {
                "visual_observation_noise": 0.0012,
                "window_size": 8,
                "camera_tracking_rate_threshold": 0.68,
                "camera_state_prune_translation_threshold": 0.10,
                "tbc": [-0.0216401454975, -0.064676986768, 0.01081073058949],
            },
        ),
    ],
}


@dataclass
class Pose:
    timestamp: float
    position: np.ndarray
    rotation: np.ndarray


@dataclass
class CaseResult:
    dataset: str
    mode: str
    profile: str
    config_path: Path
    output_dir: Path
    ate_rmse: float
    rpe_rmse: float
    final_error: float
    max_error: float
    coverage_ratio: float
    matched_count: int
    estimated_count: int
    groundtruth_count: int
    drifted: bool
    diverged: bool
    tracking_lost: bool
    stable: bool
    score: float


def run_cmd(cmd: Sequence[str], cwd: Path | None = None, timeout: int | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, timeout=timeout)


def ensure_built(repo_root: Path) -> Path:
    executable = repo_root / "build" / "data_test"
    if executable.exists():
        return executable
    run_cmd(["cmake", "-S", str(repo_root), "-B", str(repo_root / "build")], cwd=repo_root)
    run_cmd(["cmake", "--build", str(repo_root / "build"), "-j"], cwd=repo_root, timeout=1800)
    return executable


def ensure_dataset(repo_root: Path, data_root: Path, dataset: str, download: bool) -> Path:
    dataset_dir = data_root / dataset
    if (dataset_dir / "mav0" / "imu0" / "data.csv").exists():
        return dataset_dir
    if not download:
        raise FileNotFoundError(f"Dataset {dataset} not found in {data_root}")
    run_cmd([str(repo_root / "scripts" / "download_euroc.sh"), str(data_root)], cwd=repo_root, timeout=7200)
    if not (dataset_dir / "mav0" / "imu0" / "data.csv").exists():
        raise FileNotFoundError(f"Dataset {dataset} still missing after download")
    return dataset_dir


def replace_scalar(text: str, key: str, value: object) -> str:
    value_str = f"{value}"
    pattern = re.compile(rf"(^\s*{re.escape(key)}:\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(rf"\1{value_str}", text)
    return text + f"\n{key}: {value_str}\n"


def replace_matrix(text: str, key: str, values: Sequence[float]) -> str:
    data = ", ".join(f"{v:.15g}" for v in values)
    pattern = re.compile(
        rf"({re.escape(key)}:\s*!!opencv-matrix\s+rows:\s*\d+\s+cols:\s*\d+\s+dt:\s*d\s+data:\s*)\[[^\]]*\]",
        re.MULTILINE,
    )
    replacement = rf"\1[{data}]"
    return pattern.sub(replacement, text)


def render_config(template_path: Path, output_path: Path, dataset_dir: Path, overrides: Dict[str, object]) -> None:
    text = template_path.read_text()
    text = text.replace("__DATA_PATH__", str(dataset_dir.resolve()) + "/")
    for key, value in overrides.items():
        if key in {"tbc", "Rbc"}:
            text = replace_matrix(text, key, value)  # type: ignore[arg-type]
        else:
            text = replace_scalar(text, key, value)
    output_path.write_text(text)


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def load_tum(path: Path) -> List[Pose]:
    poses: List[Pose] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, px, py, pz, qx, qy, qz, qw = map(float, line.split())
            poses.append(Pose(ts, np.array([px, py, pz], dtype=float), quat_to_rot(qx, qy, qz, qw)))
    return poses


def load_euroc_gt(path: Path) -> List[Pose]:
    poses: List[Pose] = []
    with path.open() as f:
        reader = csv.reader(row for row in f if row and not row.startswith("#"))
        for row in reader:
            if len(row) < 8:
                continue
            ts = float(row[0]) * 1e-9
            px, py, pz = map(float, row[1:4])
            qw, qx, qy, qz = map(float, row[4:8])
            poses.append(Pose(ts, np.array([px, py, pz], dtype=float), quat_to_rot(qx, qy, qz, qw)))
    return poses


def associate(gt: Sequence[Pose], est: Sequence[Pose], max_delta: float = 0.02) -> List[Tuple[Pose, Pose]]:
    matches: List[Tuple[Pose, Pose]] = []
    gt_idx = 0
    est_idx = 0
    while gt_idx < len(gt) and est_idx < len(est):
        dt = est[est_idx].timestamp - gt[gt_idx].timestamp
        if abs(dt) <= max_delta:
            matches.append((gt[gt_idx], est[est_idx]))
            gt_idx += 1
            est_idx += 1
        elif dt < 0:
            est_idx += 1
        else:
            gt_idx += 1
    return matches


def umeyama_rigid(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = src_centered.T @ dst_centered / max(len(src), 1)
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_mean - r @ src_mean
    return r, t


def invert_pose(rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r_inv = rotation.T
    t_inv = -r_inv @ translation
    return r_inv, t_inv


def compose_pose(r1: np.ndarray, t1: np.ndarray, r2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return r1 @ r2, r1 @ t2 + t1


def align_matches(matches: Sequence[Tuple[Pose, Pose]]) -> List[Tuple[Pose, Pose]]:
    if len(matches) < 3:
        return list(matches)
    est_positions = np.stack([est.position for _, est in matches])
    gt_positions = np.stack([gt.position for gt, _ in matches])
    r_align, t_align = umeyama_rigid(est_positions, gt_positions)
    aligned: List[Tuple[Pose, Pose]] = []
    for gt_pose, est_pose in matches:
        aligned.append(
            (
                gt_pose,
                Pose(
                    est_pose.timestamp,
                    r_align @ est_pose.position + t_align,
                    r_align @ est_pose.rotation,
                ),
            )
        )
    return aligned


def compute_rpe(matches: Sequence[Tuple[Pose, Pose]], delta_seconds: float = 1.0, tolerance: float = 0.05) -> float:
    if len(matches) < 2:
        return float("inf")
    errors = []
    timestamps = [gt.timestamp for gt, _ in matches]
    for i, (gt_i, est_i) in enumerate(matches):
        target = gt_i.timestamp + delta_seconds
        j = next((idx for idx in range(i + 1, len(matches)) if abs(timestamps[idx] - target) <= tolerance), None)
        if j is None:
            continue
        gt_j, est_j = matches[j]
        r_gt_inv, t_gt_inv = invert_pose(gt_i.rotation, gt_i.position)
        r_est_inv, t_est_inv = invert_pose(est_i.rotation, est_i.position)
        r_gt_rel, t_gt_rel = compose_pose(r_gt_inv, t_gt_inv, gt_j.rotation, gt_j.position)
        r_est_rel, t_est_rel = compose_pose(r_est_inv, t_est_inv, est_j.rotation, est_j.position)
        r_err, t_err = compose_pose(r_gt_rel.T, -(r_gt_rel.T @ t_gt_rel), r_est_rel, t_est_rel)
        errors.append(float(np.linalg.norm(t_err)))
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(errors))))


def classify_run(errors: np.ndarray, coverage_ratio: float, estimated_count: int, gt_extent: float) -> Tuple[bool, bool, bool]:
    if estimated_count < 50 or coverage_ratio < 0.4:
        return False, False, True
    ate_rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("inf")
    max_error = float(np.max(errors)) if len(errors) else float("inf")
    if not np.isfinite(ate_rmse) or max_error > max(10.0, 2.0 * gt_extent) or ate_rmse > 5.0:
        return False, True, False
    drifted = ate_rmse > 1.5 or max_error > 3.0
    return drifted, False, False


def plot_trajectory(matches: Sequence[Tuple[Pose, Pose]], output_path: Path, title: str) -> None:
    if not matches:
        return
    gt_positions = np.stack([gt.position for gt, _ in matches])
    est_positions = np.stack([est.position for _, est in matches])
    plt.figure(figsize=(8, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], label="ground truth", linewidth=2)
    plt.plot(est_positions[:, 0], est_positions[:, 1], label="estimate", linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_case(dataset: str, mode: str, profile: str, config_path: Path, output_dir: Path, gt_path: Path) -> CaseResult:
    est_path = output_dir / "estimated_trajectory.tum"
    gt = load_euroc_gt(gt_path)
    est = load_tum(est_path) if est_path.exists() else []
    matches = align_matches(associate(gt, est))
    errors = np.array([np.linalg.norm(gt_pose.position - est_pose.position) for gt_pose, est_pose in matches], dtype=float)
    coverage_ratio = len(matches) / len(gt) if gt else 0.0
    ate_rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("inf")
    rpe_rmse = compute_rpe(matches)
    final_error = float(errors[-1]) if len(errors) else float("inf")
    max_error = float(np.max(errors)) if len(errors) else float("inf")
    gt_extent = float(np.max(np.linalg.norm(np.stack([pose.position for pose in gt]), axis=1))) if gt else 0.0
    drifted, diverged, tracking_lost = classify_run(errors, coverage_ratio, len(est), gt_extent)
    stable = not drifted and not diverged and not tracking_lost
    score = ate_rmse + 2.0 * rpe_rmse + (10.0 if drifted else 0.0) + (50.0 if diverged else 0.0) + (25.0 if tracking_lost else 0.0)
    plot_trajectory(matches, output_dir / "trajectory_comparison.svg", f"{dataset} - {mode} - {profile}")
    metrics_payload = {
        "dataset": dataset,
        "mode": mode,
        "profile": profile,
        "ate_rmse": ate_rmse,
        "rpe_rmse": rpe_rmse,
        "final_error": final_error,
        "max_error": max_error,
        "coverage_ratio": coverage_ratio,
        "matched_count": len(matches),
        "estimated_count": len(est),
        "groundtruth_count": len(gt),
        "drifted": drifted,
        "diverged": diverged,
        "tracking_lost": tracking_lost,
        "stable": stable,
        "score": score,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False))
    return CaseResult(
        dataset=dataset,
        mode=mode,
        profile=profile,
        config_path=config_path,
        output_dir=output_dir,
        ate_rmse=ate_rmse,
        rpe_rmse=rpe_rmse,
        final_error=final_error,
        max_error=max_error,
        coverage_ratio=coverage_ratio,
        matched_count=len(matches),
        estimated_count=len(est),
        groundtruth_count=len(gt),
        drifted=drifted,
        diverged=diverged,
        tracking_lost=tracking_lost,
        stable=stable,
        score=score,
    )


def run_profile(repo_root: Path, executable: Path, mode: str, dataset: str, dataset_dir: Path, profile: str, overrides: Dict[str, object]) -> CaseResult:
    runs_root = repo_root / "outputs" / "euroc" / mode / dataset / profile
    if runs_root.exists():
        shutil.rmtree(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    config_path = runs_root / "run_config.yaml"
    render_config(repo_root / "config" / TEMPLATE_BY_MODE[mode], config_path, dataset_dir, overrides)
    run_cmd([str(executable), "--config", str(config_path), "--output-dir", str(runs_root)], cwd=repo_root, timeout=1800)
    return evaluate_case(
        dataset=dataset,
        mode=mode,
        profile=profile,
        config_path=config_path,
        output_dir=runs_root,
        gt_path=dataset_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv",
    )


def select_best_result(results: Sequence[CaseResult]) -> CaseResult:
    stable_results = [result for result in results if result.stable]
    pool = stable_results if stable_results else list(results)
    return min(pool, key=lambda result: result.score)


def summarize(results: Sequence[CaseResult], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "euroc_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "mode",
                "profile",
                "ate_rmse",
                "rpe_rmse",
                "final_error",
                "max_error",
                "coverage_ratio",
                "matched_count",
                "estimated_count",
                "groundtruth_count",
                "drifted",
                "diverged",
                "tracking_lost",
                "stable",
                "config_path",
                "output_dir",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.dataset,
                    result.mode,
                    result.profile,
                    result.ate_rmse,
                    result.rpe_rmse,
                    result.final_error,
                    result.max_error,
                    result.coverage_ratio,
                    result.matched_count,
                    result.estimated_count,
                    result.groundtruth_count,
                    int(result.drifted),
                    int(result.diverged),
                    int(result.tracking_lost),
                    int(result.stable),
                    result.config_path,
                    result.output_dir,
                ]
            )

    grouped: Dict[Tuple[str, str], List[CaseResult]] = {}
    for result in results:
        grouped.setdefault((result.dataset, result.mode), []).append(result)

    lines = ["# EuRoC Benchmark Summary", "", "| Dataset | Mode | Profile | ATE RMSE | RPE RMSE | Coverage | Health | Plot |", "|---|---|---:|---:|---:|---:|---|---|"]
    for (dataset, mode), entries in sorted(grouped.items()):
        best = select_best_result(entries)
        health = "stable"
        if best.tracking_lost:
            health = "tracking_lost"
        elif best.diverged:
            health = "diverged"
        elif best.drifted:
            health = "drifted"
        lines.append(
            f"| {dataset} | {mode} | {best.profile} | {best.ate_rmse:.4f} | {best.rpe_rmse:.4f} | {best.coverage_ratio:.2%} | {health} | {best.output_dir / 'trajectory_comparison.svg'} |"
        )
        lines.append(f"|  |  | config |  |  |  |  | {best.config_path} |")
    lines.append("")
    lines.append(f"- metrics table: `{csv_path}`")
    lines.append("- per-run metrics: `outputs/euroc/<mode>/<dataset>/<profile>/metrics.json`")
    (report_dir / "euroc_summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--mode", choices=["all", "filter", "optimizer"], default="all")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve() if args.data_root else repo_root / "data" / "euroc"
    modes = ["filter", "optimizer"] if args.mode == "all" else [args.mode]
    executable = ensure_built(repo_root)

    all_results: List[CaseResult] = []
    for dataset in args.datasets:
        dataset_dir = ensure_dataset(repo_root, data_root, dataset, not args.no_download)
        for mode in modes:
            mode_results: List[CaseResult] = []
            for profile, overrides in PROFILE_CANDIDATES[mode]:
                result = run_profile(repo_root, executable, mode, dataset, dataset_dir, profile, overrides)
                mode_results.append(result)
                if result.stable:
                    break
            best = select_best_result(mode_results)
            all_results.extend(mode_results)
            print(
                f"[{dataset}][{mode}] best profile={best.profile} "
                f"ATE={best.ate_rmse:.4f} RPE={best.rpe_rmse:.4f} "
                f"stable={best.stable} drifted={best.drifted} diverged={best.diverged} tracking_lost={best.tracking_lost}"
            )

    summarize(all_results, repo_root / "reports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
