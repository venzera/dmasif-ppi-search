from __future__ import annotations

import argparse
import csv
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d

from .io import load_embedding, load_json


OUTPUT_COLUMNS = (
    "target_id",
    "status",
    "max_z",
    "max_dot",
    "n_matches",
    "n_correspondences",
    "n_ransac_inliers",
    "n_icp_correspondences",
    "fitness",
    "inlier_rmse",
    "ransac_score",
    "transform",
)


def build_correspondences(q1, q2, t1, t2, stats, z_threshold, max_correspondences):
    z_a = (q1 @ t2.T - stats["mean_a"]) / stats["std_a"]
    z_b = (q2 @ t1.T - stats["mean_b"]) / stats["std_b"]
    best = np.maximum(z_a, z_b)
    query_index, target_index = np.nonzero(best > z_threshold)
    if max_correspondences and len(query_index) > max_correspondences:
        values = best[query_index, target_index]
        keep = np.argpartition(values, -max_correspondences)[-max_correspondences:]
        query_index, target_index = query_index[keep], target_index[keep]
    return np.column_stack((query_index, target_index)).astype(np.int32)


def point_cloud(xyz, normal_radius, normal_max_nn):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    return cloud


def align(
    query_xyz,
    target_xyz,
    correspondences,
    ransac_radius,
    ransac_iterations,
    ransac_confidence,
    ransac_n,
    edge_threshold,
    distance_threshold,
    normal_angle,
    normal_radius,
    normal_max_nn,
    icp_distance,
):
    query = point_cloud(query_xyz, normal_radius, normal_max_nn)
    target = point_cloud(target_xyz, normal_radius, normal_max_nn)
    registration = o3d.pipelines.registration
    result = registration.registration_ransac_based_on_correspondence(
        query,
        target,
        o3d.utility.Vector2iVector(correspondences),
        ransac_radius,
        registration.TransformationEstimationPointToPoint(False),
        ransac_n,
        [
            registration.CorrespondenceCheckerBasedOnEdgeLength(edge_threshold),
            registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            registration.CorrespondenceCheckerBasedOnNormal(normal_angle),
        ],
        registration.RANSACConvergenceCriteria(
            ransac_iterations, ransac_confidence
        ),
    )
    n_ransac_inliers = len(result.correspondence_set)
    if n_ransac_inliers:
        result = registration.registration_icp(
            query,
            target,
            icp_distance,
            result.transformation,
            registration.TransformationEstimationPointToPlane(),
        )
    return result, n_ransac_inliers


def process_query(task):
    (
        search_path,
        query_dir,
        target_dir,
        output_dir,
        stats,
        settings,
    ) = task
    output = output_dir / search_path.name
    if output.exists():
        return search_path.stem, -1
    o3d.utility.random.seed(
        (settings["seed"] + zlib.crc32(search_path.stem.encode("utf-8"))) % (2**31)
    )
    query_xyz, q1, q2 = load_embedding(
        query_dir / (search_path.stem + ".npz"), settings["site_threshold"]
    )
    with search_path.open(newline="") as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))[: settings["top_n"]]

    rows: List[dict] = []
    for candidate in candidates:
        target_id = candidate["target_id"]
        row = {column: "" for column in OUTPUT_COLUMNS}
        row.update(
            target_id=target_id,
            max_z=candidate.get("max_z", ""),
            max_dot=candidate.get("max_dot", ""),
            n_matches=candidate.get("n_matches", ""),
        )
        target_path = target_dir / (target_id + ".npz")
        if not target_path.exists():
            row["status"] = "missing_embedding"
            rows.append(row)
            continue
        target_xyz, t1, t2 = load_embedding(target_path, settings["site_threshold"])
        correspondences = build_correspondences(
            q1,
            q2,
            t1,
            t2,
            stats,
            settings["z_threshold"],
            settings["max_correspondences"],
        )
        row["n_correspondences"] = len(correspondences)
        if len(correspondences) < settings["min_correspondences"]:
            row["status"] = "too_few_correspondences"
            rows.append(row)
            continue
        result, n_ransac_inliers = align(
            query_xyz,
            target_xyz,
            correspondences,
            settings["ransac_radius"],
            settings["ransac_iterations"],
            settings["ransac_confidence"],
            settings["ransac_n"],
            settings["edge_threshold"],
            settings["distance_threshold"],
            settings["normal_angle"],
            settings["normal_radius"],
            settings["normal_max_nn"],
            settings["icp_distance"],
        )
        n_icp_correspondences = len(result.correspondence_set)
        row.update(
            status="ok" if n_icp_correspondences else "ransac_no_alignment",
            n_ransac_inliers=n_ransac_inliers,
            n_icp_correspondences=n_icp_correspondences,
            fitness="{:.6g}".format(result.fitness),
            inlier_rmse="{:.6g}".format(result.inlier_rmse),
            ransac_score="{:.6g}".format(
                n_icp_correspondences / (1.0 + result.inlier_rmse)
                if n_icp_correspondences
                else 0.0
            ),
            transform=json.dumps(np.asarray(result.transformation).round(8).tolist()),
        )
        rows.append(row)
    rows.sort(key=lambda row: float(row["ransac_score"] or 0), reverse=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return search_path.stem, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RANSAC + ICP geometric rescoring.")
    parser.add_argument("--search-dir", required=True, type=Path)
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--target-embeddings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-threshold", type=float, default=None)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--min-correspondences", type=int, default=3)
    parser.add_argument("--max-correspondences", type=int, default=0)
    parser.add_argument("--normal-radius", type=float, default=3.0)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--ransac-radius", type=float, default=1.0)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-n", type=int, default=3)
    parser.add_argument("--edge-threshold", type=float, default=0.9)
    parser.add_argument("--distance-threshold", type=float, default=1.5)
    parser.add_argument("--normal-angle", type=float, default=np.pi / 2)
    parser.add_argument("--icp-distance", type=float, default=1.0)
    args = parser.parse_args()

    stats = load_json(args.search_dir / "background.json")
    site_threshold = (
        stats["site_threshold"] if args.site_threshold is None else args.site_threshold
    )
    settings = vars(args).copy()
    settings["site_threshold"] = site_threshold
    for key in (
        "search_dir",
        "query_embeddings",
        "target_embeddings",
        "output_dir",
        "workers",
    ):
        settings.pop(key, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in args.search_dir.glob("*.tsv") if path.name != "background.tsv"
    )
    tasks = [
        (
            path,
            args.query_embeddings,
            args.target_embeddings,
            args.output_dir,
            stats,
            settings,
        )
        for path in files
    ]
    if args.workers == 1:
        results = map(process_query, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(process_query, tasks)
    for index, (query_id, count) in enumerate(results, start=1):
        print("[{}/{}] {}: {}".format(index, len(tasks), query_id, count), flush=True)
    if args.workers != 1:
        pool.shutdown()


if __name__ == "__main__":
    main()
