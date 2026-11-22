"""Run with NumPy, Pillow and Rscript. See --help for inputs."""
import argparse
import csv
import importlib.util
import itertools
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SPACES = ("Originali", "PCA7", "TSNE2")
ALGORITHMS = ("Kmeans", "Ward", "DBSCAN")
LOG_FEATURES = ("Pixels_Areas", "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity")
R_WORKER = """
args <- commandArgs(trailingOnly = TRUE)
# Load only function definitions, without executing the section 6 workflow.
for (expression in parse(args[1])) {
  if (is.call(expression) && identical(expression[[1]], as.name("<-")) &&
      is.symbol(expression[[2]]) &&
      as.character(expression[[2]]) %in% c("dbscan_radius", "knee", "fit")) {
    eval(expression)
  }
}
x <- as.matrix(read.csv(args[2], header = FALSE))
write.csv(as.data.frame(fit(x)$labels), args[3], row.names = FALSE)
"""


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows, columns):
    return np.array([[float(row[column]) for column in columns] for row in rows])


def normalize(raw, columns):
    data = raw.copy()
    log_columns = [columns.index(column) for column in LOG_FEATURES]
    data[:, log_columns] = np.log1p(data[:, log_columns])
    q1, median, q3 = np.quantile(data, [0.25, 0.5, 0.75], axis=0)
    iqr = q3 - q1
    return (data - median) / np.where(iqr > 0, iqr, 1)


def external(a, b):
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    counts = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(counts, (ai, bi), 1)
    n = len(a)
    row, col = counts.sum(axis=1), counts.sum(axis=0)

    def pairs(x):
        return np.sum(x * (x - 1) / 2)

    intersection, a_pairs, b_pairs = pairs(counts), pairs(row), pairs(col)
    expected = a_pairs * b_pairs / pairs(n)
    ari_den = (a_pairs + b_pairs) / 2 - expected
    union = a_pairs + b_pairs - intersection
    i, j = np.nonzero(counts)
    p = counts[i, j] / n
    mi = np.sum(p * np.log(p / ((row[i] / n) * (col[j] / n))))
    entropy_sum = -sum(np.sum((m / n) * np.log(m / n)) for m in (row, col))
    return {
        "ARI": (intersection - expected) / ari_den if ari_den else 1.0,
        "NMI": 2 * mi / entropy_sum if entropy_sum else 1.0,
        "Jaccard": intersection / union if union else 1.0,
    }


def internal(x, labels, distances_squared):
    keep = labels > 0
    groups = np.unique(labels[keep])
    n, k = keep.sum(), len(groups)
    if not 2 <= k < n:
        return dict(Silhouette=np.nan, CH=np.nan, DB=np.nan)
    x, labels = x[keep], labels[keep]
    distances = np.sqrt(distances_squared[np.ix_(keep, keep)])
    masks = [labels == group for group in groups]
    sizes = np.array([mask.sum() for mask in masks])
    mean_distances = np.column_stack([distances[:, mask].mean(axis=1) for mask in masks])
    sil = np.zeros(n)
    for i, mask in enumerate(masks):
        if sizes[i] > 1:
            within = mean_distances[mask, i] * sizes[i] / (sizes[i] - 1)
            other = mean_distances[mask][:, np.arange(k) != i].min(axis=1)
            den = np.maximum(within, other)
            sil[mask] = np.divide(other - within, den, out=np.zeros_like(den), where=den > 0)
    centroids = np.array([x[mask].mean(axis=0) for mask in masks])
    residuals = [np.linalg.norm(x[mask] - center, axis=1) for mask, center in zip(masks, centroids)]
    within_ss = sum(np.sum(r**2) for r in residuals)
    between_ss = np.sum(sizes[:, None] * (centroids - x.mean(axis=0))**2)
    scatter = np.array([r.mean() for r in residuals])
    separation = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    np.fill_diagonal(separation, np.inf)
    db = ((scatter[:, None] + scatter) / separation).max(axis=1).mean()
    return dict(Silhouette=sil.mean(), CH=(between_ss / (k - 1)) / (within_ss / (n - k)), DB=db)


def score(labels, spaces, projection_tools, trial):
    rows = []
    for space, coordinates in spaces.items():
        distances = projection_tools.distances_squared(coordinates)
        for algo in ALGORITHMS:
            key = f"{space}_{algo}"
            assigned = labels[key]
            row = dict(trial=trial, space=space, algorithm=algo,
                       clusters=len(np.unique(assigned[assigned > 0])), noise=np.sum(assigned == 0))
            row.update(internal(coordinates, assigned, distances))
            if trial == "full":
                row.update(external(labels[f"Originali_{algo}"], assigned))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, help_text in (
        ("clean", "A25_clean.csv"), ("normalized", "A25_normuotas.csv"),
        ("coordinates", "A25_5_koordinates.csv"), ("assignments", "A25_6_priskyrimai.csv"),
        ("projection_script", "A25_5_dimensijos_mazinimas.py"),
        ("clustering_script", "A25_6_klasterizavimas.R"), ("output", "Output directory"),
    ):
        parser.add_argument(name, type=Path, help=help_text)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("projection_tools", args.projection_script)
    projection_tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection_tools)
    clean, normalized = read_csv(args.clean), read_csv(args.normalized)
    ids = [int(row["id"]) for row in normalized]
    assert ids == list(range(1, 1051))
    assert [r["class"] for r in clean] == [r["class"] for r in normalized]

    def aligned(path):
        rows = read_csv(path)
        by_id = {int(r["id"]): r for r in rows}
        assert len(rows) == len(by_id) == len(ids)
        aligned_rows = [by_id[i] for i in ids]
        assert [r["class"] for r in aligned_rows] == [r["class"] for r in normalized]
        return aligned_rows

    columns = [c for c in normalized[0] if c not in {"id", "class"}]
    raw = matrix(clean, columns)
    original = matrix(normalized, columns)
    np.testing.assert_allclose(normalize(raw, columns), original, atol=1e-10)
    coords, assigned = aligned(args.coordinates), aligned(args.assignments)
    spaces = dict(Originali=original, PCA7=matrix(coords, [f"PC{i}" for i in range(1, 8)]),
                  TSNE2=matrix(coords, ["tsne_1", "tsne_2"]))
    keys = [f"{space}_{algo}" for space in SPACES for algo in ALGORITHMS]
    labels = {key: matrix(assigned, [key]).ravel().astype(int) for key in keys}
    full = score(labels, spaces, projection_tools, "full")

    subsample_rows, samples = [], []
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        worker, input_path, output_path = [temp / name for name in ("fit.R", "x.csv", "labels.csv")]
        worker.write_text(R_WORKER)
        for trial in range(10):
            rng = np.random.default_rng(trial)
            indices = np.sort(rng.choice(len(ids), 840, replace=False))
            x = normalize(raw[indices], columns)
            left, singular_values, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
            projection_tools.SEED = trial
            reduced = dict(Originali=x, PCA7=left[:, :7] * singular_values[:7],
                           TSNE2=projection_tools.tsne(projection_tools.distances_squared(x)))
            sample_labels = {}
            for space, coordinates in reduced.items():
                np.savetxt(input_path, coordinates, delimiter=",")
                subprocess.run(["Rscript", str(worker), str(args.clustering_script.resolve()),
                                str(input_path), str(output_path)], check=True)
                fitted = read_csv(output_path)
                for algo in ALGORITHMS:
                    sample_labels[f"{space}_{algo}"] = np.array([int(row[algo]) for row in fitted])
            samples.append((indices, sample_labels))
            subsample_rows.extend(score(sample_labels, reduced, projection_tools, trial))
            print(f"Imtis {trial + 1}/10", flush=True)

    pairs = []
    for (i, (a_ids, a)), (j, (b_ids, b)) in itertools.combinations(enumerate(samples), 2):
        common, ai, bi = np.intersect1d(a_ids, b_ids, return_indices=True)
        for space, algo in itertools.product(SPACES, ALGORITHMS):
            key = f"{space}_{algo}"
            pairs.append(dict(space=space, algorithm=algo, sample_a=i, sample_b=j,
                              shared_objects=len(common), **external(a[key][ai], b[key][bi])))
    summary = []
    for space, algorithm in itertools.product(SPACES, ALGORITHMS):
        row = dict(space=space, algorithm=algorithm)
        for dataset, metrics in ((pairs, ("ARI", "NMI", "Jaccard")),
                                 (subsample_rows, ("Silhouette", "CH", "DB"))):
            subset = [r for r in dataset if r["space"] == space and r["algorithm"] == algorithm]
            for metric in metrics:
                values = np.array([r[metric] for r in subset])
                values = values[np.isfinite(values)]
                row[f"{metric}_n"] = len(values)
                row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
                row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
                row[f"{metric}_min"] = values.min() if len(values) else np.nan
        summary.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("A25_7_kokybe.csv", full), ("A25_7_stabilumas.csv", summary),
                           ("A25_7_imtys.csv", subsample_rows), ("A25_7_imciu_poros.csv", pairs)):
        write_csv(args.output / filename, rows)


if __name__ == "__main__":
    main()
