import argparse
import csv
from itertools import combinations
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from A25_7_vertinimas import ALGORITHMS, R_WORKER, external, matrix, normalize, read_csv

PLOT_R = """
args <- commandArgs(trailingOnly = TRUE)
res <- read.csv(args[1])
png(args[2], width = 1600, height = 650, res = 160, type = "cairo")
par(mfrow = c(1,3), mar = c(4,4,2.5,0.8), las = 1)
colors <- c("#244D7A", "#B34913", "#437653")
spaces <- c("Originali", "PCA7", "TSNE2")
for (algo in c("Kmeans", "Ward", "DBSCAN")) {
  plot(NA, xlim = c(4,21), ylim = c(0.45,1), xaxt = "n",
       xlab = "Imčių skaičius", ylab = "Vidutinis ARI",
       main = if (algo == "Kmeans") "K-means" else algo, bty = "l")
  axis(1, at = c(5,10,15,20))
  abline(h = seq(0.5,1,0.1), col = "gray90")
  for (j in seq_along(spaces)) {
    s <- res[res$algorithm == algo & res$space == spaces[j], ]
    lines(s$N, s$ARI, type = "b", lwd = 2, col = colors[j], pch = c(16,17,15)[j])
  }
  if (algo == "Kmeans") {
    legend("bottomright", c("Pradinė erdvė", "PCA (7D)", "t-SNE (2D)"),
           col = colors, lty = 1, pch = c(16,17,15), bty = "n", cex = 0.85)
  }
}
invisible(dev.off())
"""


def collect_partitions(raw, columns, tmp, clustering):
    worker = tmp / "fit.R"
    input_csv = tmp / "matrix.csv"
    output_csv = tmp / "labels.csv"
    worker.write_text(R_WORKER, encoding="utf-8")
    samples = []
    for seed in range(20):
        indices = np.sort(np.random.default_rng(seed).choice(len(raw), 840, replace=False))
        values = normalize(raw[indices], columns)
        left, singular, _ = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
        matrices = {"Originali": values, "PCA7": left[:, :7] * singular[:7]}
        partitions = {}
        for space, coordinates in matrices.items():
            np.savetxt(input_csv, coordinates, delimiter=",")
            subprocess.run(
                ["Rscript", str(worker), str(clustering), str(input_csv), str(output_csv)],
                check=True,
            )
            fitted = read_csv(output_csv)
            for algorithm in ALGORITHMS:
                partitions[space, algorithm] = np.array([int(row[algorithm]) for row in fitted])
        samples.append((indices, partitions))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean", type=Path)
    parser.add_argument("pairs", type=Path, help="A25_7_imciu_poros.csv from section 7")
    parser.add_argument("output", type=Path, help="Output PNG path")
    args = parser.parse_args()
    clustering = Path(__file__).resolve().with_name("A25_6_klasterizavimas.R")
    clean = read_csv(args.clean)
    columns = [key for key in clean[0] if key not in {"id", "class"}]
    raw = matrix(clean, columns)
    pairs = [
        (row["space"], row["algorithm"], int(row["sample_b"]), float(row["ARI"]))
        for row in read_csv(args.pairs)
        if row["space"] == "TSNE2"
    ]
    for algorithm in ALGORITHMS:
        if sum(row[1] == algorithm for row in pairs) != 45:
            raise ValueError(f"Expected 45 t-SNE sample pairs for {algorithm}")

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        samples = collect_partitions(raw, columns, tmp, clustering)
        for (_, (ids_a, labels_a)), (j, (ids_b, labels_b)) in combinations(enumerate(samples), 2):
            _, ai, bi = np.intersect1d(ids_a, ids_b, return_indices=True)
            for space, algorithm in labels_a:
                key = space, algorithm
                pairs.append((space, algorithm, j, external(labels_a[key][ai], labels_b[key][bi])["ARI"]))

        curve_csv = tmp / "curve.csv"
        with curve_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("space", "algorithm", "N", "ARI"))
            for space in ("Originali", "PCA7", "TSNE2"):
                for algorithm in ALGORITHMS:
                    for n in ((5, 10) if space == "TSNE2" else (5, 10, 15, 20)):
                        values = [ari for s, a, j, ari in pairs if s == space and a == algorithm and j < n]
                        if len(values) != n * (n - 1) // 2:
                            raise ValueError(f"Incomplete sample pairs for {space}, {algorithm}, N={n}")
                        writer.writerow((space, algorithm, n, np.mean(values)))
        plot_script = tmp / "plot.R"
        plot_script.write_text(PLOT_R, encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["Rscript", str(plot_script), str(curve_csv), str(args.output)], check=True)
        if not args.output.is_file():
            raise RuntimeError("R did not create the PNG. Check whether R supports Cairo PNG output.")
    print(args.output)


if __name__ == "__main__":
    main()
