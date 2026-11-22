"""Summarize PCA7 k-means clusters using the original cleaned feature values."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, quantiles


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean", type=Path, help="A25_clean.csv")
    parser.add_argument("assignments", type=Path, help="A25_6_priskyrimai.csv")
    parser.add_argument("output", type=Path, help="Output CSV path")
    args = parser.parse_args()
    clean = read_rows(args.clean)
    assignments = read_rows(args.assignments)
    # A25_normuotas.csv assigned sequential IDs to rows of A25_clean.csv.
    by_id = {int(row["id"]): row for row in assignments}
    if len(assignments) != len(clean) or set(by_id) != set(range(1, len(clean) + 1)):
        raise ValueError("Assignment IDs do not match cleaned dataset rows.")

    groups = defaultdict(list)
    for index, row in enumerate(clean, 1):
        assignment = by_id[index]
        if row["class"] != assignment["class"]:
            raise ValueError(f"Class mismatch for object {index}.")
        groups[int(assignment["PCA7_Kmeans"])].append(row)

    features = [key for key in clean[0] if key != "class"]
    classes = sorted({row["class"] for row in clean})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["klasteris", "n", *classes, *(
            f"{feature}_{stat}" for feature in features for stat in ("Q1", "mediana", "Q3")
        )])
        for cluster, members in sorted(groups.items()):
            class_counts = Counter(row["class"] for row in members)
            stats = []
            for feature in features:
                values = [float(row[feature]) for row in members]
                q1, _, q3 = quantiles(values, n=4, method="inclusive")
                stats.extend((q1, median(values), q3))
            writer.writerow([cluster, len(members), *(class_counts[c] for c in classes), *stats])


if __name__ == "__main__":
    main()
