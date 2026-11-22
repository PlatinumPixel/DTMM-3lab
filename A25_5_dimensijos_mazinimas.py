# 5-to skyriaus projekcijos is A25_normuotas.csv. Reikia NumPy ir Pillow biblioteku.
# Skripto naudojimas: python [skripto_pavadinimas] A25_normuotas.csv [output direktorija]

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 0
PERPLEXITY = 30
ITERATIONS = 1000
COLORS = {"Bumps": "#BA7800", "K_Scratch": "#B33346", "Other_Faults": "#2766A0"}


def distances_squared(points):
    norms = np.einsum("ij,ij->i", points, points)
    distances = norms[:, None] + norms[None, :] - 2 * (points @ points.T)
    np.maximum(distances, 0, out=distances)
    np.fill_diagonal(distances, 0)
    return distances


def joint_probabilities(distances):
    """Find a Gaussian bandwidth per observation with the requested perplexity."""
    n = len(distances)
    beta = np.ones(n)
    low = np.zeros(n)
    high = np.full(n, np.inf)
    target = np.log(PERPLEXITY)
    for _ in range(80):
        probabilities = np.exp(-distances * beta[:, None])
        np.fill_diagonal(probabilities, 0)
        totals = probabilities.sum(axis=1)
        probabilities /= totals[:, None]
        entropy = np.log(totals) + beta * np.sum(probabilities * distances, axis=1)
        error = entropy - target
        if np.max(np.abs(error)) < 1e-5:
            break
        low = np.where(error > 0, beta, low)
        high = np.where(error < 0, beta, high)
        beta = np.where(np.isfinite(high), (low + high) / 2, beta * 2)
    else:
        raise RuntimeError("Could not reach the requested perplexity.")
    return (probabilities + probabilities.T) / (2 * n)


def tsne(distances):
    """Fit exact two-dimensional t-SNE with a fixed random initialization."""
    p = joint_probabilities(distances)
    rng = np.random.default_rng(SEED)
    points = rng.normal(0, 1e-4, (len(distances), 2))
    velocity = np.zeros_like(points)
    gains = np.ones_like(points)
    for iteration in range(ITERATIONS):
        numerator = distances_squared(points)
        numerator += 1
        np.reciprocal(numerator, out=numerator)
        np.fill_diagonal(numerator, 0)
        q = numerator / numerator.sum()
        exaggeration = 12 if iteration < 250 else 1
        weights = (exaggeration * p - q) * numerator
        gradient = 4 * (weights.sum(axis=1)[:, None] * points - weights @ points)
        gains = np.maximum(
            np.where(np.sign(gradient) != np.sign(velocity), gains + 0.2, gains * 0.8),
            0.01,
        )
        momentum = 0.5 if iteration < 250 else 0.8
        velocity = momentum * velocity - 200 * gains * gradient
        points += velocity
        points -= points.mean(axis=0)
    return points


def neighborhood_ranks(distances):
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    order = distances.argsort(axis=1, kind="stable")
    return order, order.argsort(axis=1) + 1


def neighborhood_overlap(original_order, projected_points, k=15):
    original_neighbors = original_order[:, :k]
    _, rank = neighborhood_ranks(distances_squared(projected_points))
    return (np.take_along_axis(rank, original_neighbors, axis=1) <= k).mean()


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    mac_font = Path("/System/Library/Fonts/Supplemental") / name
    if mac_font.exists():
        return ImageFont.truetype(str(mac_font), size)
    return ImageFont.load_default(size=size)


def plot_projections(path, pca_2d, tsne_2d, classes, explained):
    image = Image.new("RGB", (1680, 870), "white")
    draw = ImageDraw.Draw(image)
    draw.text((68, 42), "A25 defektų projekcijos", fill="#162536", font=font(43, True))
    draw.text((69, 103), "1 050 objektų · 25 normuoti požymiai · A rinkinys",
              fill="#485868", font=font(24))
    panels = (
        (pca_2d, "PCA", f"PC1 ({explained[0]:.1%})", f"PC2 ({explained[1]:.1%})"),
        (tsne_2d, "t-SNE · perplexity = 30", "t-SNE 1", "t-SNE 2"),
    )
    for index, (points, title, x_label, y_label) in enumerate(panels):
        left = 111 + 820 * index
        top, width, height = 223, 655, 486
        draw.text((left, 172), title, fill="#162536", font=font(29, True))
        draw.rectangle((left, top, left + width, top + height), fill="#F5F8FA",
                       outline="#B0BCCA", width=2)
        for fraction in (0.25, 0.5, 0.75):
            draw.line((left + width * fraction, top, left + width * fraction, top + height),
                      fill="#DDE4EB")
            draw.line((left, top + height * fraction, left + width, top + height * fraction),
                      fill="#DDE4EB")
        scale = min(width / np.ptp(points[:, 0]), height / np.ptp(points[:, 1])) * 0.91
        midpoint = (points.max(axis=0) + points.min(axis=0)) / 2
        plotted = (points - midpoint) * [scale, -scale] + [left + width / 2, top + height / 2]
        layer = Image.new("RGBA", image.size)
        dots = ImageDraw.Draw(layer)
        for (x, y), label in zip(plotted, classes):
            color = COLORS[label]
            rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            dots.ellipse((x - 3.1, y - 3.1, x + 3.1, y + 3.1), fill=(*rgb, 165))
        image.paste(layer, (0, 0), layer)
        draw.text((left + width / 2, top + height + 36), x_label, fill="#273A4B",
                  font=font(23), anchor="mm")
        rotated = Image.new("RGBA", (240, 36))
        ImageDraw.Draw(rotated).text((120, 18), y_label, fill="#273A4B",
                                     font=font(23), anchor="mm")
        image.paste(rotated.rotate(90, expand=True),
                    (left - 55, top + height // 2 - 120), rotated.rotate(90, expand=True))
    for index, (label, color) in enumerate(COLORS.items()):
        x = 525 + 225 * index
        draw.ellipse((x, 800, x + 16, 816), fill=color)
        draw.text((x + 26, 808), label, fill="#273A4B", font=font(24), anchor="lm")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A25_normuotas.csv")
    parser.add_argument("output", type=Path, help="Output directory")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        features = [column for column in reader.fieldnames if column not in {"id", "class"}]
        rows = list(reader)
    data = np.array([[float(row[column]) for column in features] for row in rows])
    ids = [row["id"] for row in rows]
    classes = [row["class"] for row in rows]
    if data.shape != (1050, 25) or len(set(ids)) != len(ids) or not np.isfinite(data).all():
        raise ValueError("Expected 1050 distinct IDs and 25 finite features.")

    centered = data - data.mean(axis=0)
    left, singular_values, loadings = np.linalg.svd(centered, full_matrices=False)
    signs = np.sign(loadings[np.arange(len(singular_values)), np.argmax(abs(loadings), axis=1)])
    scores = left * singular_values * signs
    explained = singular_values**2 / np.sum(singular_values**2)
    dimensions = np.searchsorted(np.cumsum(explained), 0.90) + 1

    original_distances = distances_squared(data)
    reduced = tsne(original_distances)
    original_order, _ = neighborhood_ranks(original_distances)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "A25_5_koordinates.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest)
        writer.writerow(["id", "class", *[f"PC{i}" for i in range(1, dimensions + 1)],
                         "tsne_1", "tsne_2"])
        writer.writerows([id_, label, *pca, *tsne_coords] for id_, label, pca, tsne_coords
                         in zip(ids, classes, scores[:, :dimensions], reduced))
    plot_projections(args.output / "A25_5_projekcijos.png", scores[:, :2], reduced,
                     classes, explained)
    print("Explained variance by component:", explained[:dimensions])
    print("Cumulative explained variance:", np.cumsum(explained[:dimensions]))
    for name, points in (("PCA 2D", scores[:, :2]), (f"PCA {dimensions}D", scores[:, :dimensions]),
                         ("t-SNE 2D", reduced)):
        print(name, "NH15 =", neighborhood_overlap(original_order, points))


if __name__ == "__main__":
    main()
