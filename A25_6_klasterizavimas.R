# Naudojimas komandineje eiluteje: Rscript A25_6_klasterizavimas.R A25_normuotas.csv A25_5_koordinates.csv rezultatai_6

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
original <- read.csv(args[1])
projection <- read.csv(args[2])
stopifnot(!anyDuplicated(original$id), !anyDuplicated(projection$id))
projection <- projection[match(original$id, projection$id), ]
stopifnot(identical(original$id, projection$id), identical(original$class, projection$class))
out <- args[3]
dir.create(out, recursive = TRUE, showWarnings = FALSE)

spaces <- list(
  Originali = as.matrix(original[setdiff(names(original), c("id", "class"))]),
  PCA7 = as.matrix(projection[paste0("PC", 1:7)]),
  TSNE2 = as.matrix(projection[c("tsne_1", "tsne_2")])
)

# minPts includes the point itself. Labels 0 indicate noise.
dbscan_radius <- function(d, eps, min_pts) {
  neighbors <- lapply(seq_len(nrow(d)), function(i) which(d[i, ] <= eps))
  core <- lengths(neighbors) >= min_pts
  labels <- integer(nrow(d))
  group <- 0L
  for (i in which(core)) {
    if (labels[i] != 0L) next
    group <- group + 1L
    labels[i] <- group
    queue <- i
    head <- 1L
    while (head <= length(queue)) {
      fresh <- neighbors[[queue[head]]]
      fresh <- fresh[labels[fresh] == 0L]
      labels[fresh] <- group
      queue <- c(queue, fresh[core[fresh]])
      head <- head + 1L
    }
  }
  labels
}

knee <- function(d, min_pts) {
  kth <- sort(apply(d, 1, function(row) sort(row, partial = min_pts)[min_pts]))
  x <- seq(0, 1, length.out = length(kth))
  y <- (kth - min(kth)) / diff(range(kth))
  kth[which.max(x - y)]
}

fit <- function(x) {
  d <- dist(x)
  distances <- as.matrix(d)
  set.seed(0)
  km <- kmeans(x, centers = 5, nstart = 25, iter.max = 300)$cluster
  ward <- cutree(hclust(d, method = "ward.D2"), k = 5)
  eps <- round(knee(distances, 10), 2)
  db <- dbscan_radius(distances, eps, 10)
  list(labels = list(Kmeans = km, Ward = ward, DBSCAN = db), eps = eps)
}

fits <- lapply(spaces, fit)
summary_rows <- list()
cross_rows <- list()
assignments <- original[c("id", "class")]
for (space in names(fits)) {
  for (algorithm in names(fits[[space]]$labels)) {
    labels <- fits[[space]]$labels[[algorithm]]
    assignments[[paste(space, algorithm, sep = "_")]] <- labels
    sizes <- table(labels[labels > 0])
    baseline <- fits$Originali$labels[[algorithm]]
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      space, algorithm, clusters = length(sizes), noise = sum(labels == 0L),
      sizes = paste(sort(as.integer(sizes), decreasing = TRUE), collapse = "/"),
      eps = if (algorithm == "DBSCAN") fits[[space]]$eps else NA_real_,
      shared_noise = if (algorithm == "DBSCAN") sum(labels == 0L & baseline == 0L) else NA_integer_
    )
    if (space != "Originali") {
      counts <- as.data.frame(table(original = baseline, reduced = labels))
      cross_rows[[length(cross_rows) + 1L]] <- cbind(space, algorithm, counts)
      print(table(original = baseline, reduced = labels))
    }
  }
}
summary <- do.call(rbind, summary_rows)
print(summary, row.names = FALSE)
write.csv(summary, file.path(out, "A25_6_suvestine.csv"), row.names = FALSE)
write.csv(assignments, file.path(out, "A25_6_priskyrimai.csv"), row.names = FALSE)
write.csv(do.call(rbind, cross_rows), file.path(out, "A25_6_sutapimai.csv"), row.names = FALSE)
print(lapply(fits, function(x) lapply(x$labels, function(z) table(z, original$class))))

# Each column shows one representation and each row shows one algorithm.
png(file.path(out, "A25_6_klasteriai.png"), width = 1800, height = 2100, res = 200, type = "cairo")
par(mfrow = c(3, 2), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))
colors <- hcl.colors(10, "Dark 3")
for (algorithm in names(fits$Originali$labels)) {
  for (space in c("PCA7", "TSNE2")) {
    labels <- fits[[space]]$labels[[algorithm]]
    coords <- spaces[[space]][, 1:2]
    point_colors <- ifelse(labels == 0L, "#515963", colors[pmax(labels, 1L)])
    xlab <- if (space == "PCA7") "PC1" else "t-SNE 1"
    ylab <- if (space == "PCA7") "PC2" else "t-SNE 2"
    plot(coords, asp = 1, xlab = xlab, ylab = ylab,
         main = paste(algorithm, if (space == "PCA7") "PCA (7D)" else "t-SNE (2D)"),
         pch = ifelse(labels == 0L, 4, 16), cex = 0.53, col = adjustcolor(point_colors, 0.75))
    if (any(labels == 0L)) legend("topright", "Triukšmas", pch = 4, col = "#515963",
                                bty = "n", cex = 0.7)
  }
}
mtext("A25 klasteriai sumažintose erdvėse", outer = TRUE, font = 2)
dev.off()
