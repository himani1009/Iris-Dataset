# iris_eda_and_models.R
# Complete EDA + classification on the built-in iris dataset
# Sections:
# 1) Load libraries + data
# 2) Exploratory Data Analysis (distributions, correlations, outliers, clustering)
# 3) Supervised learning: Decision Tree, SVM, k-NN
# 4) Evaluation: accuracy, precision, recall, F1 (per class + macro)

# -----------------------------
# 1) Setup
# -----------------------------
set.seed(123)

# install.packages(c("tidyverse","GGally","caret","e1071","rpart","rpart.plot","class","kknn","pROC"))
library(tidyverse)
library(GGally)      # ggpairs for quick EDA
library(caret)       # training, resampling, confusionMatrix
library(e1071)       # SVM
library(rpart)       # decision tree
library(rpart.plot)  # plot tree
library(class)       # knn
library(kknn)        # alternative knn
library(pROC)

# load iris (built-in)
data(iris)
iris <- iris

# Quick look
glimpse(iris)
summary(iris)

# -----------------------------
# 2) EDA
# -----------------------------

# 2a) Distributions by species: density + boxplots
# Density plots for each numeric feature by Species
num_vars <- names(iris)[1:4]

for (v in num_vars) {
  p <- ggplot(iris, aes_string(x = v, fill = "Species")) +
    geom_density(alpha = 0.4) +
    labs(title = paste("Density of", v, "by Species")) +
    theme_minimal()
  print(p)
}

# Boxplots to inspect spread & outliers
for (v in num_vars) {
  p <- ggplot(iris, aes_string(x = "Species", y = v, fill = "Species")) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", v, "by Species")) +
    theme_minimal()
  print(p)
}

# Pairwise scatterplots (colored by species)
print(ggpairs(iris, columns = 1:4, mapping = ggplot2::aes(color = Species)))

# 2b) Correlations between features (numeric only)
cor_mat <- cor(iris[, 1:4])
print(cor_mat)

# Visual correlation matrix
library(reshape2)
cor_df <- melt(cor_mat)
ggplot(cor_df, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation matrix of numeric features") +
  theme_minimal()

# 2c) Outlier detection
# We'll use IQR rule per feature and show any rows flagged as containing outliers
find_outliers <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  which(x < lower | x > upper)
}

outlier_indices <- unique(unlist(lapply(iris[,1:4], find_outliers)))
cat("Outlier row indices (by IQR rule):\n")
print(outlier_indices)
if (length(outlier_indices) > 0) print(iris[outlier_indices, ])

# We can also mark outliers on boxplots (ggplot already shows them)

# 2d) Patterns/clusters: PCA and k-means clustering
# PCA
iris_scaled <- scale(iris[,1:4])
pca <- prcomp(iris_scaled)
summary(pca)

# Plot first two principal components colored by species
pca_df <- as.data.frame(pca$x)
pca_df$Species <- iris$Species
ggplot(pca_df, aes(x = PC1, y = PC2, color = Species)) +
  geom_point(size = 2) +
  labs(title = "PCA: PC1 vs PC2 colored by Species") +
  theme_minimal()

# k-means clustering (k = 3) on scaled features
set.seed(42)
kmeans3 <- kmeans(iris_scaled, centers = 3, nstart = 25)
table(kmeans3$cluster, iris$Species)

# Visualize cluster assignments on PCA space
pca_df$kmeans_cluster <- factor(kmeans3$cluster)
ggplot(pca_df, aes(x = PC1, y = PC2, color = kmeans_cluster, shape = Species)) +
  geom_point(size = 2) +
  labs(title = "k-means (k=3) clusters on PCA space") +
  theme_minimal()

# -----------------------------
# 3) Supervised Learning: Decision Tree, SVM, k-NN
# We'll split data into train/test (70/30) and evaluate models
# -----------------------------

set.seed(123)
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train <- iris[train_index, ]
test  <- iris[-train_index, ]

# Helper function to compute precision, recall, F1 from confusion matrix
compute_metrics <- function(cm) {
  # cm : caret::confusionMatrix object
  byClass <- as.data.frame(cm$byClass)
  # byClass rows correspond to classes, columns include Precision (Pos Pred Value), Recall (Sensitivity), F1 isn't direct
  precision <- byClass$`Pos Pred Value`
  recall <- byClass$Sensitivity
  f1 <- 2 * (precision * recall) / (precision + recall)
  res <- data.frame(
    Class = rownames(byClass),
    Precision = round(precision, 3),
    Recall = round(recall, 3),
    F1 = round(f1, 3)
  )
  accuracy <- cm$overall['Accuracy']
  list(per_class = res, accuracy = round(as.numeric(accuracy), 3))
}

# 3a) Decision Tree (rpart)
set.seed(123)
dt_model <- rpart(Species ~ ., data = train, method = "class")
print(dt_model)
rpart.plot(dt_model, main = "Decision Tree")

# predict
dt_pred <- predict(dt_model, test, type = "class")
dt_cm <- confusionMatrix(dt_pred, test$Species)
cat("Decision Tree Confusion Matrix:\n")
print(dt_cm$table)
metrics_dt <- compute_metrics(dt_cm)
print(metrics_dt$per_class)
cat("Accuracy:", metrics_dt$accuracy, "\n")

# 3b) SVM (radial) using e1071
set.seed(123)
svm_model <- e1071::svm(Species ~ ., data = train, kernel = "radial", probability = TRUE)
svm_pred <- predict(svm_model, test)
svm_cm <- confusionMatrix(svm_pred, test$Species)
cat("SVM Confusion Matrix:\n")
print(svm_cm$table)
metrics_svm <- compute_metrics(svm_cm)
print(metrics_svm$per_class)
cat("Accuracy:", metrics_svm$accuracy, "\n")

# 3c) k-NN (using caret to tune k)
set.seed(123)
ctrl <- trainControl(method = "cv", number = 5)
knn_fit <- train(Species ~ ., data = train, method = "knn", tuneLength = 10, trControl = ctrl)
print(knn_fit)

knn_pred <- predict(knn_fit, test)
knn_cm <- confusionMatrix(knn_pred, test$Species)
cat("k-NN Confusion Matrix:\n")
print(knn_cm$table)
metrics_knn <- compute_metrics(knn_cm)
print(metrics_knn$per_class)
cat("Accuracy:", metrics_knn$accuracy, "\n")

# -----------------------------
# 4) Summary comparison
# -----------------------------
results_summary <- tibble(
  Model = c("Decision Tree", "SVM (radial)", "k-NN"),
  Accuracy = c(metrics_dt$accuracy, metrics_svm$accuracy, metrics_knn$accuracy)
)
print(results_summary)

# For readability, show macro-averaged Precision, Recall, F1
macro <- function(per_class_df) {
  p <- mean(per_class_df$Precision, na.rm = TRUE)
  r <- mean(per_class_df$Recall, na.rm = TRUE)
  f <- mean(per_class_df$F1, na.rm = TRUE)
  round(c(Precision = p, Recall = r, F1 = f), 3)
}

results_summary <- results_summary %>%
  mutate(
    Precision = c(macro(metrics_dt$per_class), macro(metrics_svm$per_class), macro(metrics_knn$per_class))[,1],
    Recall    = c(macro(metrics_dt$per_class), macro(metrics_svm$per_class), macro(metrics_knn$per_class))[,2],
    F1        = c(macro(metrics_dt$per_class), macro(metrics_svm$per_class), macro(metrics_knn$per_class))[,3]
  )
print(results_summary)

# Nicely print per-model per-class tables
cat("\nPer-model per-class metrics:\n")
cat("--- Decision Tree ---\n")
print(metrics_dt$per_class)
cat("--- SVM ---\n")
print(metrics_svm$per_class)
cat("--- k-NN ---\n")
print(metrics_knn$per_class)

# End of script

# Notes/Observations (to help you interpret results):
# - Expect petal length and petal width to be the most discriminative features (high correlation with species separation).
# - Petal measurements usually separate Setosa cleanly from Versicolor & Virginica; Versicolor and Virginica can overlap.
# - k-means often recovers clusters that align with species quite well but may swap labels.
# - SVM and k-NN often perform very well on iris; decision tree is interpretable and often near-competitive.

# If you want: save results to CSV or create plots to files. For example:
# ggsave("pca_plot.png")
# write_csv(results_summary, "model_summary.csv")
