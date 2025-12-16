library(ranger)

data <- read.csv("netmix_training_data.csv")

# 入力変数のカラム名
input_cols <- c("p_gpa", "p_ff", "p_ns", "p_ma", "p_tra", "p_adm",
                "alpha", "beta", "m", "p_burn", "K", "is_assort")

# 出力変数（ターゲット）のカラム名
target_cols <- c("avg_degree", "clustering", "path_len", "assort", "modularity")

print("モデル学習を開始します...")

model_list <- list()

for (target in target_cols) {
  cat(sprintf("Training model for: %s ... ", target))
  
  # フォーミュラ作成
  fmla <- as.formula(paste(target, "~", paste(input_cols, collapse = "+")))
  
  # rangerで学習 (高速版ランダムフォレスト)
  rf_model <- ranger(fmla, data = data, num.trees = 500, importance = "impurity")
  
  model_list[[target]] <- rf_model
  cat("Done.\n")
}

# モデルリストを保存
saveRDS(model_list, "surrogate_models_ranger.rds")
print("完了: surrogate_models_ranger.rds を保存！")
