# ---------------------------------------------------------
# test_all_metrics_serial.R (GA & Serial Validation)
# 並列化なし・安定動作版
# ---------------------------------------------------------
library(Rcpp)
library(igraph)
library(ranger)
library(GA)

# 1. ファイル読み込み
sourceCpp("gengine.cpp")
#if(file.exists("optimizer_GA.R")) {
    source("optimizer_GA.R") 
#} else {
  # ファイル名が optimizer.R のままの場合はこちら
  #source("optimizer.R") 
#}

# 2. ターゲット設定 (E. coli の例)
target_metrics <- c(
  avg_degree = 5.8,
  clustering = 0.15,
  path_len   = 2.9,
  assort     = -0.24,
  modularity = 0.43
)

# 検証回数
N_VALIDATION_TRIALS <- 100
N_NODES <- 500

cat("=== 目標とする特徴量 ===\n")
print(target_metrics)
cat("\n")

# ---------------------------------------------------------
# 3. GA最適化実行
# ---------------------------------------------------------
# optimizer側は並列化(parallel=TRUE)していても、GAパッケージの内部処理なので
# 比較的安全ですが、もしここでも固まる場合は optimizer.R 内の parallel=FALSE にしてください。
opt_res <- run_netmix_optimization(target_metrics)

cat("\n最適化完了。スコア(誤差):", opt_res$optimization_score, "\n")
cat("推奨レシピ:\n")
print(round(opt_res$probs, 3))
cat("内部パラメタ: m =", opt_res$params$m, ", K =", opt_res$params$K, 
    ", Assort =", opt_res$params$is_assort, "\n\n")

# ---------------------------------------------------------
# 4. 検証 (Validation) - シングルスレッド版
# ---------------------------------------------------------
cat(sprintf("推奨レシピで %d 回の検証を開始します (直列処理)...\n", N_VALIDATION_TRIALS))

best_probs  <- opt_res$cpp_probs
best_params <- opt_res$cpp_params

# 結果格納用のマトリクス (事前割り当てで高速化)
results_mat <- matrix(NA, nrow = N_VALIDATION_TRIALS, ncol = 5)
colnames(results_mat) <- names(target_metrics)

# 時間計測開始
start_time <- Sys.time()

# プログレスバー作成
pb <- txtProgressBar(min = 0, max = N_VALIDATION_TRIALS, style = 3)

for (i in 1:N_VALIDATION_TRIALS) {
  tryCatch({
    # 生成
    gen_res <- netmix_core(N_NODES, best_probs, best_params)
    g <- graph_from_edgelist(cbind(gen_res$from, gen_res$to), directed = FALSE)
    g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)
    
    # --- 指標計算 ---
    # 巨大連結成分
    comps <- components(g)
    g_giant <- induced_subgraph(g, which(comps$membership == which.max(comps$csize)))
    
    # 1. 平均次数
    d_val <- mean(degree(g))
    # 2. クラスタリング
    c_val <- transitivity(g, type="global")
    if(is.na(c_val)) c_val <- 0
    # 3. 平均経路長
    p_val <- mean_distance(g_giant, directed=FALSE)
    # 4. 次数相関
    a_val <- assortativity_degree(g)
    if(is.na(a_val)) a_val <- 0
    # 5. モジュラリティ
    comm <- cluster_fast_greedy(g_giant)
    m_val <- modularity(comm)
    
    # 結果を行に格納
    results_mat[i, ] <- c(d_val, c_val, p_val, a_val, m_val)
    
  }, error = function(e) {
    # エラー時はNAのままにしておく(何もしない)
  })
  
  # プログレスバー更新
  setTxtProgressBar(pb, i)
}

close(pb)
print(Sys.time() - start_time)

# ---------------------------------------------------------
# 5. 結果表示
# ---------------------------------------------------------
results_df <- as.data.frame(results_mat)
results_df <- na.omit(results_df) # エラー行を除外

res_mean <- colMeans(results_df)
res_sd   <- apply(results_df, 2, sd)

cat("\n=== 検証結果 (平均 ± 標準偏差) ===\n")
summary_table <- data.frame(
  Metric = names(target_metrics),
  Target = as.numeric(target_metrics),
  Actual_Mean = round(res_mean, 4),
  Std_Dev = round(res_sd, 4),
  Error_Mean = round(abs(res_mean - target_metrics), 4)
)
print(summary_table)

# 最後に1つ可視化
cat("\nサンプルネットワークを描画します...\n")
plot(g, vertex.size=3, vertex.label=NA, layout=layout_with_fr, main="Optimized Sample")
