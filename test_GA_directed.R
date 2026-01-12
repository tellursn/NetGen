library(Rcpp)
library(igraph)
library(ranger)
library(GA)
library(tictoc)

tic()

sourceCpp("gengine_directed.cpp")

# optimizerは代理モデル(RandomForest)を使うだけなので、
# "optimizer_GA.R" の中身を変更していなくても、
# 新しいデータで学習した "surrogate_models_ranger.rds" さえあれば動きます。
if(file.exists("optimizer_GA.R")) {
  source("optimizer_GA.R") 
} else {
  source("optimizer.R") 
}

# 2. ターゲット設定 (E. coli 等)
# ※有向グラフとして計測されたターゲット値であることを確認してください
target_metrics <- c(
  avg_degree = 5.8,
  clustering = 0.15,
  path_len   = 2.9,
  assort     = -0.24,
  modularity = 0.43
)

# 検証回数
N_VALIDATION_TRIALS <- 1000
N_NODES <- 500

cat("=== 目標とする特徴量 (Directed) ===\n")
print(target_metrics)
cat("\n")

# ---------------------------------------------------------
# 3. GA最適化実行
# ---------------------------------------------------------
opt_res <- run_netmix_optimization(target_metrics)

cat("\n最適化完了。スコア(誤差):", opt_res$optimization_score, "\n")
cat("推奨レシピ:\n")
print(round(opt_res$probs, 3))
cat("内部パラメタ: m =", opt_res$params$m, ", K =", opt_res$params$K, 
    ", Assort =", opt_res$params$is_assort, "\n\n")

# ---------------------------------------------------------
# 4. 検証 (Validation) - 有向版
# ---------------------------------------------------------
cat(sprintf("推奨レシピで %d 回の検証を開始します (有向・直列処理)...\n", N_VALIDATION_TRIALS))

best_probs  <- opt_res$cpp_probs
best_params <- opt_res$cpp_params

results_mat <- matrix(NA, nrow = N_VALIDATION_TRIALS, ncol = 5)
colnames(results_mat) <- names(target_metrics)

start_time <- Sys.time()
pb <- txtProgressBar(min = 0, max = N_VALIDATION_TRIALS, style = 3)

for (i in 1:N_VALIDATION_TRIALS) {
  tryCatch({
    # ★変更: 有向版エンジン呼び出し
    gen_res <- netmix_core_directed(N_NODES, best_probs, best_params)
    
    # ★変更: directed = TRUE
    g <- graph_from_edgelist(cbind(gen_res$from, gen_res$to), directed = TRUE)
    g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)
    
    # --- 指標計算 (有向対応) ---
    
    # 巨大弱連結成分 (到達可能性を見るため)
    comps <- components(g, mode = "weak")
    g_giant <- induced_subgraph(g, which(comps$membership == which.max(comps$csize)))
    
    # 1. 平均次数 (★変更: エッジ数/ノード数)
    # directedの場合、mean(degree(g))は入次数+出次数(2k)になることが多いので
    # 一般的な「平均次数(k)」に合わせるため ecount/vcount を使用
    d_val <- ecount(g) / vcount(g)
    
    # 2. クラスタリング係数
    # 有向グラフの厳密な定義は難しいが、igraphのtransitivity(global)は
    # 向きを無視して三角形を数えることが多い（モチーフ解析以外では一般的）
    c_val <- transitivity(g, type="global")
    if(is.na(c_val)) c_val <- 0
    
    # 3. 平均経路長 (★変更: directed=TRUE)
    # 到達できないペアは無視されるかInfになるが、mean_distanceは到達可能ペアのみで計算
    p_val <- mean_distance(g_giant, directed=TRUE)
    
    # 4. 次数相関 (★変更: directed=TRUE)
    a_val <- assortativity_degree(g, directed=TRUE)
    if(is.na(a_val)) a_val <- 0
    
    # 5. モジュラリティ (★変更: 無向化して計算)
    # fast_greedy は有向グラフに対応していないため、無向とみなして計算
    comm <- cluster_fast_greedy(as.undirected(g_giant, mode="collapse"))
    m_val <- modularity(comm)
    
    results_mat[i, ] <- c(d_val, c_val, p_val, a_val, m_val)
    
  }, error = function(e) {
    # エラー時は何もしない
  })
  
  setTxtProgressBar(pb, i)
}

close(pb)
print(Sys.time() - start_time)

# ---------------------------------------------------------
# 5. 結果表示
# ---------------------------------------------------------
results_df <- as.data.frame(results_mat)
results_df <- na.omit(results_df)

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

cat("\nサンプルネットワークを描画します...\n")
# 描画時は矢印が見えやすいように調整
plot(g, vertex.size=3, vertex.label=NA, edge.arrow.size=0.2, 
     layout=layout_with_fr, main="Optimized Directed Sample")

toc()
