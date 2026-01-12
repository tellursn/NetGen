# ---------------------------------------------------------
# training.R (Serial version with fractional m)
# ---------------------------------------------------------
library(Rcpp)
library(igraph)
library(tictoc)
tic()
# C++エンジンを読み込む
# 必ず最新の v5 (Smart-TRA & Fractional-m) を保存してから実行してください
sourceCpp("gengine.cpp")

# --- 設定 ---
N_SAMPLES <- 10000  # 学習データ数
N_NODES   <- 500    # グラフサイズ

# --- 結果格納用データフレーム ---
# 入力パラメタ (X)
df_params <- data.frame(
  p_gpa = numeric(N_SAMPLES),
  p_ff  = numeric(N_SAMPLES),
  p_ns  = numeric(N_SAMPLES),
  p_ma  = numeric(N_SAMPLES),
  p_tra = numeric(N_SAMPLES),
  p_adm = numeric(N_SAMPLES),
  alpha = numeric(N_SAMPLES),
  beta  = numeric(N_SAMPLES),
  m     = numeric(N_SAMPLES),
  p_burn= numeric(N_SAMPLES),
  K     = numeric(N_SAMPLES),
  is_assort = numeric(N_SAMPLES)
)

# 出力特徴量 (Y)
df_metrics <- data.frame(
  avg_degree = numeric(N_SAMPLES), # 平均次数
  clustering = numeric(N_SAMPLES), # クラスター係数
  path_len   = numeric(N_SAMPLES), # 平均経路長
  assort     = numeric(N_SAMPLES), # 次数相関
  modularity = numeric(N_SAMPLES)  # モジュラリティ
)

cat(sprintf("データ収集を開始します (Samples: %d, Nodes: %d)...\n", N_SAMPLES, N_NODES))
start_time <- Sys.time()

for (i in 1:N_SAMPLES) {
  
  # 1. パラメタをランダム生成
  # 確率 (ランダム値を正規化して合計1にする)
  raw_probs <- runif(6)
  probs <- raw_probs / sum(raw_probs)
  
  # 内部パラメタ
  alpha <- runif(1, 0.5, 3.0)   # Non-linear度
  beta  <- runif(1, 1.0, 5.0)   # Threshold形状
  
  # ★修正箇所: m を整数(sample)から小数(runif)に変更★
  # これにより、平均次数 5.8 などを正確に狙えるようになります
  m     <- runif(1, 1.0, 20.0) 
  
  p_burn<- runif(1, 0.0, 1.0)   # FF燃焼率
  K     <- sample(1:10, 1)      # TRA近傍数など
  adm_type <- sample(c(0, 1), 1) # 0: Disassort, 1: Assort
  
  params_vec <- c(alpha, beta, m, p_burn, K, adm_type)
  
  # 2. C++エンジン実行
  tryCatch({
    res <- netmix_core_directed(N_NODES, probs, params_vec)
    g <- graph_from_edgelist(cbind(res$from, res$to), directed = FALSE) 
    
    # 単純化 (多重エッジ・自己ループ削除)
    g <- simplify(g)
    
    # 3. 特徴量計算
    
    # 連結成分の抽出 (最大成分のみ使う)
    comps <- components(g)
    giant_id <- which.max(comps$csize)
    g_giant <- induced_subgraph(g, which(comps$membership == giant_id))
    
    # 指標計算
    m_deg  <- mean(degree(g))
    m_clus <- transitivity(g, type="global")
    if(is.na(m_clus)) m_clus <- 0
    m_path <- mean_distance(g_giant, directed=FALSE)
    m_assort <- assortativity_degree(g)
    if(is.na(m_assort)) m_assort <- 0
    
    # モジュラリティ (高速なGreedy法)
    comm <- cluster_fast_greedy(g_giant)
    m_mod <- modularity(comm)
    
    # 4. 保存
    df_params[i, ] <- c(probs, alpha, beta, m, p_burn, K, adm_type)
    df_metrics[i,] <- c(m_deg, m_clus, m_path, m_assort, m_mod)
    
  }, error = function(e) {
    # エラー時はNA埋め (後で削除される)
    cat(sprintf("\nError at iter %d: %s", i, e$message))
    df_metrics[i,] <- NA
  })
  
  # 進捗表示 (500回ごと)
  if (i %% 500 == 0) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    per_iter <- elapsed / i
    remain <- (N_SAMPLES - i) * per_iter
    cat(sprintf("Progress: %d / %d (%.1f%%) - Elapsed: %.1f min - Remain: %.1f min\n", 
                i, N_SAMPLES, (i/N_SAMPLES)*100, elapsed, remain))
  }
}

# 結合して保存
full_data <- cbind(df_params, df_metrics)
# NAを含む行（失敗分）を削除
original_rows <- nrow(full_data)
full_data <- na.omit(full_data) 
valid_rows <- nrow(full_data)

cat(sprintf("\n完了: %d 件中 %d 件が有効でした。\n", original_rows, valid_rows))

if(valid_rows > 0) {
  write.csv(full_data, "netmix_training_data.csv", row.names = FALSE)
  print("netmix_training_data.csv を保存しました。")
} else {
  print("エラー: 有効なデータが1件もありませんでした。")
}
toc()
