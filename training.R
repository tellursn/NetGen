library(Rcpp)
library(igraph)
library(foreach)
library(doParallel)

# --- 設定 ---
N_SAMPLES <- 30000  # データ数
N_NODES   <- 500
CORES <- parallel::detectCores(logical = FALSE) 

cat(sprintf("使用するCPUコア数: %d\n", CORES))

# クラスタ立ち上げ
cl <- makeCluster(CORES)
registerDoParallel(cl)

clusterEvalQ(cl, {
  library(Rcpp)
  library(igraph)
  sourceCpp("gengine.cpp") # 各コアが個別にロードする
})

cat("データ収集を開始\n")

# 時間計測開始
start_time <- Sys.time()

# 並列ループ実行 (.combine=rbind で結果を行列として結合)
full_data <- foreach(i = 1:N_SAMPLES, .combine = rbind) %dopar% {
  
  # --- 1. 確率生成 ---
  # [GPA, FF, NS, MA, TRA, ADM]
  raw_probs <- runif(6) 
  probs_vec <- raw_probs / sum(raw_probs)
  
  # --- 2. 内部パラメータ (実データに合わせるなら範囲調整してください) ---
  alpha <- runif(1, 0.5, 3.0)
  beta  <- runif(1, 1.0, 5.0)
  m     <- sample(1:20, 1)
  p_burn<- runif(1, 0.0, 1.0)
  K     <- sample(1:10, 1) # TRA/MA用 (一応残す)
  adm_type <- sample(c(0, 1), 1) # 0 or 1
  
  params_vec <- c(alpha, beta, m, p_burn, K, adm_type)
  
  # --- 3. C++エンジン実行 & 指標計算 ---
  tryCatch({
    # C++関数呼び出し
    res <- netmix_core(N_NODES, probs_vec, params_vec)
    g <- graph_from_edgelist(cbind(res$from, res$to), directed = FALSE)
    g <- simplify(g)
    
    # 巨大連結成分 (Giant Component)
    comps <- components(g)
    g_giant <- induced_subgraph(g, which(comps$membership == which.max(comps$csize)))
    
    # 指標計算 
    m_deg  <- mean(degree(g))
    m_clus <- transitivity(g, type="global")
    if(is.na(m_clus)) m_clus <- 0
    m_path <- mean_distance(g_giant, directed=FALSE) # これが少し重い
    m_assort <- assortativity_degree(g)
    if(is.na(m_assort)) m_assort <- 0
    
    # モジュラリティ (Fast Greedyは比較的速い)
    comm <- cluster_fast_greedy(g_giant)
    m_mod <- modularity(comm)
    
    # 結果ベクトルを返す
    c(probs_vec, params_vec, m_deg, m_clus, m_path, m_assort, m_mod)
    
  }, error = function(e) {
    # エラー時はNAで埋める (後で削除)
    return(rep(NA, 17)) 
  })
}

# クラスタ停止 (メモリ解放)
stopCluster(cl)

# 時間計測終了
end_time <- Sys.time()
print(end_time - start_time)

# --- データ整形 ---
df <- as.data.frame(full_data)
colnames(df) <- c("p_gpa", "p_ff", "p_ns", "p_ma", "p_tra", "p_adm",
                  "alpha", "beta", "m", "p_burn", "K", "is_assort",
                  "avg_degree", "clustering", "path_len", "assort", "modularity")

# エラー行を削除
df <- na.omit(df)

# 保存
write.csv(df, "netmix_training_data.csv", row.names = FALSE)
cat("完了: netmix_training_data.csv を保存しました。\n")
