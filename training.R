library(Rcpp)
library(igraph)

sourceCpp("gengine.cpp")

# --- 設定 ---
N_SAMPLES <- 30000  # 学習データ数
N_NODES   <- 500    # グラフサイズ（固定）

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

print("データ収集を開始します...")

for (i in 1:N_SAMPLES) {
  
  # 1. パラメタをランダム生成
  # 確率 (ランダム値を正規化して合計1にする)
  raw_probs <- runif(6)
  probs <- raw_probs / sum(raw_probs)
  
  # 内部パラメタ (探索範囲を想定してランダムに振る)
  alpha <- runif(1, 0.5, 3.0)   # Non-linear度
  beta  <- runif(1, 1.0, 5.0)   # Threshold形状
  m     <- sample(1:20, 1)  # 最大10本まで増やす (密度が高いとクラスター化しやすい)
  K     <- sample(1:10, 1)  # 近傍も広げる
  #m     <- sample(1:5, 1)       # 接続本数
  p_burn<- runif(1, 0.0, 1.0)   # FF燃焼率
  #K     <- sample(1:5, 1)       # TRA近傍数
  adm_type <- sample(c(0, 1), 1) # 0: Disassort, 1: Assort
  
  params_vec <- c(alpha, beta, m, p_burn, K, adm_type)
  
  # 2. C++エンジン実行
  tryCatch({
    res <- netmix_core(N_NODES, probs, params_vec)
    g <- graph_from_edgelist(cbind(res$from, res$to), directed = FALSE)
    
    # 【追加】単純化
    g <- simplify(g)
    # 3. 特徴量計算 (ここがRでの計算ボトルネック)
    
    # 連結成分の抽出 (経路長計算のため最大成分のみ使う)
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
    # エラー時はNA埋め
    df_metrics[i,] <- NA
  })
  
  if (i %% 500 == 0) cat(sprintf("Progress: %d / %d\n", i, N_SAMPLES))
}

# 結合して保存
full_data <- cbind(df_params, df_metrics)
full_data <- na.omit(full_data) # 失敗分を削除

write.csv(full_data, "netmix_training_data.csv", row.names = FALSE)
print("完了: netmix_training_data.csv を保存しました。")
```
</details>

<details><summary>ab_training.R</summary>

```r
# ---------------------------------------------------------
# step2_ablation.R
# FF(Forest Fire)とNS(Neighbor Strength)を無効化したデータ収集
# ---------------------------------------------------------
library(Rcpp)
library(igraph)

# 修正版gengine.cppを読み込み
sourceCpp("gengine.cpp")

# --- 設定 ---
N_SAMPLES <- 30000  # 比較用なので1万回で十分傾向は見えます
N_NODES   <- 500

# --- 結果格納用データフレーム ---
# カラム構成は変えず、p_ffとp_nsを常に0にする
df_params <- data.frame(
  p_gpa = numeric(N_SAMPLES),
  p_ff  = numeric(N_SAMPLES), # 常に0
  p_ns  = numeric(N_SAMPLES), # 常に0
  p_ma  = numeric(N_SAMPLES),
  p_tra = numeric(N_SAMPLES),
  p_adm = numeric(N_SAMPLES),
  alpha = numeric(N_SAMPLES),
  beta  = numeric(N_SAMPLES),
  m     = numeric(N_SAMPLES),
  p_burn= numeric(N_SAMPLES), # FFが無効なら意味を持たないが形式上残す
  K     = numeric(N_SAMPLES),
  is_assort = numeric(N_SAMPLES)
)

df_metrics <- data.frame(
  avg_degree = numeric(N_SAMPLES),
  clustering = numeric(N_SAMPLES),
  path_len   = numeric(N_SAMPLES),
  assort     = numeric(N_SAMPLES),
  modularity = numeric(N_SAMPLES)
)

print("アブレーション用データ収集を開始します...")

for (i in 1:N_SAMPLES) {
  
  # 1. 確率生成 (4プロセスのみ)
  # GPA, MA, TRA, ADM の4つに乱数を振る
  raw_probs <- runif(4) 
  norm_probs <- raw_probs / sum(raw_probs)
  
  # 全体ベクトルにマッピング (FF=0, NS=0)
  # 順序: [GPA, FF, NS, MA, TRA, ADM]
  probs_vec <- c(
    norm_probs[1], # GPA
    0.0,           # FF (無効化)
    0.0,           # NS (無効化)
    norm_probs[2], # MA
    norm_probs[3], # TRA
    norm_probs[4]  # ADM
  )
  
  # 2. 内部パラメタ (範囲はフルモデルと同じにする)
  alpha <- runif(1, 0.5, 3.0)
  beta  <- runif(1, 1.0, 5.0)
  m     <- sample(1:20, 1)
  p_burn<- runif(1, 0.0, 1.0) # FF無効なら使われない
  K     <- sample(1:10, 1)
  adm_type <- sample(c(0, 1), 1)
  
  params_vec <- c(alpha, beta, m, p_burn, K, adm_type)
  
  # 3. C++エンジン実行
  tryCatch({
    res <- netmix_core(N_NODES, probs_vec, params_vec)
    g <- graph_from_edgelist(cbind(res$from, res$to), directed = FALSE)
    
    # 単純化 (多重エッジ削除)
    g <- simplify(g)
    
    # 特徴量計算
    comps <- components(g)
    giant_id <- which.max(comps$csize)
    g_giant <- induced_subgraph(g, which(comps$membership == giant_id))
    
    m_deg  <- mean(degree(g))
    m_clus <- transitivity(g, type="global")
    if(is.na(m_clus)) m_clus <- 0
    m_path <- mean_distance(g_giant, directed=FALSE)
    m_assort <- assortativity_degree(g)
    if(is.na(m_assort)) m_assort <- 0
    
    comm <- cluster_fast_greedy(g_giant)
    m_mod <- modularity(comm)
    
    # 保存
    df_params[i, ] <- c(probs_vec, alpha, beta, m, p_burn, K, adm_type)
    df_metrics[i,] <- c(m_deg, m_clus, m_path, m_assort, m_mod)
    
  }, error = function(e) {
    df_metrics[i,] <- NA
  })
  
  if (i %% 1000 == 0) cat(sprintf("Progress: %d / %d\n", i, N_SAMPLES))
}

full_data <- cbind(df_params, df_metrics)
full_data <- na.omit(full_data)
# 別名で保存
write.csv(full_data, "netmix_training_data_4proc.csv", row.names = FALSE)
print("完了: netmix_training_data_4proc.csv を保存しました。")
