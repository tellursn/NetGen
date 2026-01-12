# ---------------------------------------------------------
# optimizer.R (GA Serial version)
# 並列化なし
# ---------------------------------------------------------
library(GA)
library(ranger)

# --- 1. 代理モデルの読み込み ---
if (!exists("surrogate_models")) {
  if (file.exists("surrogate_models_ranger.rds")) {
    surrogate_models <- readRDS("surrogate_models_ranger.rds")
  } else {
    stop("エラー: 'surrogate_models_ranger.rds' が見つかりません。step3を実行してください。")
  }
}

# --- 2. 最適化実行関数 ---
run_netmix_optimization <- function(target_metrics) {
  
  # --- 遺伝子の定義 ---
  # [確率ウェイトx6, alpha, beta, m, K, is_assort] 計11変数
  lower_bound <- c(rep(0, 6),  0.5, 1.0, 1,  1,  0)
  upper_bound <- c(rep(10, 6), 3.0, 5.0, 20, 10, 1)
  
  # 変数名
  names(lower_bound) <- c("w_gpa", "w_ff", "w_ns", "w_ma", "w_tra", "w_adm",
                          "alpha", "beta", "m", "K", "is_assort")
  names(upper_bound) <- names(lower_bound)
  
  # --- 適応度関数 (Fitness Function) ---
  fitness_func <- function(x) {
    
    # 確率の正規化
    w_probs <- x[1:6]
    if(sum(w_probs) < 1e-9) return(-1e9)
    norm_probs <- w_probs / sum(w_probs)
    
    # パラメータ (整数化)
    alpha <- x[7]
    beta  <- x[8]
    m     <- (x[9])
    K     <- round(x[10])
    is_assort <- round(x[11])
    
    # 入力データ
    input_df <- data.frame(
      p_gpa = norm_probs[1], p_ff = norm_probs[2], p_ns = norm_probs[3],
      p_ma  = norm_probs[4], p_tra = norm_probs[5], p_adm = norm_probs[6],
      alpha = alpha, beta = beta, m = m, p_burn = 0.1, 
      K = K, is_assort = is_assort
    )
    
    # 誤差計算
    total_error <- 0
    valid_metrics <- 0
    
    for (metric in names(target_metrics)) {
      model <- surrogate_models[[metric]]
      if (!is.null(model)) {
        pred <- predict(model, data = input_df)$predictions
        
        # 重み付け (オプション: 難しい指標を重視)
        weight <- 1.0
        if(metric == "clustering") weight <- 2.0 
        if(metric == "assort") weight <- 2.0
        
        diff <- pred - target_metrics[[metric]]
        total_error <- total_error + (diff^2) * weight
        valid_metrics <- valid_metrics + 1
      }
    }
    
    if (valid_metrics == 0) return(-1e9)
    return(-total_error) # GAは最大化を目指すのでマイナスにする
  }
  
  # --- GA実行 (直列処理) ---
  cat("GAによる最適化を開始します... (直列処理)\n")
  
  ga_res <- ga(
    type = "real-valued",
    fitness = fitness_func,
    lower = lower_bound,
    upper = upper_bound,
    popSize = 200,    # 個体数
    maxiter = 500,    # 世代数
    run = 50,         # 早期終了判定
    parallel = FALSE, # ★ここをFALSEに変更しました★
    monitor = TRUE,
    optim = TRUE,     # 局所探索あり
    seed = 123
  )
  
  # --- 結果取得 ---
  best_sol <- ga_res@solution[1, ]
  best_score <- -ga_res@fitnessValue
  
  w_best <- best_sol[1:6]
  final_probs <- w_best / sum(w_best)
  names(final_probs) <- c("GPA", "FF", "NS", "MA", "TRA", "ADM")
  
  final_params <- list(
    alpha = best_sol[7],
    beta  = best_sol[8],
    m     = round(best_sol[9]),
    K     = round(best_sol[10]),
    is_assort = round(best_sol[11])
  )
  
  cpp_probs <- as.numeric(final_probs)
  cpp_params <- c(final_params$alpha, final_params$beta, final_params$m, 0.1, final_params$K, final_params$is_assort)
  
  return(list(
    optimization_score = best_score,
    probs = final_probs,
    params = final_params,
    cpp_probs = cpp_probs,
    cpp_params = cpp_params
  ))
}
