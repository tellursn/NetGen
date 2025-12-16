#include <Rcpp.h>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace Rcpp;

// 乱数生成
std::random_device rd;
std::mt19937 gen(rd());

// -------------------------------------------------------------------------
// ヘルパー関数群
// -------------------------------------------------------------------------

// 1. GPA (Non-linear + Threshold)
// Rejection Sampling
int select_generalized_pa(int n_current, const std::vector<int>& degrees, 
                          const std::vector<int>& capacity, 
                          double alpha, const std::vector<int>& degree_bag) {
    if (degree_bag.empty()) return 0;
    
    int max_trials = 10;
    for (int t = 0; t < max_trials; ++t) {
        // Linear PA
        std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
        int cand = degree_bag[bag_dis(gen)];
        
        // Threshold PA
        if (degrees[cand] >= capacity[cand]) continue;
        
        // Non-linear PA
        // alpha=1なら常にok．alpha!=1なら k^alpha に基づいて確率的に受諾
        if (std::abs(alpha - 1.0) > 0.01) {
            double p_accept = std::pow((double)degrees[cand], alpha - 1.0);
            // 簡易正規化 
            // 簡易的に確率判定
            if (std::uniform_real_distribution<>(0.0, 1.0)(gen) > p_accept) continue; 
        }
        
        return cand;
    }
    // 決まらない場合はランダム
    std::uniform_int_distribution<> rand_dis(0, n_current - 1);
    return rand_dis(gen);
}

// 2. FF
void process_forest_fire(int new_node, double p_burn, 
                         std::vector<std::vector<int>>& adj, 
                         std::vector<int>& from, std::vector<int>& to, 
                         std::vector<int>& degrees) {
    int n_current = new_node; 
    std::uniform_int_distribution<> dis(0, n_current - 1);
    int ambassador = dis(gen); // 大使を選択

    // 接続
    adj[new_node].push_back(ambassador);
    adj[ambassador].push_back(new_node);
    from.push_back(new_node + 1); to.push_back(ambassador + 1);
    degrees[new_node]++; degrees[ambassador]++;

    // 燃え広がり (簡易再帰なし版: 1ホップ隣人のみ)
    // 完全な再帰は計算コストが高いため，1階層探索
    if (adj[ambassador].empty()) return;
    
    std::geometric_distribution<> geom(1.0 - p_burn);
    int n_spread = geom(gen);
    
    if (n_spread > 0) {
        std::vector<int> targets = adj[ambassador];
        std::shuffle(targets.begin(), targets.end(), gen);
        int count = 0;
        for(int t : targets) {
            if (t == new_node) continue;
            // 接続
            adj[new_node].push_back(t);
            adj[t].push_back(new_node);
            from.push_back(new_node + 1); to.push_back(t + 1);
            degrees[new_node]++; degrees[t]++;
            
            count++;
            if(count >= n_spread) break;
        }
    }
}

// 3. Neighbor Strength
int select_neighbor_strength(const std::vector<std::vector<int>>& adj, 
                             const std::vector<int>& degree_bag) {
    if (degree_bag.empty()) return 0;
    // Step 1: PAでハブを選ぶ
    std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
    int hub = degree_bag[bag_dis(gen)];
    
    // Step 2: ハブの隣人から選ぶ
    if (adj[hub].empty()) return hub;
    std::uniform_int_distribution<> neigh_dis(0, adj[hub].size() - 1);
    return adj[hub][neigh_dis(gen)];
}

// 4. Modular Attachment 
int select_modular(int n_current, const std::vector<std::vector<int>>& adj) {
    // ランダムなプロトタイプを選ぶ
    std::uniform_int_distribution<> dis(0, n_current - 1);
    int proto = dis(gen);
    
    // プロトタイプの隣人から選ぶ (コピー)
    if (adj[proto].empty()) return proto;
    std::uniform_int_distribution<> neigh_dis(0, adj[proto].size() - 1);
    return adj[proto][neigh_dis(gen)];
}

// 5. TRA 
// 直近K個のノードと接続しようとする
int select_tra(int new_node, int K) {
    int target = new_node - 1 - (gen() % K); 
    if (target < 0) target = 0;
    return target;
}

// 6. ADM (Assortative)
// 次数が「自分に近い(Assort)」または「遠い(Disassort)」ノードを選ぶ
// new_nodeの次数はまだ低いので、Assortなら低次数、Disassortなら高次数(ハブ)を狙う
int select_adm(int n_current, const std::vector<int>& degrees, bool assortative) {
    // 簡易実装: ランダムに数個サンプリングして、条件に合うものを選ぶ
    int best_cand = -1;
    int best_score = assortative ? 100000 : -1;
    
    for(int i=0; i<5; ++i) { // 5回試行
        std::uniform_int_distribution<> dis(0, n_current - 1);
        int cand = dis(gen);
        int deg = degrees[cand];
        
        if (assortative) {
            // 次数が小さい方を選ぶ (Assortative: 低次数同士)
            if (deg < best_score) { best_score = deg; best_cand = cand; }
        } else {
            // 次数が大きい方を選ぶ (Disassortative: 低-高)
            if (deg > best_score) { best_score = deg; best_cand = cand; }
        }
    }
    return best_cand;
}


// -------------------------------------------------------------------------
// メイン関数
// -------------------------------------------------------------------------
// [[Rcpp::export]]
List netmix_core(int n_target, NumericVector probs, NumericVector params) {
    
    // --- パラメータの展開 ---
    // probs: [p_GPA, p_FF, p_NS, p_MA, p_TRA, p_ADM] (合計1)
    // params: [alpha(GPA), beta(Thres), m(Conn), p_burn(FF), K(TRA), adm_type(0/1)]
    
    double p_gpa = probs[0];
    double p_ff  = probs[1];
    double p_ns  = probs[2];
    double p_ma  = probs[3];
    double p_tra = probs[4];
    // 残りが ADM
    
    double alpha   = params[0];
    double beta    = params[1];
    int    m       = (int)std::round(params[2]); if(m<1) m=1;
    double p_burn  = params[3];
    int    K       = (int)std::round(params[4]); if(K<1) K=1;
    bool   is_assort = (params[5] > 0.5); // 0.5より大きければ同類、小さければ異類

    // --- データ構造初期化 ---
    std::vector<std::vector<int>> adj(n_target);
    std::vector<int> degrees(n_target, 0);
    std::vector<int> capacity(n_target);
    std::vector<int> degree_bag; // Linear PA用のサンプリングバッグ
    
    // igraph用エッジリスト (1-based index)
    std::vector<int> from, to;

    // 初期状態: 2ノード完全グラフ
    adj[0].push_back(1); adj[1].push_back(0);
    from.push_back(1); to.push_back(2);
    degrees[0]=1; degrees[1]=1;
    degree_bag.push_back(0); degree_bag.push_back(1);

    // 定員(Capacity)の設定: パレート分布
    std::uniform_real_distribution<> runif(0.0, 1.0);
    for(int i=0; i<n_target; ++i) {
        double u = runif(gen);
        // betaが大きいほど定員の格差が開く
        double cap = (double)m / std::pow(u, 1.0/beta); 
        if (cap > n_target) cap = n_target;
        capacity[i] = (int)cap;
    }

    // --- 生成ループ ---
    for (int i = 2; i < n_target; ++i) {
        
        // 今回のノードがどのプロセスを使うか決定
        double dice = runif(gen);
        double cum = 0.0;
        
        // Forest Fireは特殊（複数エッジを張る可能性がある）なので分岐
        cum += p_ff;
        if (dice < cum) {
            process_forest_fire(i, p_burn, adj, from, to, degrees);
            // FFで増えた次数分をbagに追加
            degree_bag.push_back(i); 
            // (厳密には接続先も追加すべきだが高速化のため省略可、あるいは以下で追加)
            continue; 
        }

        // それ以外のプロセス（m本のエッジを張る）
        for (int k = 0; k < m; ++k) {
            int target = -1;
            
            // ルーレット選択の続き
            // cumは維持されている
            
            // Generalized PA
            if (target == -1) {
                cum += p_gpa;
                if (dice < cum) target = select_generalized_pa(i, degrees, capacity, alpha, degree_bag);
            }
            
            // Neighbor Strength
            if (target == -1) {
                cum += p_ns;
                if (dice < cum) target = select_neighbor_strength(adj, degree_bag);
            }
            
            // Modular Attachment
            if (target == -1) {
                cum += p_ma;
                if (dice < cum) target = select_modular(i, adj);
            }
            
            // TRA
            if (target == -1) {
                cum += p_tra;
                if (dice < cum) target = select_tra(i, K);
            }
            
            // ADM (Fallback含む)
            if (target == -1) {
                target = select_adm(i, degrees, is_assort);
            }

            // 自己ループ・重複回避 (簡易)
            if (target == i) target = 0; 
            
            // 接続処理
            adj[i].push_back(target);
            adj[target].push_back(i);
            from.push_back(i + 1); to.push_back(target + 1);
            degrees[i]++; degrees[target]++;
            
            degree_bag.push_back(i);
            degree_bag.push_back(target);
        }
    }

    return List::create(Named("from") = from, Named("to") = to);
}
