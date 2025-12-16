// 20251211
#include <Rcpp.h>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>

using namespace Rcpp;

// 乱数生成器
std::random_device rd;
std::mt19937 gen(rd());

// --- ヘルパー関数群 ---

// 1. Generalized PA
int select_generalized_pa(int n_current, const std::vector<int>& degrees, 
                          const std::vector<int>& capacity, 
                          double alpha, const std::vector<int>& degree_bag) {
    if (degree_bag.empty()) return 0;
    
    int max_trials = 10;
    for (int t = 0; t < max_trials; ++t) {
        std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
        int cand = degree_bag[bag_dis(gen)];
        
        if (degrees[cand] >= capacity[cand]) continue;
        
        if (std::abs(alpha - 1.0) > 0.01) {
            double p_accept = std::pow((double)degrees[cand], alpha - 1.0);
            if (std::uniform_real_distribution<>(0.0, 1.0)(gen) > p_accept) continue; 
        }
        return cand;
    }
    std::uniform_int_distribution<> rand_dis(0, n_current - 1);
    return rand_dis(gen);
}

// 2. Forest Fire
void process_forest_fire(int new_node, double p_burn, 
                         std::vector<std::vector<int>>& adj, 
                         std::vector<int>& degrees,
                         std::vector<int>& degree_bag) {
    int n_current = new_node; 
    std::uniform_int_distribution<> dis(0, n_current - 1);
    int ambassador = dis(gen); 

    // FFは複数エッジを張るため、ここでも重複チェックをするのが理想だが
    // 計算コストと「燃え広がり」の性質上、簡易的に済ませる
    // (ただし自己ループだけは避ける)
    if (ambassador == new_node) return;

    // 接続1 (大使)
    bool connected = false;
    for(int existing : adj[new_node]) if(existing == ambassador) connected = true;
    
    if (!connected) {
        adj[new_node].push_back(ambassador);
        adj[ambassador].push_back(new_node);
        degrees[new_node]++; degrees[ambassador]++;
        degree_bag.push_back(new_node); degree_bag.push_back(ambassador);
    }

    if (adj[ambassador].empty()) return;
    
    std::geometric_distribution<> geom(1.0 - p_burn);
    int n_spread = geom(gen);
    
    if (n_spread > 0) {
        std::vector<int> targets = adj[ambassador];
        std::shuffle(targets.begin(), targets.end(), gen);
        int count = 0;
        for(int t : targets) {
            if (t == new_node) continue;
            
            // 重複チェック
            bool already = false;
            for(int existing : adj[new_node]) if(existing == t) already = true;
            if(already) continue;

            adj[new_node].push_back(t);
            adj[t].push_back(new_node);
            degrees[new_node]++; degrees[t]++;
            degree_bag.push_back(new_node); degree_bag.push_back(t);

            count++;
            if(count >= n_spread) break;
        }
    }
}

// 3. Neighbor Strength
int select_neighbor_strength(const std::vector<std::vector<int>>& adj, 
                             const std::vector<int>& degree_bag) {
    if (degree_bag.empty()) return 0;
    std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
    int hub = degree_bag[bag_dis(gen)];
    
    if (adj[hub].empty()) return hub;
    std::uniform_int_distribution<> neigh_dis(0, adj[hub].size() - 1);
    return adj[hub][neigh_dis(gen)];
}

// 4. MA & 5. TRA はメインループ内でロジック処理するため関数化しない

// 6. ADM
int select_adm(int n_current, const std::vector<int>& degrees, bool assortative) {
    int best_cand = -1;
    int best_score = assortative ? 100000 : -1;
    for(int i=0; i<5; ++i) {
        std::uniform_int_distribution<> dis(0, n_current - 1);
        int cand = dis(gen);
        int deg = degrees[cand];
        if (assortative) {
            if (deg < best_score) { best_score = deg; best_cand = cand; }
        } else {
            if (deg > best_score) { best_score = deg; best_cand = cand; }
        }
    }
    return best_cand;
}

// --- メイン関数 ---
// [[Rcpp::export]]
List netmix_core(int n_target, NumericVector probs, NumericVector params) {
    // パラメータ展開
    double p_gpa = probs[0];
    double p_ff  = probs[1];
    double p_ns  = probs[2];
    double p_ma  = probs[3];
    double p_tra = probs[4];
    
    double alpha   = params[0];
    double beta    = params[1];
    int    m       = (int)std::round(params[2]); if(m<1) m=1;
    double p_burn  = params[3];
    int    K       = (int)std::round(params[4]); if(K<1) K=1; // TRAでは使わないが引数互換性のため残す
    bool   is_assort = (params[5] > 0.5); 

    std::vector<std::vector<int>> adj(n_target);
    std::vector<int> degrees(n_target, 0);
    std::vector<int> capacity(n_target);
    std::vector<int> degree_bag; 
    
    // igraph用
    std::vector<int> from, to;

    // 初期化
    adj[0].push_back(1); adj[1].push_back(0);
    degrees[0]=1; degrees[1]=1;
    degree_bag.push_back(0); degree_bag.push_back(1);
    // from/toへの追加は最後のエッジリスト構築でまとめて行うか、逐次行う。
    // ここでは逐次行う
    from.push_back(1); to.push_back(2);

    std::uniform_real_distribution<> runif(0.0, 1.0);
    for(int i=0; i<n_target; ++i) {
        double u = runif(gen);
        double cap = (double)m / std::pow(u, 1.0/beta); 
        if (cap > n_target) cap = n_target;
        capacity[i] = (int)cap;
    }

    // 生成ループ
    for (int i = 2; i < n_target; ++i) {
        double dice = runif(gen);
        double cum = 0.0;
        
        // Forest Fire
        cum += p_ff;
        if (dice < cum) {
            // FFは内部でエッジ追加処理を行う(from/toは今回は省略、adjのみ更新して最後に変換する方が安全だが、
            // 既存コードとの互換性のため、ここではadjのみ更新し、戻り値構築時にadjから全エッジを生成する方式に変更する)
            // ★重要: 重複排除を確実にするため、from/toは最後にadjから一括生成します。
            process_forest_fire(i, p_burn, adj, degrees, degree_bag);
            continue; 
        }

        // --- MA用のプロトタイプ固定 ---
        // このノード生成ステップの間、ずっと同じプロトタイプを使う
        int ma_prototype = -1;
        if (degree_bag.size() > 0) {
             std::uniform_int_distribution<> dis(0, degree_bag.size() - 1);
             ma_prototype = degree_bag[dis(gen)]; // 次数比例でプロトタイプを選ぶとより効果的
        }

        // --- 今回追加した接続先を記録するリスト (TRA用) ---
        std::vector<int> current_targets;

        for (int k = 0; k < m; ++k) {
            int target = -1;
            int max_retries = 5; // 重複回避のためのリトライ回数
            
            for(int retry = 0; retry < max_retries; ++retry) {
                target = -1;
                dice = runif(gen); // 毎回サイコロを振る(プロセス混合)
                cum = 0.0; // リセット

                // 1. GPA
                if (target == -1) { cum += p_gpa; if (dice < cum) target = select_generalized_pa(i, degrees, capacity, alpha, degree_bag); }
                
                // 2. NS
                if (target == -1) { cum += p_ns;  if (dice < cum) target = select_neighbor_strength(adj, degree_bag); }
                
                // 3. MA (固定プロトタイプ版)
                if (target == -1) { 
                    cum += p_ma;  
                    if (dice < cum) {
                        if (ma_prototype != -1 && !adj[ma_prototype].empty()) {
                            std::uniform_int_distribution<> neigh_dis(0, adj[ma_prototype].size() - 1);
                            target = adj[ma_prototype][neigh_dis(gen)];
                        } else {
                            target = ma_prototype;
                        }
                    } 
                }
                
                // 4. TRA (三角形閉包版 - Triangle Closure)
                // 「さっき繋いだ相手(current_targets)」の隣人と繋ぐ
                if (target == -1) { 
                    cum += p_tra; 
                    if (dice < cum) {
                        if (!current_targets.empty()) {
                            // さっき繋いだ相手をランダムに選ぶ
                            std::uniform_int_distribution<> cur_dis(0, current_targets.size() - 1);
                            int u = current_targets[cur_dis(gen)];
                            if (!adj[u].empty()) {
                                std::uniform_int_distribution<> neigh_dis(0, adj[u].size() - 1);
                                target = adj[u][neigh_dis(gen)];
                            }
                        } else {
                            // まだ誰も繋いでないなら、仕方ないのでGPAかMAで代替
                            target = select_generalized_pa(i, degrees, capacity, alpha, degree_bag);
                        }
                    } 
                }
                
                // 5. ADM
                if (target == -1) { target = select_adm(i, degrees, is_assort); }
                
                // 自己ループ回避
                if (target == i) target = -1;

                // 重複回避 (既に繋がっているならやり直し)
                if (target != -1) {
                    bool already_connected = false;
                    for (int existing : adj[i]) {
                        if (existing == target) { already_connected = true; break; }
                    }
                    if (already_connected) target = -1; // リトライへ
                }

                if (target != -1) break; // 成功！ループを抜ける
            }

            // リトライしてもダメなら諦めてランダム接続 (次数確保のため)
            if (target == -1) {
                 std::uniform_int_distribution<> r_dis(0, i - 1);
                 target = r_dis(gen);
                 // それでも重複する可能性はあるが、これ以上はコスト高なので許容
            }

            // 接続処理
            adj[i].push_back(target);
            adj[target].push_back(i);
            degrees[i]++; degrees[target]++;
            degree_bag.push_back(i); degree_bag.push_back(target);
            
            current_targets.push_back(target); // TRA用に記録
        }
    }

    // 最後に adj から from/to を一括生成 (これが最も安全で正確)
    from.clear(); to.clear();
    for(int i=0; i<n_target; ++i) {
        for(int neighbor : adj[i]) {
            if (i < neighbor) { // 重複防止 (i < j のペアのみ保存)
                from.push_back(i + 1);
                to.push_back(neighbor + 1);
            }
        }
    }

    return List::create(Named("from") = from, Named("to") = to);
}
