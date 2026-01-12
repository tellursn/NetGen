#include <Rcpp.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <set>

using namespace Rcpp;

// 乱数生成
std::random_device rd;
std::mt19937 gen(rd());

// Generalized PA 
int select_gpa(int n_current, const std::vector<int>& degrees, 
               const std::vector<int>& capacity, 
               double alpha, const std::vector<int>& degree_bag) {
    if (degree_bag.empty()) return 0;
    int max_trials = 10;
    for (int t = 0; t < max_trials; ++t) {
        std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
        int cand = degree_bag[bag_dis(gen)];
        if (degrees[cand] >= capacity[cand]) continue; // 定員チェック
        if (std::abs(alpha - 1.0) > 0.01) {
            double p_accept = std::pow((double)degrees[cand], alpha - 1.0);
            if (std::uniform_real_distribution<>(0.0, 1.0)(gen) > p_accept) continue;
        }
        return cand;
    }
    // フォールバック: 完全ランダム
    std::uniform_int_distribution<> rand_dis(0, n_current - 1);
    return rand_dis(gen);
}

// Neighbor Strength
int select_ns(const std::vector<std::vector<int>>& adj, const std::vector<int>& degree_bag) {
    if (degree_bag.empty()) return 0;
    std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
    int hub = degree_bag[bag_dis(gen)];
    if (adj[hub].empty()) return hub;
    std::uniform_int_distribution<> neigh_dis(0, adj[hub].size() - 1);
    return adj[hub][neigh_dis(gen)];
}

// ADM
int select_adm(int n_current, const std::vector<int>& degrees, bool assortative) {
    int best_cand = -1;
    int best_score = assortative ? 100000 : -1;
    for(int i=0; i<5; ++i) { // 候補数5
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

// Forest Fire
int process_forest_fire(int new_node, double p_burn, 
                        std::vector<std::vector<int>>& adj, 
                        std::vector<int>& degrees,
                        std::vector<int>& degree_bag) {
    int n_current = new_node;
    std::uniform_int_distribution<> dis(0, n_current - 1);
    int ambassador = dis(gen);
    
    if (ambassador == new_node) return 0;

    int edges_added = 0;

    // 大使との接続
    bool connected = false;
    for(int ex : adj[new_node]) if(ex == ambassador) connected = true;
    if (!connected) {
        adj[new_node].push_back(ambassador);
        adj[ambassador].push_back(new_node);
        degrees[new_node]++; degrees[ambassador]++;
        degree_bag.push_back(new_node); degree_bag.push_back(ambassador);
        edges_added++;
    }

    if (adj[ambassador].empty()) return edges_added;

    // 燃え広がり
    std::geometric_distribution<> geom(1.0 - p_burn);
    int n_spread = geom(gen);
    
    if (n_spread > 0) {
        std::vector<int> targets = adj[ambassador];
        std::shuffle(targets.begin(), targets.end(), gen);
        int count = 0;
        for(int t : targets) {
            if (t == new_node) continue;
            bool already = false;
            for(int ex : adj[new_node]) if(ex == t) already = true;
            if (already) continue;

            adj[new_node].push_back(t);
            adj[t].push_back(new_node);
            degrees[new_node]++; degrees[t]++;
            degree_bag.push_back(new_node); degree_bag.push_back(t);
            edges_added++;
            count++;
            if(count >= n_spread) break;
        }
    }
    return edges_added;
}

// --- メイン関数 ---
// [[Rcpp::export]]
List netmix_core(int n_target, NumericVector probs, NumericVector params) {
    
    // 確率: [GPA, FF, NS, MA, TRA, ADM]
    double p_gpa = probs[0];
    double p_ff  = probs[1];
    double p_ns  = probs[2];
    double p_ma  = probs[3];
    double p_tra = probs[4];
    double p_adm = probs[5];
    
    // パラメータ
    double alpha   = params[0];
    double beta    = params[1];
    int    m       = (int)std::round(params[2]); if(m<1) m=1;
    double p_burn  = params[3];
    // KはTRAでは直接使わないが互換性のため維持
    bool   is_assort = (params[5] > 0.5); 

    // FF以外の確率を再正規化 (Growth Step用)
    double sum_growth = p_gpa + p_ns + p_ma + p_tra + p_adm;
    if (sum_growth < 1e-9) sum_growth = 1.0; // avoid div0

    std::vector<std::vector<int>> adj(n_target);
    std::vector<int> degrees(n_target, 0);
    std::vector<int> capacity(n_target);
    std::vector<int> degree_bag; 
    
    // 初期化 (0-1)
    adj[0].push_back(1); adj[1].push_back(0);
    degrees[0]=1; degrees[1]=1;
    degree_bag.push_back(0); degree_bag.push_back(1);

    std::uniform_real_distribution<> runif(0.0, 1.0);
    
    // 定員設定
    for(int i=0; i<n_target; ++i) {
        double u = runif(gen);
        double cap = (double)m / std::pow(u, 1.0/beta); 
        if (cap > n_target) cap = n_target;
        capacity[i] = (int)cap;
    }

    for (int i = 2; i < n_target; ++i) {
        // Step 1: Forest Fire かどうか？
        bool is_ff_step = false;
        if (runif(gen) < p_ff) is_ff_step = true;

        int added_count = 0;

        if (is_ff_step) {
            added_count = process_forest_fire(i, p_burn, adj, degrees, degree_bag);
        }

        // Step 2: 不足分を埋める (Growth Step or Filler)
        // FFで足りなかった場合、またはFFじゃなかった場合、合計m本になるまで追加する
        int needed = m - added_count;
        if (needed <= 0) continue; // 十分足りてるなら次へ

        // Growth Step用の準備
        int ma_prototype = -1;
        if (!degree_bag.empty()) {
             std::uniform_int_distribution<> dis(0, degree_bag.size() - 1);
             ma_prototype = degree_bag[dis(gen)];
        }
        std::vector<int> step_targets; // このステップで接続した相手リスト(TRA用)

        for (int k = 0; k < needed; ++k) {
            int target = -1;
            int retries = 0;
            
            while (target == -1 && retries < 5) {
                retries++;
                double r = runif(gen) * sum_growth;
                double cum = 0.0;

                // --- ルーレット選択 ---
                
                // 1. GPA
                cum += p_gpa;
                if (target == -1 && r < cum) {
                    target = select_gpa(i, degrees, capacity, alpha, degree_bag);
                }

                // 2. NS
                cum += p_ns;
                if (target == -1 && r < cum) {
                    target = select_ns(adj, degree_bag);
                }

                // 3. MA (Sticky)
                cum += p_ma;
                if (target == -1 && r < cum) {
                     if (ma_prototype != -1) {
                        if (!adj[ma_prototype].empty()) {
                            std::uniform_int_distribution<> nd(0, adj[ma_prototype].size()-1);
                            target = adj[ma_prototype][nd(gen)];
                        } else {
                            target = ma_prototype;
                        }
                     } else {
                         target = select_gpa(i, degrees, capacity, alpha, degree_bag);
                     }
                }

                // 4. TRA (Strong Triangle Closure)
                cum += p_tra;
                if (target == -1 && r < cum) {
                    // さっき繋いだ相手(step_targets)がいれば、その隣人を狙う
                    if (!step_targets.empty()) {
                        std::uniform_int_distribution<> sd(0, step_targets.size() - 1);
                        int u = step_targets[sd(gen)];
                        if (!adj[u].empty()) {
                            std::uniform_int_distribution<> nd(0, adj[u].size() - 1);
                            target = adj[u][nd(gen)];
                        }
                    } 
                    // もしいなければ、GPAで「起点」を作る
                    if (target == -1) {
                        target = select_gpa(i, degrees, capacity, alpha, degree_bag);
                    }
                }

                // 5. ADM
                cum += p_adm;
                if (target == -1 && r <= cum + 1e-9) { // 浮動小数点誤差対策
                    target = select_adm(i, degrees, is_assort);
                }

                // --- チェック ---
                if (target == i) { target = -1; continue; } // 自己ループ
                for(int ex : adj[i]) if(ex == target) { target = -1; break; } // 重複
            }

            // フォールバック (諦めてランダム)
            if (target == -1) {
                std::uniform_int_distribution<> rd(0, i - 1);
                target = rd(gen);
                // ここでの重複は許容する(稀なので)
            }

            // 接続確定
            adj[i].push_back(target);
            adj[target].push_back(i);
            degrees[i]++; degrees[target]++;
            degree_bag.push_back(i); degree_bag.push_back(target);
            
            step_targets.push_back(target);
        }
    }

    // エッジリスト変換
    std::vector<int> from, to;
    for(int i=0; i<n_target; ++i) {
        for(int neighbor : adj[i]) {
            if (i < neighbor) {
                from.push_back(i + 1);
                to.push_back(neighbor + 1);
            }
        }
    }

    return List::create(Named("from") = from, Named("to") = to);
}
