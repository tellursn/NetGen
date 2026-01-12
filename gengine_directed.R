#include <Rcpp.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <set>

using namespace Rcpp;

// 乱数生成器
std::random_device rd;
std::mt19937 gen(rd());

// --- ヘルパー関数 (有向版) ---

// 1. GPA (入次数に基づく選択)
// 人気 = 「どれだけリンクを受けているか (in-degree)」
int select_gpa_directed(int n_current, const std::vector<int>& in_degrees, 
                        double alpha, const std::vector<int>& degree_bag) {
    // degree_bag は入次数ベースで管理する
    if (degree_bag.empty()) return 0;
    
    int max_trials = 10;
    for (int t = 0; t < max_trials; ++t) {
        std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
        int cand = degree_bag[bag_dis(gen)];
        
        // Alpha補正 (入次数に対する非線形性)
        if (std::abs(alpha - 1.0) > 0.01) {
            // 入次数0のノードが選ばれないのを防ぐため +1 する慣習があるが
            // bagに入っている時点でin-degree >= 1 なのでそのまま計算
            double p_accept = std::pow((double)in_degrees[cand], alpha - 1.0);
            if (std::uniform_real_distribution<>(0.0, 1.0)(gen) > p_accept) continue;
        }
        return cand;
    }
    // フォールバック: ランダム
    std::uniform_int_distribution<> rand_dis(0, n_current - 1);
    return rand_dis(gen);
}

// 2. NS (Neighbor Strength)
// ハブ(in-degreeが高い)が「指している先(out-neighbor)」を選ぶ
// 論文の引用関係に近い (有名な論文が引用している論文もまた読みたくなる)
int select_ns_directed(const std::vector<std::vector<int>>& adj, 
                       const std::vector<int>& degree_bag) {
    if (degree_bag.empty()) return 0;
    std::uniform_int_distribution<> bag_dis(0, degree_bag.size() - 1);
    int hub = degree_bag[bag_dis(gen)]; // 人気者を選ぶ
    
    if (adj[hub].empty()) return hub; // 誰も指していなければその人自身
    
    std::uniform_int_distribution<> neigh_dis(0, adj[hub].size() - 1);
    return adj[hub][neigh_dis(gen)]; // その人が指している先を返す
}

// 3. Smart TRA (有向版)
// A -> B -> C というパスがあるとき、A -> C を張る (Transitivity)
int select_smart_tra_directed(int node_i, const std::vector<int>& step_targets, 
                              const std::vector<std::vector<int>>& adj, 
                              const std::vector<int>& in_degrees,
                              bool want_assortative) {
    
    if (step_targets.empty()) return -1;
    std::uniform_int_distribution<> sd(0, step_targets.size() - 1);
    int pivot = step_targets[sd(gen)]; // さっき自分が繋いだ相手(B)
    
    if (adj[pivot].empty()) return -1; // Bが誰も指していなければ終了
    
    const std::vector<int>& candidates = adj[pivot]; // Bが指している相手(Cたち)
    
    int best_cand = -1;
    double best_score = -1.0;
    // 新規ノードはまだin-degree=0なので、mを代用するか0とする
    // ここでは「相手のin-degree」と「自分の予定out-degree(m)」の差を見るなどの戦略があるが
    // シンプルに相手のin-degreeだけで判断するロジックに変更
    
    int n_sample = std::min((int)candidates.size(), 10);
    std::vector<int> sample_indices(candidates.size());
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    std::shuffle(sample_indices.begin(), sample_indices.end(), gen);

    for(int k=0; k<n_sample; ++k) {
        int cand = candidates[sample_indices[k]];
        if (cand == node_i) continue; 

        // ここでは単純にターゲットの人気度(in-degree)を見る
        // E. coli (Disassortative) なら、人気のないノードとも繋がるべき？
        // あるいは次数差？
        // ここでは前のロジックを踏襲し、in_degrees[cand] を評価
        
        double score = (double)in_degrees[cand]; 
        
        // Disassortative (負の相関) を狙う場合
        // 「自分(弱小)は、相手(強者)と繋がるべき」なので、scoreが高い方がいい？
        // 実はAssortativityの定義によるが、ここではシンプルにランダムに近い選択をする
        
        best_cand = cand; // 一旦ランダム (シャッフルされているため)
        break; 
    }
    return best_cand;
}

// 4. ADM (有向版)
// 入次数(in-degree)に基づいて選ぶ
int select_adm_directed(int n_current, const std::vector<int>& in_degrees, bool assortative) {
    int best_cand = -1;
    int best_val = assortative ? 100000 : -1;
    
    int trials = 10; 
    for(int i=0; i<trials; ++i) { 
        std::uniform_int_distribution<> dis(0, n_current - 1);
        int cand = dis(gen);
        int deg = in_degrees[cand];
        
        if (assortative) {
            if (deg < best_val) { best_val = deg; best_cand = cand; }
        } else {
            if (deg > best_val) { best_val = deg; best_cand = cand; }
        }
    }
    return best_cand;
}

// 5. Forest Fire (有向版)
// 矢印の方向に燃え広がる
int process_forest_fire_directed(int new_node, double p_burn, 
                                 std::vector<std::vector<int>>& adj, 
                                 std::vector<int>& in_degrees,
                                 std::vector<int>& degree_bag) {
    int n_current = new_node;
    std::uniform_int_distribution<> dis(0, n_current - 1);
    int ambassador = dis(gen); // ランダムな大使を選ぶ
    
    if (ambassador == new_node) return 0;

    int edges_added = 0;
    
    // まず大使に繋ぐ (New -> Ambassador)
    adj[new_node].push_back(ambassador);
    in_degrees[ambassador]++;
    degree_bag.push_back(ambassador); // 入次数が増えたのでバッグに入れる
    edges_added++;

    if (adj[ambassador].empty()) return edges_added;

    // 大使の「出力先」へ燃え広がる
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
            for(int ex : adj[new_node]) if(ex == t) already = true;
            if (already) continue;

            adj[new_node].push_back(t);
            in_degrees[t]++;
            degree_bag.push_back(t);
            edges_added++;
            count++;
            if(count >= n_spread) break;
        }
    }
    return edges_added;
}

// --- メイン関数 (有向版) ---
// [[Rcpp::export]]
List netmix_core_directed(int n_target, NumericVector probs, NumericVector params) {
    
    double p_gpa = probs[0];
    double p_ff  = probs[1];
    double p_ns  = probs[2];
    double p_ma  = probs[3];
    double p_tra = probs[4];
    double p_adm = probs[5];
    
    double alpha   = params[0];
    double beta    = params[1];
    double m_param = params[2]; 
    double p_burn  = params[3];
    bool   is_assort = (params[5] > 0.5); 

    double sum_growth = p_gpa + p_ns + p_ma + p_tra + p_adm;
    if (sum_growth < 1e-9) sum_growth = 1.0; 

    // adj[u] は u から出るエッジのリスト
    std::vector<std::vector<int>> adj(n_target);
    // in_degrees[u] は u に入るエッジの数
    std::vector<int> in_degrees(n_target, 0);
    // degree_bag は「入次数に比例して選ばれる」ための袋
    std::vector<int> degree_bag; 
    
    // 初期化: 0 <-> 1 (相互結合) にして、両方に入次数1を持たせる
    // これをしないと、誰も入次数を持たずGPAが機能しなくなる
    adj[0].push_back(1); in_degrees[1]++;
    adj[1].push_back(0); in_degrees[0]++;
    degree_bag.push_back(0); degree_bag.push_back(1);

    std::uniform_real_distribution<> runif(0.0, 1.0);
    int m_base = (int)m_param;

    for (int i = 2; i < n_target; ++i) {
        
        int current_m = m_base;
        if (runif(gen) < (m_param - m_base)) current_m++;
        if (current_m < 1) current_m = 1;

        bool is_ff_step = (runif(gen) < p_ff);
        int added_count = 0;

        if (is_ff_step) {
            added_count = process_forest_fire_directed(i, p_burn, adj, in_degrees, degree_bag);
        }

        int needed = current_m - added_count;
        if (needed <= 0) continue; 

        // Growth Step
        int ma_prototype = -1;
        if (!degree_bag.empty()) {
             std::uniform_int_distribution<> dis(0, degree_bag.size() - 1);
             ma_prototype = degree_bag[dis(gen)];
        }
        
        std::vector<int> step_targets; 

        for (int k = 0; k < needed; ++k) {
            int target = -1;
            int retries = 0;
            
            while (target == -1 && retries < 10) {
                retries++;
                double r = runif(gen) * sum_growth;
                double cum = 0.0;

                // 1. GPA (入次数比例)
                cum += p_gpa;
                if (target == -1 && r < cum) {
                    target = select_gpa_directed(i, in_degrees, alpha, degree_bag);
                }
                // 2. NS (ハブの出力先)
                cum += p_ns;
                if (target == -1 && r < cum) {
                    target = select_ns_directed(adj, degree_bag);
                }
                // 3. MA (モジュール)
                cum += p_ma;
                if (target == -1 && r < cum) {
                     if (ma_prototype != -1) {
                        if (!adj[ma_prototype].empty()) {
                            std::uniform_int_distribution<> nd(0, adj[ma_prototype].size()-1);
                            target = adj[ma_prototype][nd(gen)];
                        } else { target = ma_prototype; }
                     } else { target = select_gpa_directed(i, in_degrees, alpha, degree_bag); }
                }
                // 4. TRA (推移性)
                cum += p_tra;
                if (target == -1 && r < cum) {
                    target = select_smart_tra_directed(i, step_targets, adj, in_degrees, is_assort);
                    if (target == -1) target = select_gpa_directed(i, in_degrees, alpha, degree_bag);
                }
                // 5. ADM
                cum += p_adm;
                if (target == -1 && r <= cum + 1e-9) { 
                    target = select_adm_directed(i, in_degrees, is_assort);
                }

                // チェック (自分自身 or 既に接続済み)
                if (target == i) { target = -1; continue; }
                for(int ex : adj[i]) if(ex == target) { target = -1; break; }
            }

            if (target == -1) {
                std::uniform_int_distribution<> rd(0, i - 1);
                target = rd(gen);
            }

            // 有向接続 (i -> target)
            adj[i].push_back(target);
            in_degrees[target]++;
            degree_bag.push_back(target); // 入次数が増えたのでバッグへ
            
            step_targets.push_back(target);
        }
    }

    std::vector<int> from, to;
    for(int i=0; i<n_target; ++i) {
        for(int neighbor : adj[i]) {
            // 有向なので、i -> neighbor をそのまま記録
            from.push_back(i + 1);
            to.push_back(neighbor + 1);
        }
    }
    return List::create(Named("from") = from, Named("to") = to);
}
