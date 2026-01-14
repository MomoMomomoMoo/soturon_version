/**
 * @file max_clique_hybrid_fast.cpp
 * @brief 最大クリーク問題：近傍分解 + 重み付きランダム (高速化版)
 * @details thread_localを使用して乱数生成器の初期化コストを極限まで削減
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <numeric>
#include <mutex>
#include <thread>
#include <iomanip>
#include <atomic> // 高速化のためAtomic使用

#include "taskflow.hpp"

// --- Graphクラス ---
class Graph {
private:
    int num_vertices_;
    std::vector<std::unordered_set<int>> adj_list_;

public:
    Graph(int vertices) : num_vertices_(vertices), adj_list_(vertices) {}

    int get_num_vertices() const { return num_vertices_; }

    void add_edge(int u, int v) {
        if (u >= 0 && u < num_vertices_ && v >= 0 && v < num_vertices_) {
            adj_list_[u].insert(v);
            adj_list_[v].insert(u);
        }
    }

    bool is_adjacent(int u, int v) const {
        return adj_list_[u].count(v);
    }

    int get_degree(int u) const {
        return (u >= 0 && u < num_vertices_) ? adj_list_[u].size() : 0;
    }

    std::vector<int> get_neighbors(int u) const {
        if (u < 0 || u >= num_vertices_) return {};
        std::vector<int> neighbors(adj_list_[u].begin(), adj_list_[u].end());
        return neighbors;
    }

    // 渡された順序をそのまま守って探索する関数
    std::vector<int> find_greedy_max_clique(const std::vector<int>& vertex_order) const {
        std::vector<int> clique;
        clique.reserve(100); 
        for (int u : vertex_order) {
            bool can_add = true;
            for (int v : clique) {
                if (!is_adjacent(u, v)) {
                    can_add = false;
                    break;
                }
            }
            if (can_add) {
                clique.push_back(u);
            }
        }
        return clique;
    }
};

// --- DIMACS読み込み関数 ---
Graph load_dimacs_graph(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error: " << filename << std::endl; exit(1); }
    std::string line; int num_vertices = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        if (line[0] == 'p') { std::stringstream ss(line); std::string t; ss >> t >> t >> num_vertices; break; }
    }
    Graph g(num_vertices);
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        std::stringstream ss(line); char type; int u, v;
        if (line[0] == 'e') ss >> type >> u >> v;
        else { std::stringstream tss(line); if(!(tss >> u >> v)) continue; }
        if (u > 0) u--; if (v > 0) v--;
        g.add_edge(u, v);
    }
    return g;
}

// --- メイン関数 ---
int main() {
    const std::string filename = "C500.9.clq"; 
    const int NUM_EXPERIMENTS = 10;       // 実験回数
    const int NUM_TRIALS_PER_RUN = 50000; // 試行回数（高速化したので増やしてもOK）

    std::cout << "Reading graph..." << std::endl;
    Graph graph = load_dimacs_graph(filename);
    std::cout << "Graph loaded.\n" << std::endl;

    // 事前に全頂点の次数を計算（高速化）
    std::vector<int> global_degrees(graph.get_num_vertices());
    for(int i=0; i<graph.get_num_vertices(); ++i) global_degrees[i] = graph.get_degree(i);

    std::cout << "Starting FAST HYBRID experiments (thread_local optimized)..." << std::endl;
    std::cout << "Trials per run: " << NUM_TRIALS_PER_RUN << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    tf::Executor executor;
    std::vector<int> results;
    std::vector<double> times;
    int total_vertices = graph.get_num_vertices();

    for (int run = 0; run < NUM_EXPERIMENTS; ++run) {
        
        tf::Taskflow taskflow;
        std::vector<int> best_clique;
        std::mutex mtx; 
        std::atomic<size_t> max_size(0); // ダブルチェック用Atomic

        auto start_time = std::chrono::high_resolution_clock::now();

        // 並列ループ
        for (int i = 0; i < NUM_TRIALS_PER_RUN; ++i) {
            // Taskflowのタスク
            taskflow.emplace([&, i]() { 
                
                // ★★★ ここが修正ポイント ★★★
                // static thread_local を使うことで、スレッドごとに1回だけ初期化され、
                // 以降はずっと使い回される（爆速になる）
                static thread_local std::random_device rd;
                static thread_local std::mt19937 gen(rd());
                
                // 1. シード頂点を選ぶ
                std::uniform_int_distribution<> dist_v(0, total_vertices - 1);
                int seed_vertex = dist_v(gen);

                // 2. 近傍（部分グラフ）を取得
                std::vector<int> neighbors = graph.get_neighbors(seed_vertex);

                // 3. 重み付きランダムソート
                std::vector<std::pair<double, int>> weighted_candidates;
                weighted_candidates.reserve(neighbors.size());
                
                // ノイズ分布
                std::uniform_real_distribution<> noise_dist(-50.0, 50.0);

                for(int u : neighbors) {
                    // 「本来の次数 + ノイズ」をスコアにする
                    double score = global_degrees[u] + noise_dist(gen);
                    weighted_candidates.push_back({score, u});
                }

                // スコア順にソート
                std::sort(weighted_candidates.rbegin(), weighted_candidates.rend());

                // 探索順序リストに戻す
                std::vector<int> search_order;
                search_order.reserve(neighbors.size());
                for(const auto& p : weighted_candidates) {
                    search_order.push_back(p.second);
                }

                // 4. 貪欲法を実行
                std::vector<int> local_clique = graph.find_greedy_max_clique(search_order);

                // シード頂点を追加
                local_clique.push_back(seed_vertex);

                // 5. 結果更新（Atomicによる高速なダブルチェック）
                if (local_clique.size() > max_size.load(std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (local_clique.size() > best_clique.size()) {
                        best_clique = std::move(local_clique);
                        max_size.store(best_clique.size(), std::memory_order_relaxed);
                    }
                }
            });
        }

        executor.run(taskflow).wait();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        results.push_back(best_clique.size());
        times.push_back(duration.count());

        std::cout << "Run " << std::setw(2) << (run + 1) 
                  << ": Best Size = " << best_clique.size() 
                  << ", Time = " << duration.count() << " ms" << std::endl;
    }

    // 統計
    double avg_size = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    int max_s = *std::max_element(results.begin(), results.end());
    int min_s = *std::min_element(results.begin(), results.end());

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Summary (Hybrid Fast / " << NUM_EXPERIMENTS << " runs):" << std::endl;
    std::cout << "  Max Size : " << max_s << std::endl;
    std::cout << "  Min Size : " << min_s << std::endl;
    std::cout << "  Avg Size : " << avg_size << std::endl;
    std::cout << "  Avg Time : " << avg_time << " ms" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    return 0;
}