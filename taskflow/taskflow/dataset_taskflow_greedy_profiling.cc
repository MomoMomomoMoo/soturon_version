/**
 * @file max_clique_simple_random_profiling.cpp
 * @brief 最大クリーク問題：単純ランダム版のボトルネック特定プロファイリング
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
#include <atomic> // ★追加：計測用

#include "taskflow.hpp"

// --- Graphクラス (変更なし) ---
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
    bool is_adjacent(int u, int v) const { return adj_list_[u].count(v); }
    int get_degree(int u) const { return (u >= 0 && u < num_vertices_) ? adj_list_[u].size() : 0; }
    
    std::vector<int> get_vertices_sorted_by_degree() const {
        std::vector<std::pair<int, int>> degrees;
        degrees.reserve(num_vertices_);
        for (int i = 0; i < num_vertices_; ++i) {
            degrees.push_back({-static_cast<int>(adj_list_[i].size()), i});
        }
        std::sort(degrees.begin(), degrees.end());
        std::vector<int> sorted_vertices;
        sorted_vertices.reserve(num_vertices_);
        for (const auto& p : degrees) sorted_vertices.push_back(p.second);
        return sorted_vertices;
    }
    
    std::vector<int> find_greedy_max_clique(const std::vector<int>& vertex_order) const {
        std::vector<int> clique;
        clique.reserve(100); 
        for (int u : vertex_order) {
            bool can_add = true;
            for (int v : clique) {
                if (!is_adjacent(u, v)) { can_add = false; break; }
            }
            if (can_add) clique.push_back(u);
        }
        return clique;
    }
};

// --- DIMACS読み込み関数 (変更なし) ---
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
    
    // プロファイリング用に1回だけ実行
    const int NUM_EXPERIMENTS = 1;      
    const int NUM_TRIALS_PER_RUN = 10000; 

    std::cout << "Reading graph file..." << std::endl;
    Graph large_graph = load_dimacs_graph(filename);
    
    std::cout << "Starting SIMPLE RANDOM PROFILING (Trials: " << NUM_TRIALS_PER_RUN << ")..." << std::endl;

    tf::Executor executor;
    
    // ★追加: 計測用カウンタ（ナノ秒単位）
    std::atomic<long long> total_rng_init_ns{0}; // 乱数生成器の初期化
    std::atomic<long long> total_shuffle_ns{0};  // シャッフル処理
    std::atomic<long long> total_search_ns{0};   // 探索処理

    // ベースとなる頂点リスト
    std::vector<int> base_vertices(large_graph.get_num_vertices());
    std::iota(base_vertices.begin(), base_vertices.end(), 0);

    tf::Taskflow taskflow;
    std::vector<int> best_clique;
    std::mutex mtx; 

    // プロファイリングループ
    for (int i = 0; i < NUM_TRIALS_PER_RUN; ++i) {
        taskflow.emplace([&, base_vertices]() mutable { 
            // --- [区間1] 乱数生成器の初期化 ---
            auto t0 = std::chrono::high_resolution_clock::now();
            
            std::random_device t_rd;      // ★重いと予想される
            std::mt19937 t_gen(t_rd());   // ★重いと予想される

            auto t1 = std::chrono::high_resolution_clock::now();

            // --- [区間2] シャッフル ---
            std::shuffle(base_vertices.begin(), base_vertices.end(), t_gen);

            auto t2 = std::chrono::high_resolution_clock::now();

            // --- [区間3] 貪欲法探索 ---
            std::vector<int> current_clique = large_graph.find_greedy_max_clique(base_vertices);

            auto t3 = std::chrono::high_resolution_clock::now();

            // 結果更新（ロック頻度低減）
            if (current_clique.size() > 45) { 
                std::lock_guard<std::mutex> lock(mtx);
                if (current_clique.size() > best_clique.size()) {
                    best_clique = std::move(current_clique);
                }
            }

            // --- 計測時間を加算 ---
            total_rng_init_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            total_shuffle_ns  += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            total_search_ns   += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        });
    }

    auto start_total = std::chrono::high_resolution_clock::now();
    executor.run(taskflow).wait();
    auto end_total = std::chrono::high_resolution_clock::now();
    
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

    // --- 結果の分析表示 ---
    std::cout << "\n=== Profiling Results (Simple Random) ===" << std::endl;
    std::cout << "Total Wall Time: " << duration_total.count() << " ms" << std::endl;
    
    double rng_ms     = total_rng_init_ns / 1000000.0;
    double shuffle_ms = total_shuffle_ns / 1000000.0;
    double search_ms  = total_search_ns / 1000000.0;
    
    double total_sum = rng_ms + shuffle_ms + search_ms;

    std::cout << "\nCumulative CPU Time (across all threads):" << std::endl;
    std::cout << "  [1] Random Init : " << std::fixed << std::setprecision(2) << rng_ms << " ms (" 
              << (rng_ms / total_sum * 100.0) << "%)" << std::endl;
    std::cout << "  [2] Shuffle     : " << shuffle_ms << " ms (" 
              << (shuffle_ms / total_sum * 100.0) << "%)" << std::endl;
    std::cout << "  [3] Search      : " << search_ms << " ms (" 
              << (search_ms / total_sum * 100.0) << "%)" << std::endl;

    std::cout << "\nBest clique size found: " << best_clique.size() << std::endl;

    return 0;
}