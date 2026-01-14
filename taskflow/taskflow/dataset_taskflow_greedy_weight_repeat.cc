/**
 * @file max_clique_weighted_random_repeat.cpp
 * @brief 最大クリーク問題：重み付きランダムソート（複数回実行・統計収集版）
 */

// --- ヘッダファイルのインクルード ---
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
#include <iomanip> // 出力フォーマット用

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
        if (u >= 0 && u < num_vertices_) {
            return adj_list_[u].size();
        }
        return 0;
    }

    // 次数順ソート（ベースライン用）
    std::vector<int> get_vertices_sorted_by_degree() const {
        std::vector<std::pair<int, int>> degrees;
        degrees.reserve(num_vertices_);
        for (int i = 0; i < num_vertices_; ++i) {
            degrees.push_back({-static_cast<int>(adj_list_[i].size()), i});
        }
        std::sort(degrees.begin(), degrees.end());

        std::vector<int> sorted_vertices;
        sorted_vertices.reserve(num_vertices_);
        for (const auto& p : degrees) {
            sorted_vertices.push_back(p.second);
        }
        return sorted_vertices;
    }
    
    // 貪欲法
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
    if (!ifs) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    int num_vertices = 0;
    int num_edges = 0;
    
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        if (line[0] == 'p') {
            std::stringstream ss(line);
            std::string temp, format;
            ss >> temp >> format >> num_vertices >> num_edges;
            break;
        }
    }

    if (num_vertices == 0) {
        std::cerr << "Error: Invalid DIMACS format." << std::endl;
        exit(1);
    }

    std::cout << "Loading DIMACS graph: " << filename << std::endl;
    std::cout << "Vertices: " << num_vertices << ", Edges: " << num_edges << std::endl;

    Graph g(num_vertices);

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        std::stringstream ss(line);
        char type;
        int u, v;
        if (line[0] == 'e') {
            ss >> type >> u >> v;
        } else {
            std::stringstream temp_ss(line);
            if (!(temp_ss >> u >> v)) continue; 
        }
        if (u > 0) u--; 
        if (v > 0) v--;
        g.add_edge(u, v);
    }
    return g;
}

// --- メイン関数 ---
int main() {
    // --- 設定 ---
    const std::string filename = "C2000.9.clq"; 
    const int NUM_EXPERIMENTS = 100;   // 実験を繰り返す回数（サンプル数）
    const int NUM_TRIALS_PER_RUN = 10000; // 1回の実験で行うTaskflowの試行回数

    // --- 1. グラフ読み込み（これは1回だけで良い） ---
    std::cout << "Reading graph file..." << std::endl;
    Graph large_graph = load_dimacs_graph(filename);
    std::cout << "Graph loaded successfully.\n" << std::endl;

    // --- 2. 次数計算（これも1回だけで良い） ---
    std::vector<int> degrees(large_graph.get_num_vertices());
    for(int i=0; i<large_graph.get_num_vertices(); ++i) {
        degrees[i] = large_graph.get_degree(i);
    }

    std::cout << "Starting experiments (" << NUM_EXPERIMENTS << " runs, " 
              << NUM_TRIALS_PER_RUN << " trials/run)..." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    // スレッドプールはループの外で作って使い回す（生成コスト削減）
    tf::Executor executor;
    
    // 統計用データ
    std::vector<int> results;
    std::vector<double> times;

    // --- 3. 実験ループ ---
    for (int run = 0; run < NUM_EXPERIMENTS; ++run) {
        
        // 各実験ごとの変数
        tf::Taskflow taskflow;
        std::vector<int> best_clique; // 毎回リセット
        std::mutex mtx; 

        auto start_time = std::chrono::high_resolution_clock::now();

        // (A) タスク1: 純粋な次数順（ベースライン）
        taskflow.emplace([&]() {
            std::vector<int> initial_order = large_graph.get_vertices_sorted_by_degree();
            std::vector<int> current_clique = large_graph.find_greedy_max_clique(initial_order);

            std::lock_guard<std::mutex> lock(mtx);
            if (current_clique.size() > best_clique.size()) {
                best_clique = std::move(current_clique);
            }
        });

        // (B) タスク2〜N: 重み付きランダムソート
        for (int i = 0; i < NUM_TRIALS_PER_RUN - 1; ++i) {
            taskflow.emplace([&, i]() { 
                std::random_device t_rd;
                std::mt19937 t_gen(t_rd());
                
                // ノイズ設定 (±50)
                std::uniform_real_distribution<> noise_dist(-50.0, 50.0);

                std::vector<std::pair<double, int>> weighted_vertices;
                weighted_vertices.reserve(degrees.size());

                for(size_t v = 0; v < degrees.size(); ++v) {
                    double score = degrees[v] + noise_dist(t_gen);
                    weighted_vertices.push_back({score, static_cast<int>(v)});
                }

                std::sort(weighted_vertices.rbegin(), weighted_vertices.rend());

                std::vector<int> search_order;
                search_order.reserve(degrees.size());
                for(const auto& p : weighted_vertices) {
                    search_order.push_back(p.second);
                }

                std::vector<int> current_clique = large_graph.find_greedy_max_clique(search_order);

                std::lock_guard<std::mutex> lock(mtx);
                if (current_clique.size() > best_clique.size()) {
                    best_clique = std::move(current_clique);
                }
            });
        }

        // 実行待機
        executor.run(taskflow).wait();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // 結果記録
        results.push_back(best_clique.size());
        times.push_back(duration.count());

        std::cout << "Run " << std::setw(2) << (run + 1) 
                  << ": Best Size = " << best_clique.size() 
                  << ", Time = " << duration.count() << " ms" << std::endl;
    }

    // --- 4. 統計データの表示 ---
    double avg_size = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    int max_size = *std::max_element(results.begin(), results.end());
    int min_size = *std::min_element(results.begin(), results.end());

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Summary (" << NUM_EXPERIMENTS << " runs):" << std::endl;
    std::cout << "  Max Size : " << max_size << std::endl;
    std::cout << "  Min Size : " << min_size << std::endl;
    std::cout << "  Avg Size : " << avg_size << std::endl;
    std::cout << "  Avg Time : " << avg_time << " ms" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    return 0;
}