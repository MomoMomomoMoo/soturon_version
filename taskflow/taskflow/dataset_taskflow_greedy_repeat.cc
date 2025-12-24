/**
 * @file max_clique_simple_random_repeat.cpp
 * @brief 最大クリーク問題：単純ランダムシャッフル版（複数回実行・統計収集版）
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

    // 次数順ソート（決定的アプローチ用）
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
    
    // 貪欲法によるクリーク探索
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
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        if (line[0] == 'p') {
            std::stringstream ss(line);
            std::string temp, format;
            ss >> temp >> format >> num_vertices; // 簡易的な読み込み
            break;
        }
    }
    if (num_vertices == 0) exit(1);

    std::cout << "Loading DIMACS graph: " << filename << " (Vertices: " << num_vertices << ")" << std::endl;
    Graph g(num_vertices);

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        std::stringstream ss(line);
        char type; int u, v;
        if (line[0] == 'e') ss >> type >> u >> v;
        else { std::stringstream tss(line); if(!(tss >> u >> v)) continue; }
        if (u > 0) u--; if (v > 0) v--;
        g.add_edge(u, v);
    }
    return g;
}

// --- メイン関数 ---
int main() {
    // --- 設定 ---
    const std::string filename = "C500.9.clq"; 
    const int NUM_EXPERIMENTS = 100;     // 実験を繰り返す回数
    const int NUM_TRIALS_PER_RUN = 10000; // 1回の実験で行うTaskflowの試行回数（多めに設定）

    // --- 1. グラフ読み込み ---
    std::cout << "Reading graph file..." << std::endl;
    Graph large_graph = load_dimacs_graph(filename);
    std::cout << "Graph loaded successfully.\n" << std::endl;

    std::cout << "Starting SIMPLE RANDOM experiments (" << NUM_EXPERIMENTS << " runs, " 
              << NUM_TRIALS_PER_RUN << " trials/run)..." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    // Executorは使い回す
    tf::Executor executor;
    
    // 統計用データ
    std::vector<int> results;
    std::vector<double> times;

    // ベースとなる頂点リスト（0, 1, 2...）を予め作っておく
    std::vector<int> base_vertices(large_graph.get_num_vertices());
    std::iota(base_vertices.begin(), base_vertices.end(), 0);

    // --- 2. 実験ループ ---
    for (int run = 0; run < NUM_EXPERIMENTS; ++run) {
        
        tf::Taskflow taskflow;
        std::vector<int> best_clique; 
        std::mutex mtx; 

        auto start_time = std::chrono::high_resolution_clock::now();

        // (A) タスク1: 次数順（決定的・ベースライン）
        taskflow.emplace([&]() {
            std::vector<int> initial_order = large_graph.get_vertices_sorted_by_degree();
            std::vector<int> current_clique = large_graph.find_greedy_max_clique(initial_order);

            std::lock_guard<std::mutex> lock(mtx);
            if (current_clique.size() > best_clique.size()) {
                best_clique = std::move(current_clique);
            }
        });

        // (B) タスク2〜N: 単純ランダムシャッフル
        for (int i = 0; i < NUM_TRIALS_PER_RUN - 1; ++i) {
            // base_vertices をコピーしてシャッフルする
            taskflow.emplace([&, base_vertices]() mutable { 
                std::random_device t_rd;
                std::mt19937 t_gen(t_rd());
                
                // コピーされたvectorをシャッフル（単純ランダム）
                std::shuffle(base_vertices.begin(), base_vertices.end(), t_gen);

                std::vector<int> current_clique = large_graph.find_greedy_max_clique(base_vertices);

                std::lock_guard<std::mutex> lock(mtx);
                if (current_clique.size() > best_clique.size()) {
                    best_clique = std::move(current_clique);
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

    // --- 3. 統計出力 ---
    double avg_size = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    int max_size = *std::max_element(results.begin(), results.end());
    int min_size = *std::min_element(results.begin(), results.end());

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Summary (Simple Random / " << NUM_EXPERIMENTS << " runs):" << std::endl;
    std::cout << "  Max Size : " << max_size << std::endl;
    std::cout << "  Min Size : " << min_size << std::endl;
    std::cout << "  Avg Size : " << avg_size << std::endl;
    std::cout << "  Avg Time : " << avg_time << " ms" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    return 0;
}