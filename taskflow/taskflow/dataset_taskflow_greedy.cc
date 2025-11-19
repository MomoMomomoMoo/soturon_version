/**
 * @file max_clique_dimacs_parallel.cpp
 * @brief 最大クリーク問題：DIMACSベンチマーク対応 + Taskflow並列化 + Vector高速化版
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
#include <sstream> // 文字列解析用に追加
#include <numeric>
#include <mutex>
#include <thread>

// C++ TaskFlowライブラリのヘッダ
#include "taskflow.hpp"

/**
 * @class Graph
 * @brief グラフ構造を表現するクラス（std::vector版による高速化適用）
 */
class Graph {
private:
    int num_vertices_;
    std::vector<std::unordered_set<int>> adj_list_;

public:
    Graph(int vertices) : num_vertices_(vertices), adj_list_(vertices) {}

    int get_num_vertices() const { return num_vertices_; }

    // 辺の追加
    void add_edge(int u, int v) {
        // 範囲チェック（念のため）
        if (u >= 0 && u < num_vertices_ && v >= 0 && v < num_vertices_) {
            adj_list_[u].insert(v);
            adj_list_[v].insert(u);
        }
    }

    bool is_adjacent(int u, int v) const {
        return adj_list_[u].count(v);
    }

    // 次数順ソート
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

// --- ★追加: DIMACS形式のファイルを読み込む関数 ---
Graph load_dimacs_graph(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        std::cerr << "Make sure the file exists in the current directory." << std::endl;
        exit(1);
    }

    std::string line;
    int num_vertices = 0;
    int num_edges = 0;
    
    // 1. ヘッダー行 (p edge V E) を探す
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue; // コメントスキップ
        if (line[0] == 'p') {
            std::stringstream ss(line);
            std::string temp, format;
            ss >> temp >> format >> num_vertices >> num_edges;
            break;
        }
    }

    if (num_vertices == 0) {
        std::cerr << "Error: Invalid DIMACS format (header 'p edge V E' not found)." << std::endl;
        exit(1);
    }

    std::cout << "Loading DIMACS graph: " << filename << std::endl;
    std::cout << "Vertices: " << num_vertices << ", Edges: " << num_edges << std::endl;

    Graph g(num_vertices);

    // 2. 辺のデータを読み込む
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        
        std::stringstream ss(line);
        char type;
        int u, v;

        // 行が 'e' で始まる場合と、数字だけの場合に対応
        if (line[0] == 'e') {
            ss >> type >> u >> v;
        } else {
            // 数字のみの場合のフォールバック
            std::stringstream temp_ss(line);
            if (!(temp_ss >> u >> v)) continue; 
        }

        // DIMACSは1始まり、C++は0始まりなので変換
        if (u > 0) u--; 
        if (v > 0) v--;

        g.add_edge(u, v);
    }

    return g;
}

// --- メイン関数 ---
int main() {
    auto total_start = std::chrono::high_resolution_clock::now();

    // --- ★変更: ファイル名を指定 ---
    const std::string filename = "C500.9.clq"; 

    std::cout << "Reading graph file..." << std::endl;
    Graph large_graph = load_dimacs_graph(filename);
    std::cout << "Graph loaded successfully.\n" << std::endl;

    // --- 並列処理の準備 ---
    std::cout << "Starting parallel greedy search with Taskflow..." << std::endl;
    
    tf::Executor executor;
    tf::Taskflow taskflow;

    std::vector<int> best_clique;
    std::mutex mtx; 

    // --- すべてのコアを使用　---
    unsigned int cores = std::thread::hardware_concurrency();
    const int num_trials = 10000; // 試行回数（必要に応じて調整）

    std::cout << "Total Cores: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Concurrent trials (Full Power): " << num_trials << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // タスク1: 次数順（決定的）
    taskflow.emplace([&]() {
        std::vector<int> initial_order = large_graph.get_vertices_sorted_by_degree();
        std::vector<int> current_clique = large_graph.find_greedy_max_clique(initial_order);

        std::lock_guard<std::mutex> lock(mtx);
        if (current_clique.size() > best_clique.size()) {
            best_clique = std::move(current_clique);
        }
    });

    // タスク2〜N: ランダム順（確率的）
    for (int i = 0; i < num_trials - 1; ++i) {
        taskflow.emplace([&]() {
            std::random_device t_rd;
            std::mt19937 t_gen(t_rd());

            std::vector<int> vertices(large_graph.get_num_vertices());
            std::iota(vertices.begin(), vertices.end(), 0);

            std::shuffle(vertices.begin(), vertices.end(), t_gen);

            std::vector<int> current_clique = large_graph.find_greedy_max_clique(vertices);

            std::lock_guard<std::mutex> lock(mtx);
            if (current_clique.size() > best_clique.size()) {
                best_clique = std::move(current_clique);
            }
        });
    }

    executor.run(taskflow).wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    // --- 結果出力 ---
    std::cout << "\n--- Result ---" << std::endl;
    std::cout << "Best clique size found: " << best_clique.size() << std::endl;
    std::cout << "Search time: " << duration.count() << " ms" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " ms" << std::endl;

    std::cout << "Clique vertices: ";
    std::sort(best_clique.begin(), best_clique.end());
    
    // DIMACS形式に合わせて +1 して表示（任意）
    for (size_t i = 0; i < best_clique.size(); ++i) {
        std::cout << (best_clique[i] + 1) << " "; 
    }
    std::cout << std::endl;

    return 0;
}