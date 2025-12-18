/**
 * @file max_clique_neighborhood_parallel_v2.cpp
 * @brief 最大クリーク問題：近傍分解による並列化（for_each_index不使用版）
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
#include <mutex>
#include <thread>
#include <atomic>

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

    // 近傍リストを返す
    std::vector<int> get_neighbors(int u) const {
        if (u < 0 || u >= num_vertices_) return {};
        std::vector<int> neighbors(adj_list_[u].begin(), adj_list_[u].end());
        return neighbors;
    }

    // 部分集合の中での貪欲法
    std::vector<int> find_greedy_clique_in_subset(std::vector<int>& candidates) const {
        // 次数順ソート（高速化のためグローバル次数利用）
        std::sort(candidates.begin(), candidates.end(), [this](int a, int b) {
            return adj_list_[a].size() > adj_list_[b].size();
        });

        std::vector<int> clique;
        clique.reserve(candidates.size());

        for (int u : candidates) {
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
            std::string temp;
            ss >> temp >> temp >> num_vertices;
            break;
        }
    }
    if (num_vertices == 0) exit(1);
    std::cout << "Loading DIMACS graph: " << filename << " (Vertices: " << num_vertices << ")" << std::endl;
    Graph g(num_vertices);
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == 'c') continue;
        std::stringstream ss(line);
        char type; 
        int u, v;
        if (line[0] == 'e') ss >> type >> u >> v;
        else { std::stringstream tss(line); if(!(tss >> u >> v)) continue; }
        if (u > 0) u--; if (v > 0) v--;
        g.add_edge(u, v);
    }
    return g;
}

// --- メイン関数 ---
int main() {
    // ファイル名（適宜変更してください）
    const std::string filename = "C500.9.clq"; 
    
    std::cout << "Reading graph..." << std::endl;
    Graph graph = load_dimacs_graph(filename);

    std::cout << "Starting parallel neighborhood search..." << std::endl;

    tf::Executor executor;
    tf::Taskflow taskflow;

    std::vector<int> best_clique;
    std::mutex mtx;
    
    int total_vertices = graph.get_num_vertices();

    auto start_time = std::chrono::high_resolution_clock::now();

    // ★修正箇所: for_each_index をやめて、通常のループ + emplace に変更
    // これなら確実に動きます
    for (int i = 0; i < total_vertices; ++i) {
        // [&, i] で i を値としてキャプチャするのが重要です！
        // & だけだと、ループが終わったあとの i を参照してバグります。
        taskflow.emplace([&, i]() {
            int u = i; // 頂点u = ループ変数のi

            // 1. 頂点 u の近傍を取得（コンパクト化）
            std::vector<int> neighbors = graph.get_neighbors(u);

            // 2. 近傍内だけで貪欲法
            std::vector<int> local_clique = graph.find_greedy_clique_in_subset(neighbors);

            // 3. 自分自身を追加
            local_clique.push_back(u);

            // 4. 結果更新（排他制御）
            if (local_clique.size() > best_clique.size()) { // ロック前の軽いチェック
                std::lock_guard<std::mutex> lock(mtx);
                if (local_clique.size() > best_clique.size()) {
                    best_clique = std::move(local_clique);
                }
            }
        });
    }

    executor.run(taskflow).wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // --- 結果出力 ---
    std::cout << "\n--- Result ---" << std::endl;
    std::cout << "Best clique size found: " << best_clique.size() << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;

    std::cout << "Clique vertices: ";
    std::sort(best_clique.begin(), best_clique.end());
    for (size_t i = 0; i < best_clique.size(); ++i) {
        std::cout << (best_clique[i] + 1) << " ";
    }
    std::cout << std::endl;

    return 0;
}