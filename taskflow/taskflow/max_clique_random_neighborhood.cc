/**
 * @file max_clique_random_neighborhood_repeat.cpp
 * @brief 最大クリーク問題：ランダムシード近傍探索（複数回実行・統計収集版）
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
        return (u >= 0 && u < num_vertices_) ? adj_list_[u].size() : 0;
    }

    // 指定した頂点の「近傍」だけを返す
    std::vector<int> get_neighbors(int u) const {
        if (u < 0 || u >= num_vertices_) return {};
        std::vector<int> neighbors(adj_list_[u].begin(), adj_list_[u].end());
        return neighbors;
    }

    // 部分集合の中での貪欲法（次数順）
    std::vector<int> find_greedy_clique_in_subset(std::vector<int>& candidates) const {
        // 部分グラフ内での有望度（元のグラフでの次数）順にソート
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
    // --- 設定 ---
    const std::string filename = "C500.9.clq"; 
    const int NUM_EXPERIMENTS = 10;      // 実験を繰り返す回数
    const int NUM_TRIALS_PER_RUN = 10000; // 1回の実験で行うTaskflowの試行回数

    // --- 1. グラフ読み込み ---
    std::cout << "Reading graph file..." << std::endl;
    Graph graph = load_dimacs_graph(filename);
    std::cout << "Graph loaded successfully.\n" << std::endl;

    std::cout << "Starting Random Neighborhood Search experiments (" << NUM_EXPERIMENTS << " runs)..." << std::endl;
    std::cout << "Trials per run: " << NUM_TRIALS_PER_RUN << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    tf::Executor executor;
    
    // 統計用データ
    std::vector<int> results;
    std::vector<double> times;
    
    int total_vertices = graph.get_num_vertices();

    // --- 2. 実験ループ ---
    for (int run = 0; run < NUM_EXPERIMENTS; ++run) {
        
        tf::Taskflow taskflow;
        std::vector<int> best_clique; 
        std::mutex mtx; 

        auto start_time = std::chrono::high_resolution_clock::now();

        // 並列ループ生成
        for (int i = 0; i < NUM_TRIALS_PER_RUN; ++i) {
            taskflow.emplace([&]() {
                // 1. ランダムな点（シード）を決める
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dist(0, total_vertices - 1);
                
                int seed_vertex = dist(gen);

                // 2. 幅優先探索（深さ1）を行い、小さな部分グラフを作る
                std::vector<int> subgraph_nodes = graph.get_neighbors(seed_vertex);

                // 3. その中で最大クリークを探す（貪欲法）
                std::vector<int> local_clique = graph.find_greedy_clique_in_subset(subgraph_nodes);

                // シード頂点を追加
                local_clique.push_back(seed_vertex);

                // 4. 結果更新
                if (local_clique.size() > best_clique.size()) {
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
    std::cout << "Summary (Random Neighborhood / " << NUM_EXPERIMENTS << " runs):" << std::endl;
    std::cout << "  Max Size : " << max_size << std::endl;
    std::cout << "  Min Size : " << min_size << std::endl;
    std::cout << "  Avg Size : " << avg_size << std::endl;
    std::cout << "  Avg Time : " << avg_time << " ms" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    return 0;
}