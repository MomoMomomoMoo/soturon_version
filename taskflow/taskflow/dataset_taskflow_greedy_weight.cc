/**
 * @file max_clique_weighted_random.cpp
 * @brief 最大クリーク問題：DIMACS対応 + Taskflow並列 + 重み付きランダムソート
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

    // ★追加: 特定の頂点の次数を取得する関数
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
    // --- ファイル設定 ---
    const std::string filename = "C500.9.clq"; // または C125.9.clq

    std::cout << "Reading graph file..." << std::endl;
    Graph large_graph = load_dimacs_graph(filename);
    std::cout << "Graph loaded successfully.\n" << std::endl;

    // --- 事前に全頂点の次数を計算しておく（高速化のため） ---
    // これを各タスク内でやると遅くなるので、ここで一回計算して使い回します
    std::vector<int> degrees(large_graph.get_num_vertices());
    for(int i=0; i<large_graph.get_num_vertices(); ++i) {
        degrees[i] = large_graph.get_degree(i);
    }

    // --- Taskflow準備 ---
    std::cout << "Starting parallel greedy search with Weighted Random Sort..." << std::endl;
    
    tf::Executor executor;
    tf::Taskflow taskflow;

    std::vector<int> best_clique;
    std::mutex mtx; 

    // taskflow内で実行する試行回数
    const int num_trials = 10000; 

    std::cout << "Trials: " << num_trials << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // タスク1: 純粋な次数順（ベースラインとして必ず入れる）
    taskflow.emplace([&]() {
        std::vector<int> initial_order = large_graph.get_vertices_sorted_by_degree();
        std::vector<int> current_clique = large_graph.find_greedy_max_clique(initial_order);

        std::lock_guard<std::mutex> lock(mtx);
        if (current_clique.size() > best_clique.size()) {
            best_clique = std::move(current_clique);
        }
    });

    // タスク2〜N: ★重み付きランダム（ノイズ付きソート）
    // ループごとに異なる「揺らぎ」を与えて多様な解を探す
    for (int i = 0; i < num_trials - 1; ++i) {
        taskflow.emplace([&, i]() { // degreesを参照キャプチャ
            std::random_device t_rd;
            std::mt19937 t_gen(t_rd());
            
            // ノイズの幅を設定（次数の±10%〜20%くらいが目安）
            // C500.9の次数は約450なので、±50くらいの幅を持たせる
            std::uniform_real_distribution<> noise_dist(-50.0, 50.0);

            // ソート用のペア配列 (スコア, 頂点ID)
            std::vector<std::pair<double, int>> weighted_vertices;
            weighted_vertices.reserve(degrees.size());

            for(size_t v = 0; v < degrees.size(); ++v) {
                // スコア = 本来の次数 + ランダムな揺らぎ
                double score = degrees[v] + noise_dist(t_gen);
                weighted_vertices.push_back({score, static_cast<int>(v)});
            }

            // スコアが高い順（降順）にソート
            std::sort(weighted_vertices.rbegin(), weighted_vertices.rend());

            // 頂点IDだけのリストに変換
            std::vector<int> search_order;
            search_order.reserve(degrees.size());
            for(const auto& p : weighted_vertices) {
                search_order.push_back(p.second);
            }

            // 探索実行
            std::vector<int> current_clique = large_graph.find_greedy_max_clique(search_order);

            // 結果更新
            std::lock_guard<std::mutex> lock(mtx);
            if (current_clique.size() > best_clique.size()) {
                best_clique = std::move(current_clique);
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