/**
 * @file max_clique_greedy_parallel_optimized.cpp
 * @brief 最大クリーク問題の近似解法（Taskflow並列化 + Vector高速化版）
 * * @details
 * 以前のボトルネックであった unordered_map を std::vector に戻し、
 * ベースの探索速度を極限まで高めた上で、Taskflowによる並列多スタート探索を行います。
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
    // ★修正: unordered_map から vector に戻しました。メモリアクセスが爆速になります。
    std::vector<std::unordered_set<int>> adj_list_;

public:
    Graph(int vertices) : num_vertices_(vertices), adj_list_(vertices) {}

    int get_num_vertices() const { return num_vertices_; }

    // 辺の追加
    void add_edge(int u, int v) {
        adj_list_[u].insert(v);
        adj_list_[v].insert(u);
    }

    // ★修正: 配列アクセスにより計算量 O(1) でリストに到達します。
    // ハッシュ計算のオーバーヘッドが消滅します。
    bool is_adjacent(int u, int v) const {
        return adj_list_[u].count(v);
    }

    // ファイル保存
    void save_to_file_adj_list(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Error: Could not open the file " << filename << std::endl;
            return;
        }
        ofs << "# Vertices: " << num_vertices_ << std::endl;
        for (int i = 0; i < num_vertices_; ++i) {
            ofs << i << ":";
            for (int neighbor : adj_list_[i]) {
                ofs << " " << neighbor;
            }
            ofs << std::endl;
        }
    }

    // 次数順ソート
    std::vector<int> get_vertices_sorted_by_degree() const {
        std::vector<std::pair<int, int>> degrees;
        degrees.reserve(num_vertices_);
        for (int i = 0; i < num_vertices_; ++i) {
            // vectorなので .size() を取得するのも高速です
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
    
    // 貪欲法によるクリーク探索（探索順序を引数で受け取る）
    std::vector<int> find_greedy_max_clique(const std::vector<int>& vertex_order) const {
        std::vector<int> clique;
        clique.reserve(100); // 予想されるサイズ分だけ予約しておくと少し速い

        for (int u : vertex_order) {
            bool can_add = true;
            // 既存のクリーク内全頂点と隣接しているかチェック
            for (int v : clique) {
                if (!is_adjacent(u, v)) { // ここの高速化が効く
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

// グラフ生成関数
Graph create_random_graph(int num_vertices, double edge_probability, std::mt19937& gen) {
    Graph g(num_vertices);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = i + 1; j < num_vertices; ++j) {
            if (dis(gen) < edge_probability) {
                g.add_edge(i, j);
            }
        }
    }
    return g;
}

// --- メイン関数 ---
int main() {
    // 全体の時間計測
    auto total_start = std::chrono::high_resolution_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());

    // --- パラメータ設定 ---
    const int NUM_VERTICES = 1000000;
    const double EDGE_PROBABILITY = 0.1;

    std::cout << "Generating graph (Vertices: " << NUM_VERTICES << ", Prob: " << EDGE_PROBABILITY << ")..." << std::endl;
    Graph large_graph = create_random_graph(NUM_VERTICES, EDGE_PROBABILITY, gen);
    std::cout << "Graph generation complete.\n" << std::endl;

    // --- 並列処理の準備 ---
    std::cout << "Starting parallel greedy search with Taskflow..." << std::endl;
    
    tf::Executor executor;
    tf::Taskflow taskflow;

    std::vector<int> best_clique;
    std::mutex mtx; // 結果書き込み用の排他制御

    // スレッド数（論理コア数）を取得
    const int num_trials = std::thread::hardware_concurrency();
    std::cout << "Concurrent trials: " << num_trials << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // タスク1: 次数順（決定的な貪欲法）
    taskflow.emplace([&]() {
        std::vector<int> initial_order = large_graph.get_vertices_sorted_by_degree();
        std::vector<int> current_clique = large_graph.find_greedy_max_clique(initial_order);

        std::lock_guard<std::mutex> lock(mtx);
        if (current_clique.size() > best_clique.size()) {
            best_clique = std::move(current_clique);
        }
    });

    // タスク2〜N: ランダム順（確率的な貪欲法）
    // hardware_concurrencyの回数分ループしてタスクを追加
    for (int i = 0; i < num_trials - 1; ++i) {
        taskflow.emplace([&]() {
            // 各スレッドで独自の乱数生成器を持つ（シードをずらす）
            std::random_device t_rd;
            std::mt19937 t_gen(t_rd());

            std::vector<int> vertices(large_graph.get_num_vertices());
            std::iota(vertices.begin(), vertices.end(), 0); // 0, 1, 2...

            // シャッフル
            std::shuffle(vertices.begin(), vertices.end(), t_gen);

            // 探索
            std::vector<int> current_clique = large_graph.find_greedy_max_clique(vertices);

            // 結果更新
            std::lock_guard<std::mutex> lock(mtx);
            if (current_clique.size() > best_clique.size()) {
                best_clique = std::move(current_clique);
            }
        });
    }

    // タスクグラフの実行
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

    std::cout << "Clique vertices (first 20): ";
    std::sort(best_clique.begin(), best_clique.end());
    for (size_t i = 0; i < std::min(best_clique.size(), static_cast<size_t>(20)); ++i) {
        std::cout << best_clique[i] << " ";
    }
    if (best_clique.size() > 20) std::cout << "...";
    std::cout << std::endl;

    return 0;
}