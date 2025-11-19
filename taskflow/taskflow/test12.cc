/**
 * @file max_clique_greedy_parallel.cpp
 * @brief 大規模ランダムグラフの生成と最大クリークの近似解法（TaskFlowによる並列化版）
 *
 * @details
 * このプログラムは、指定された頂点数と辺密度を持つランダムなグラフを生成します。
 * 生成されたグラフに対し、複数の異なる頂点順序を用いて貪欲法を並列実行し、
 * 最大クリークの近似解を探索します。並列処理にはC++ TaskFlowライブラリを使用します。
 */

// --- ヘッダファイルのインクルード ---
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <numeric> // std::iota のためにインクルード
#include <mutex>   // ★変更点: 並列実行時の共有リソース保護のためにインクルード
#include <functional> // ★変更点: std::function を使うためにインクルード

// C++ TaskFlowライブラリのヘッダ
// GitHubなどからダウンロードし、インクルードパスを通してください。
// https://github.com/taskflow/taskflow
#include "taskflow.hpp"
/**
 * @class Graph
 * @brief グラフ構造を表現し、関連する操作をカプセル化するクラス
 */
class Graph {
private:
    int num_vertices_;
    std::unordered_map<int, std::unordered_set<int>> adj_list_;

public:
    Graph(int vertices) : num_vertices_(vertices) {}

    int get_num_vertices() const { return num_vertices_; }

    void add_edge(int u, int v) {
        adj_list_[u].insert(v);
        adj_list_[v].insert(u);
    }

    bool is_adjacent(int u, int v) const {
        return adj_list_.count(u) && adj_list_.at(u).count(v);
    }

    void save_to_file_adj_list(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Error: Could not open the file " << filename << std::endl;
            return;
        }

        ofs << "# Vertices: " << num_vertices_ << std::endl;

        for (int i = 0; i < num_vertices_; ++i) {
            ofs << i << ":";
            if (adj_list_.count(i)) {
                for (int neighbor : adj_list_.at(i)) {
                    ofs << " " << neighbor;
                }
            }
            ofs << std::endl;
        }
    }

    std::vector<int> get_vertices_sorted_by_degree() const {
        std::vector<std::pair<int, int>> degrees;
        for (int i = 0; i < num_vertices_; ++i) {
            int degree = adj_list_.count(i) ? adj_list_.at(i).size() : 0;
            degrees.push_back({-degree, i}); // 降順ソートのため次数を負にする
        }
        std::sort(degrees.begin(), degrees.end());

        std::vector<int> sorted_vertices;
        sorted_vertices.reserve(num_vertices_);
        for (const auto& p : degrees) {
            sorted_vertices.push_back(p.second);
        }
        return sorted_vertices;
    }
    
    /**
     * @brief ★変更点: 指定された頂点順序で貪欲法を実行する
     * @param vertex_order 探索に使用する頂点の順序
     * @return std::vector<int> 見つかったクリークを構成する頂点のリスト
     */
    std::vector<int> find_greedy_max_clique(const std::vector<int>& vertex_order) const {
        std::vector<int> clique;
        
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
    std::random_device rd;
    std::mt19937 gen(rd());

    // --- グラフ生成のパラメータ設定 ---
    const int NUM_VERTICES = 10000;
    const double EDGE_PROBABILITY = 0.1;

    std::cout << "Generating a large random graph..." << std::endl;
    std::cout << "Vertices: " << NUM_VERTICES << ", Edge Probability: " << EDGE_PROBABILITY << std::endl;
    
    Graph large_graph = create_random_graph(NUM_VERTICES, EDGE_PROBABILITY, gen);

    const std::string filename = "graph_adj_list1.txt";
    large_graph.save_to_file_adj_list(filename);
    std::cout << "Graph saved to " << filename << std::endl;

    std::cout << "\nGraph generated. Now finding max clique using parallel greedy search..." << std::endl;

    // --- ★変更点: TaskFlowを用いた並列探索 ---

    // 1. TaskFlow Executorを準備 (利用可能なハードウェアのスレッド数で初期化)
    tf::Executor executor;
    tf::Taskflow taskflow;

    // 2. 最良のクリークを格納する変数と、それを保護するミューテックス
    std::vector<int> best_clique;
    std::mutex mtx;

    // 3. 並列で試行する回数 (ハードウェアの並列度などに合わせると良い)
    const int num_trials = std::thread::hardware_concurrency();
    std::cout << "Running " << num_trials << " trials in parallel..." << std::endl;

    // クリーク探索と実行時間の計測開始
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 最初の1回は、効果的なヒューリスティックである「次数降順ソート」で試す
    std::vector<int> initial_order = large_graph.get_vertices_sorted_by_degree();
    best_clique = large_graph.find_greedy_max_clique(initial_order);


    // 4. 残りの回数分、ランダムな順序で探索するタスクを生成
    std::function<void(int)> worker = [&](int /*i*/) {

        // 各タスク（スレッド）で独立した乱数生成器を使う
        std::random_device thread_rd;
        std::mt19937 thread_gen(thread_rd());
        
        std::vector<int> vertices(large_graph.get_num_vertices());
        std::iota(vertices.begin(), vertices.end(), 0);

        std::shuffle(vertices.begin(), vertices.end(), thread_gen);
        
        std::vector<int> current_clique = large_graph.find_greedy_max_clique(vertices);
        
        std::lock_guard<std::mutex> lock(mtx);
        if (current_clique.size() > best_clique.size()) {
            best_clique = std::move(current_clique);
        }
    };
    taskflow.for_each_index(0, num_trials - 1, 1, worker);
    // 5. タスクを実行し、完了を待つ
    executor.run(taskflow).wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // --- 結果の出力 ---
    std::cout << "\nFound best clique with size: " << best_clique.size() << std::endl;
    std::cout << "Time taken to find the clique: " << duration.count() << " ms" << std::endl;
    
    std::cout << "Clique vertices (first 20): ";
    size_t num_to_print = std::min(best_clique.size(), static_cast<size_t>(20));
    std::sort(best_clique.begin(), best_clique.end()); // 見やすくするためにソート
    for (size_t i = 0; i < num_to_print; ++i) {
        std::cout << best_clique[i] << " ";
    }
    if (best_clique.size() > 20) {
        std::cout << "...";
    }
    std::cout << std::endl;

    return 0;
}