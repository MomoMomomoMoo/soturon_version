/**
 * @file max_clique_greedy_mt.cpp
 * @brief 大規模ランダムグラフの生成と最大クリークの近似解法（メルセンヌ・ツイスター版）
 *
 * @details
 * このプログラムは、指定された頂点数と辺密度を持つランダムなグラフを生成します。
 * 乱数生成には高品質なメルセンヌ・ツイスター法を使用しています。
 * 生成されたグラフに対し、貪欲法（Greedy Algorithm）を用いて最大クリークの近似解を求めます。
 * また、生成したグラフの隣接リスト表現をファイルに出力する機能も備えています。
 */

// --- ヘッダファイルのインクルード ---
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <random>       // ★変更点: 高品質な乱数生成のためにインクルード

/**
 * @class Graph
 * @brief グラフ構造を表現し、関連する操作（辺の追加、クリーク探索など）をカプセル化するクラス
 */
class Graph {
private:
    int num_vertices_;
    std::vector<std::unordered_set<int>> adj_list_; // 隣接リスト表現 (変更なし)

public:
    /**
     * @brief Graphクラスのコンストラクタ
     * @param vertices 生成するグラフの頂点数
     */
    Graph(int vertices) : num_vertices_(vertices), adj_list_(vertices) {}

    /**
     * @brief 2頂点間に無向の辺を追加する
     * @param u 頂点u
     * @param v 頂点v
     */
    void add_edge(int u, int v) {
        adj_list_[u].insert(v);
        adj_list_[v].insert(u);
    }

    /**
     * @brief 2頂点が隣接しているか（辺で繋がっているか）を判定する
     * @param u 頂点u
     * @param v 頂点v
     * @return bool 隣接していれば true, そうでなければ false
     */
    bool is_adjacent(int u, int v) const {
        return adj_list_[u].count(v);
    }
    
    /**
     * @brief グラフの構造を隣接リスト形式でファイルに保存する
     * @param filename 保存先のファイル名
     */
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

    /**
     * @brief 全ての頂点を次数の降順（接続する辺の数が多い順）にソートして返す
     * @return std::vector<int> 次数でソートされた頂点のリスト
     */
    std::vector<int> get_vertices_sorted_by_degree() const {
        std::vector<std::pair<int, int>> degrees;
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
    
    /**
     * @brief 貪欲法（Greedy Algorithm）を用いて最大クリークの近似解を求める
     * @return std::vector<int> 見つかったクリークを構成する頂点のリスト
     */
    std::vector<int> find_greedy_max_clique() const {
        if (num_vertices_ == 0) return {};

        std::vector<int> sorted_vertices = get_vertices_sorted_by_degree();
        std::vector<int> clique;
        
        for (int u : sorted_vertices) {
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

/**
 * @brief 指定されたパラメータに基づいてランダムグラフを生成する
 * @param num_vertices グラフの頂点数
 * @param edge_probability 2頂点間に辺が存在する確率
 * @param gen ★変更点: メルセンヌ・ツイスター乱数生成器への参照を受け取る
 * @return Graph 生成されたグラフオブジェクト
 */
Graph create_random_graph(int num_vertices, double edge_probability, std::mt19937& gen) {
    Graph g(num_vertices);
    // ★変更点: 0.0から1.0までの一様実数分布を定義
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = i + 1; j < num_vertices; ++j) {
            // ★変更点: 古いrand()の代わりに、高品質な乱数生成器で確率判定
            if (dis(gen) < edge_probability) {
                g.add_edge(i, j);
            }
        }
    }
    return g;
}


// --- メイン関数 ---
int main() {
    // --- ★追加: プログラム全体の実行時間計測を開始 ---
    auto total_start_time = std::chrono::high_resolution_clock::now();
    // ★変更点: srand(time(0)) を削除し、より高品質な乱数生成器をセットアップ
    // 1. 非決定論的な乱数生成器を用いて、実行ごとに異なるシード値を生成
    std::random_device rd;
    // 2. メルセンヌ・ツイスターエンジンをそのシードで初期化
    std::mt19937 gen(rd());

    // --- グラフ生成のパラメータ設定 ---
    const int NUM_VERTICES = 1000000;
    const double EDGE_PROBABILITY = 0.1;

    std::cout << "Generating a large random graph..." << std::endl;
    std::cout << "Vertices: " << NUM_VERTICES << ", Edge Probability: " << EDGE_PROBABILITY << std::endl;
    
    // ★変更点: 乱数生成器 `gen` を関数に渡してグラフを生成
    Graph large_graph = create_random_graph(NUM_VERTICES, EDGE_PROBABILITY, gen);

    // --- グラフのファイル保存 ---
    //const std::string filename = "graph_adj_list.txt";
    //large_graph.save_to_file_adj_list(filename);
    //std::cout << "Graph saved to " << filename << std::endl;

    std::cout << "\nGraph generated. Now finding max clique..." << std::endl;

    // --- クリーク探索と実行時間の計測 ---
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> max_clique = large_graph.find_greedy_max_clique();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);


    // --- 結果の出力 ---
    std::cout << "\nFound clique with size: " << max_clique.size() << std::endl;
    std::cout << "Time taken to find the clique: " << duration.count() << " ms" << std::endl;
    
    std::cout << "Clique vertices (first 20): ";
    size_t num_to_print = std::min(max_clique.size(), static_cast<size_t>(20));
    for (size_t i = 0; i < num_to_print; ++i) {
        std::cout << max_clique[i] << " ";
    }
    if (max_clique.size() > 20) {
        std::cout << "...";
    }
    std::cout << std::endl;
    
    // --- ★追加: プログラム全体の実行時間計測を終了し、結果を表示 ---
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Total program execution time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    return 0;
}