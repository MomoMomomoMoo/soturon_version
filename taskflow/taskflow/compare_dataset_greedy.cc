/**
 * @file max_clique_dimacs_single.cpp
 * @brief 最大クリーク問題：DIMACSベンチマーク対応 + シングルスレッド（ベースライン）
 * * @details
 * Taskflow版との比較用コードです。
 * 同じDIMACSファイルを読み込み、従来の「次数順ソートによる貪欲法」を1回だけ実行します。
 */

// --- ヘッダファイルのインクルード ---
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream> // 文字列解析用に追加

/**
 * @class Graph
 * @brief グラフ構造を表現するクラス（std::vector版による高速化適用済み）
 */
class Graph {
private:
    int num_vertices_;
    std::vector<std::unordered_set<int>> adj_list_; // 高速なvector版

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
    
    // 貪欲法（1回実行用）
    std::vector<int> find_greedy_max_clique() const {
        if (num_vertices_ == 0) return {};

        std::vector<int> sorted_vertices = get_vertices_sorted_by_degree();
        std::vector<int> clique;
        clique.reserve(100);
        
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

// --- DIMACS形式のファイルを読み込む関数 ---
Graph load_dimacs_graph(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    int num_vertices = 0;
    int num_edges = 0;
    
    // ヘッダー行を探す
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

    // 辺データを読み込む
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

        // 1-based to 0-based
        if (u > 0) u--; 
        if (v > 0) v--;

        g.add_edge(u, v);
    }

    return g;
}

// --- メイン関数 ---
int main() {
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // --- ★変更: ファイル名を指定 ---
    const std::string filename = "C500.9.clq";

    std::cout << "Reading graph file (Single Thread Baseline)..." << std::endl;
    Graph large_graph = load_dimacs_graph(filename);
    std::cout << "Graph loaded.\n" << std::endl;

    std::cout << "Finding max clique (Single greedy run)..." << std::endl;

    // --- クリーク探索 (シングルスレッド・1回のみ) ---
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> max_clique = large_graph.find_greedy_max_clique();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // --- 結果の出力 ---
    std::cout << "\n--- Result (Baseline) ---" << std::endl;
    std::cout << "Found clique with size: " << max_clique.size() << std::endl;
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    
    std::cout << "Clique vertices: ";
    std::sort(max_clique.begin(), max_clique.end());
    
    // Taskflow版と比較しやすいように +1 して表示
    size_t num_to_print = std::min(max_clique.size(), static_cast<size_t>(20));
    for (size_t i = 0; i < num_to_print; ++i) {
        std::cout << (max_clique[i] + 1) << " ";
    }
    if (max_clique.size() > 20) {
        std::cout << "...";
    }
    std::cout << std::endl;
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total program execution time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    return 0;
}