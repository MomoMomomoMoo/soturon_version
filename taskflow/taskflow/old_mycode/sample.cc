#include <iostream>
#include <string>

// ★重要: ダウンロードした単一ヘッダー版の "taskflow.hpp"
#include "taskflow.hpp"

int main() {

    // 1. Executor（実行エンジン）と Taskflow（グラフ）を準備
    tf::Executor executor;
    tf::Taskflow taskflow;

    // 2. タスクを定義
    // emplace はタスクを作成し、そのタスクハンドルを返す
    tf::Task A = taskflow.emplace([]() { 
        std::cout << "Running Task A" << std::endl; 
    }).name("Task A"); // .name() はデバッグ用の名前（任意）

    tf::Task B = taskflow.emplace([]() { 
        std::cout << "Running Task B" << std::endl; 
    }).name("Task B");

    // 3. 依存関係を設定 (A が B の前に実行される)
    A.precede(B);

    std::cout << "Taskflow setup complete. Running..." << std::endl;

    // 4. タスクフローを実行し、完了を待つ
    executor.run(taskflow).wait();

    std::cout << "Taskflow execution finished." << std::endl;

    return 0;
}