"""
JSONファイルに保存されたmetricsを読み込み指標ごとに平均値を計算
計算された平均値の値は新たにJSONファイルとして保存
"""

from data_load import load_json
import json
import os
import argparse

def parse_arguments():
    # コマンドライン引数を解析する
    parser = argparse.ArgumentParser(description="指定されたディレクトリを作成し、config.jsonをそのディレクトリにコピー")
    parser.add_argument('directory', type=str, help='作成または使用するディレクトリのパス')
    return parser.parse_args()

if __name__ == '__main__':
    # コマンドライン引数の解析
    args = parse_arguments()
    
    output_dir = args.directory

    # output_dir を探索し、.jsonファイルを読み込みパスを取得
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    json_paths = [os.path.join(output_dir, f) for f in json_files]

    # config.jsonは除外
    json_paths = [path for path in json_paths if 'config.json' not in path]

    # 各ファイルのmetricsを読み込み
    recall_values = 0
    ndcg_values = 0
    mrr_values = 0
    reranked_recall_values = 0
    reranked_ndcg_values = 0
    reranked_mrr_values = 0

    for json_path in json_paths:
        base_dict = load_json(json_path)
        result_dict = base_dict['Evaluation']
        evaluation_keys = result_dict.keys()
        recall_keys = [key for key in evaluation_keys if key.startswith('recall@')][0]
        k = int(recall_keys.split('@')[-1])

        recall_values += result_dict[f'recall@{k}']
        ndcg_values += result_dict[f'ndcp@{k}']
        mrr_values += result_dict['mrr']
        # もしkeysの中にreranked_recall@kがあれば、rerankedの指標も計算
        if 'Reranked_Evaluation' in base_dict.keys():
            reranked_result_dict = base_dict['Reranked_Evaluation']
            reranked_recall_values += reranked_result_dict[f'reranked_recall@{k}']
            reranked_ndcg_values += reranked_result_dict[f'reranked_ndcp@{k}']
            reranked_mrr_values += reranked_result_dict['reranked_mrr']

    # 平均値を計算
    num_files = len(json_paths)
    mean_recall = recall_values / num_files
    mean_ndcg = ndcg_values / num_files
    mean_mrr = mrr_values / num_files
    mean_reranked_recall = reranked_recall_values / num_files
    mean_reranked_ndcg = reranked_ndcg_values / num_files
    mean_reranked_mrr = reranked_mrr_values / num_files

    # 平均値を保存
    mean_metrics = {
        f'mean_recall@{k}': mean_recall,
        f'mean_ndcg@{k}': mean_ndcg,
        f'mean_mrr': mean_mrr,
        f'mean_reranked_recall@{k}': mean_reranked_recall,
        f'mean_reranked_ndcg@{k}': mean_reranked_ndcg,
        f'mean_reranked_mrr': mean_reranked_mrr
    }

    # 保存
    mean_metrics_path = os.path.join(output_dir, 'mean_metrics.json')
    with open(mean_metrics_path, 'w') as f:
        json.dump(mean_metrics, f, indent=4, ensure_ascii=False)

    print(f"Mean metrics are saved in {mean_metrics_path}")
