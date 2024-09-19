"""
対象ディレクトリ内の全ての質問データを読み込み、回答を生成するためのテストプログラム
コマンドライン引数で対象ディレクトリを指定
Input: 質問データのディレクトリパス、設定ファイルへのパス
Output: 回答ディレクトリを作成し、回答データおよび設定を保存
"""

import os
import pandas as pd
import json
import argparse
import shutil
from main import test
from RAG_Agriculture.utils.data_load import load_json
from RAG_Agriculture.utils.metrics import ndcg_from_taskname, recall_k_from_taskname, mrr


def parse_arguments():
    # コマンドライン引数を解析する
    parser = argparse.ArgumentParser(description="指定されたディレクトリを作成し、config.jsonをそのディレクトリにコピー")
    parser.add_argument('directory', type=str, help='作成または使用するディレクトリのパス')
    return parser.parse_args()


def create_output_directory(directory):
    # output ディレクトリを作成する
    output_dir = os.path.join(os.path.dirname(directory), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリ {output_dir} を作成しました。")
    else:
        print(f"ディレクトリ {output_dir} は既に存在します。")
    
    return output_dir


def save_json_config(directory, config_data):
    # 指定されたディレクトリにconfig.jsonを保存
    config_path = os.path.join(directory, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)
    print(f"設定ファイルを {config_path} に保存しました。")


def read_txt_files(directory):
    # ディレクトリ内のすべての .txt ファイルを読み込む
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    file_contents = {}
    
    for txt_file in txt_files:
        file_path = os.path.join(directory, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_contents[txt_file] = f.read()
    
    return file_contents


def save_processed_files(output_dir, processed_data):
     # dictデータを別々のファイルに保存
    for file_name, data in processed_data.items():
        # ファイルパスを作成
        file_path = os.path.join(output_dir, f"{file_name}.json")
        
        # ファイルにdictデータを保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"ファイル {file_path} を保存しました。")


def add_evaluation(gt_dir_path:str, processed_data:dict, k=20) -> dict:
    # 評価指標を計算し、データに追加
    for file_name, data in processed_data.items():
        try:
            ids_list = data['searched_ids_list']
            gt_path = os.path.join(gt_dir_path, file_name.replace('.txt', '.csv'))
            gt_ids = pd.read_csv(gt_path)['aao_id'].tolist()
            processed_data[file_name]['GT'] = gt_ids

            # 評価指標を計算
            ndcp = ndcg_from_taskname(ids_list, gt_ids, k=k)
            recall = recall_k_from_taskname(ids_list, gt_ids, k=k)
            mrr_score = mrr(ids_list, gt_ids)

            # データに追加
            processed_data[file_name]['Evaluation'] = {
                f'ndcp@{k}': ndcp,
                f'recall@{k}': recall,
                'mrr': mrr_score
            }
            print(processed_data[file_name]['Evaluation'])
        
            # "Reranked search result: task_name_list"がdataにkeyとして含まれている場合のみの処理
            if 'reranked_ids_list' in data:
                reranked_ids_list = data['reranked_ids_list']
                ndcp = ndcg_from_taskname(reranked_ids_list, gt_ids, k=k)
                recall = recall_k_from_taskname(reranked_ids_list, gt_ids, k=k)
                mrr_score = mrr(reranked_ids_list, gt_ids)
                processed_data[file_name]['Reranked_Evaluation'] = {
                    f'reranked_ndcp@{k}': ndcp,
                    f'reranked_recall@{k}': recall,
                    'reranked_mrr': mrr_score
                }
                print(processed_data[file_name]['Reranked_Evaluation'])
        except Exception as e:
            print(f"Error: {e}")
            print(f"ファイル {file_name} に対する評価指標の計算に失敗しました。")
            continue
    
    return processed_data


if __name__ == '__main__':
    # コマンドライン引数の解析
    args = parse_arguments()
    
    # configファイルの読み込み
    config = load_json('./config.json')

    # 指定されたディレクトリ内の .txt ファイルを読み込む
    file_contents = read_txt_files(args.directory)

    # output ディレクトリを作成する
    output_dir = create_output_directory(args.directory)

    # outputディレクトリにプロンプトファイルをコピー
    extract_prompt_path = config['extract_keywords_prompt_path']
    generate_prompt_path = config['generate_answer_prompt_path']
    # ファイルコピー
    shutil.copy(extract_prompt_path, output_dir)
    shutil.copy(generate_prompt_path, output_dir)

    # 各ファイルの内容を test() に通して処理
    processed_data = {}

    for file_name, data in file_contents.items():
        try:
            processed_data[file_name] = test(data, config)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # 得られた結果を評価し新たに追加
    gt_dir = output_dir.replace('output', 'GT')
    print('------ Evaluation ------')
    processed_data = add_evaluation(gt_dir, processed_data, k=config['metrics_k'])

    # 処理されたデータを output ディレクトリに保存する
    save_processed_files(output_dir, processed_data)

    # output ディレクトリに config.json を保存
    save_json_config(output_dir, config)
