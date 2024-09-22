'''
Weaviateを使用して、ベクトルDBを作成するためのスクリプト。
ベクトルDBの作成、データの追加をHuggingface のモデルを使用して行う
'''

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import weaviate
from weaviate.util import generate_uuid5
import pandas as pd

from RAG_Agriculture.utils.data_load import load_json
from RAG_Agriculture.utils.text_preprocess import mE5_preprocess
from RAG_Agriculture.utils.embedding_process import text_embedding


def joined_embedding(config_dict:dict):
    embedding_model = config_dict['embedding_model']
    df = pd.read_csv(config_dict['weaviate']['data_path'])

    # Data frame の要素を全て接続し1つの文字列にする
    vector_columns = list(config_dict['weaviate']['vector_columns'].values())
    vector_columns.sort()
    df.fillna('', inplace=True)
    # 対称の列の要素を列名と共に一つの文字列に結合
    df['joined'] = df.apply(
        lambda row: ', '.join([mE5_preprocess(f'{col}: {row[col]}'.replace('(ja)', ''), 'answer')
                            for col in vector_columns]), axis=1)

    # Huggingface のモデルを使用してベクトルを取得
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    model.eval()

    embeddings = text_embedding(df['joined'].tolist(), model, tokenizer).detach().numpy()

    # weaviate にアップロードするカラム辞書を作成
    vector_columns = config_dict['weaviate']['vector_columns']
    rule_based_columns = config_dict['weaviate']['rule_based_columns']
    prompt_only_columns = config_dict['weaviate']['prompt_only_columns']
    weaviate_columns = {**vector_columns, **rule_based_columns, **prompt_only_columns}

    try:
        client = weaviate.connect_to_local()
        client.collections.delete(config_dict['weaviate']['schema']['class'])
        client.collections.create_from_dict(config_dict['weaviate']['schema'])
        print('------collection created------')
        collection = client.collections.get(config_dict['weaviate']['schema']['class'])

        with collection.batch.dynamic() as batch:
            for i, batch_df in tqdm(df.iterrows()):
                obj = {k:batch_df[v] for k,v in weaviate_columns.items()}

                batch.add_object(
                    properties=obj,
                    uuid=generate_uuid5(batch_df['作業名']),
                    vector=embeddings[i].tolist()
                )

        if len(collection.batch.failed_objects) > 0:
            print(collection.batch.failed_objects)
            print(f"Failed to import {len(collection.batch.failed_objects)} objects")
    finally:
        client.close()



def search_test():
    config_dict = load_json('./config.json')
    embedding_model = config_dict['embedding_model']
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    model.eval()
    client = weaviate.connect_to_local()
    collection = client.collections.get(config_dict['weaviate']['schema']['class'])
    print(f'------connected to {config_dict['weaviate']['schema']['class']}')
    text2 = "query: ACTION:  CONDITION:  CROP_EXAMPLE: イネ LOCATION: 水田 MATERIAL:  METHOD:  PURPOSE: 発芽安定化 REGION:  SUBTARGET:  TARGET: 種子 TASK_NAME:  URI:  ？"
    embeddings2 = text_embedding([text2], model, tokenizer).detach().numpy()[0]
    print(len(embeddings2))

    # Weaviateのクエリ機能を使用して、特定のエンベディングに最も近いテキストデータを検索します。
    query_result = collection.query.near_vector(
        near_vector=embeddings2, limit=1
    )
    client.close()

    # 検索結果を出力します。この結果には、クエリに最も近いテキストデータが含まれます。
    print(query_result)

if __name__ == '__main__':
    config_dict = load_json('./config.json')
    joined_embedding(config_dict)
    search_test()

