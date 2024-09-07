'''
Weaviateを使用して、ベクトルDBを作成するためのスクリプト。
ベクトルDBの作成、データの追加をHuggingface のモデルを使用して行う
'''

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import weaviate
from weaviate.util import generate_uuid5
import pandas as pd

from RAG_Agriculture.utils.data_load import load_json
from RAG_Agriculture.utils.text_preprocess import mE5_preprocess, mE5_preprocess_instruct


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def text_embedding(text: list[str], model, tokenizer) -> Tensor:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    embeddings = average_pool(last_hidden_states, attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def main():
    config_dict = load_json('./config.json')
    embedding_model = config_dict['embedding_model']
    df = pd.read_csv(config_dict['weaviate']['data_path'])

    # Data frame の要素を全て接続し1つの文字列にする
    target_columns = list(config_dict['weaviate']['target_columns'].values())
    target_columns.sort()
    df.fillna('', inplace=True)
    df['joined'] = df.apply(
        lambda row: ', '.join([mE5_preprocess(f'{col}: {row[col]}'.replace('(ja)', ''), 'answer')
                            for col in target_columns]), axis=1)

    # Huggingface のモデルを使用してベクトルを取得
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    model.eval()

    embeddings = text_embedding(df['joined'].tolist(), model, tokenizer).detach().numpy()

    try:
        client = weaviate.connect_to_local()
        client.collections.delete(config_dict['weaviate']['schema']['class'])
        client.collections.create_from_dict(config_dict['weaviate']['schema'])
        print('------collection created------')
        collection = client.collections.get(config_dict['weaviate']['schema']['class'])

        with collection.batch.dynamic() as batch:
            for i, batch_df in tqdm(df.iterrows()):
                obj = {k:batch_df[v] for k,v in config_dict['weaviate']['target_columns'].items()}

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


if __name__ == '__main__':
    main()

# def get_test():
#     collection = client.collections.get(config_dict['weaviate']['schema']['class'])
#     text2 = "鉄コーティングとはなんですか？"
#     embeddings2 = text_embedding([text2], model, tokenizer).detach().numpy()[0]

#     # Weaviateのクエリ機能を使用して、特定のエンベディングに最も近いテキストデータを検索します。
#     query_result = collection.query.near_vector(
#         near_vector=embeddings2, limit=1
#     )

#     # 検索結果を出力します。この結果には、クエリに最も近いテキストデータが含まれます。
#     print(query_result)