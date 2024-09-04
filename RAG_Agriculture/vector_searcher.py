"""
キーワード群からベクトル検索により関連情報を取得
JSONでキーワードを受け取り、検索結果をJSONもしくはJSONのリストで返す
"""
import polars as pl
import json
import os
from utils.data_load import load_json, json2str
from openai import OpenAI
import weaviate
import weaviate.classes.config as wc
import weaviate.classes.query as wq


class DummyVectorSearcher:
    def __init__(self):
        pass

    def search(self, keywords:dict, n=10)->list:
        '''
        Input: Key words dict with blanck
        Output List of Key words dicts filled with related information
        '''
        return [
            {
                "PURPOSE": "発芽安定化",
                "ACTION": "まぶす",
                "TARGET": "種子",
                "SUBTARGET": "",
                "LOCATION": "水田",
                "METHOD": "",
                "MATERIAL": "鉄",
                "CROP_EXAMPLE": "イネ",
                "REGION": "",
                "CONDITION": "",
                "TASK_NAME": "鉄コーティング",
                "URI": "http://cavoc.org/aao/ns/4/A51"
            }
        ] * n


class VectorSearcher:

    def __init__(self, embedding_model:str):
        self.embedding_model = embedding_model

    def search(self, keywords:dict, n=2)->list:
        keywords_list = []
        str_keywords = json.dumps(keywords)

        if self.embedding_model == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
            headers = {"X-OpenAI-Api-Key": api_key}  # Replace with your OpenAI API key

            try:
                client = weaviate.connect_to_local(headers=headers)

                agris = client.collections.get("Agriculture")
                
                response = agris.query.near_text(
                    query=str_keywords, limit=n, return_metadata=wq.MetadataQuery(distance=True)
                )

            finally:
                client.close()
            
            keywords_list = [o.properties for o in response.objects]
            return keywords_list


def test():
    test_path = './tests/vector_searcher/input.json'
    with open(test_path, 'r') as f:
        keywords = json.load(f)
    
    vector_searcher = VectorSearcher(embedding_model='openai')
    keywords_list = vector_searcher.search(keywords)
    print(keywords_list)


if __name__ == '__main__':
    test()
