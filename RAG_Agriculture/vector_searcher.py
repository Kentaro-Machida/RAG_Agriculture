"""
キーワード群からベクトル検索により関連情報を取得
JSONでキーワードを受け取り、検索結果をJSONもしくはJSONのリストで返す
"""
from transformers import AutoTokenizer, AutoModel
import polars as pl
import json
import os
import sys
from .utils.embedding_process import text_embedding
from .utils.text_preprocess import mE5_preprocess, keywords2text
import weaviate
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

    def __init__(self, embedding_model:str, connection_name:str):
        self.embedding_model = embedding_model
        self.connection_name = connection_name

    def search(self, keywords:dict, n=2)->list:
        keywords_list = []
        str_keywords = json.dumps(keywords)

        if self.embedding_model == 'openai':
            print("Embedding model: openai")
            api_key = os.environ.get('OPENAI_API_KEY')
            headers = {"X-OpenAI-Api-Key": api_key}  # Replace with your OpenAI API key

            try:
                client = weaviate.connect_to_local(headers=headers)

                agris = client.collections.get(self.connection_name)
                
                response = agris.query.near_text(
                    query=str_keywords, limit=n, return_metadata=wq.MetadataQuery(distance=True)
                )

            finally:
                client.close()
        
        elif self.embedding_model == 'intfloat/multilingual-e5-large':
            print("Embedding model: intfloat/multilingual-e5-large")
            model = AutoModel.from_pretrained(self.embedding_model)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            
            try:
                client = weaviate.connect_to_local()
                print(f'------connected to {self.connection_name}------')
                agris = client.collections.get(self.connection_name)

                json_str = keywords2text(keywords)
                me5_input = mE5_preprocess(json_str, 'query')
                embed_vector = text_embedding([me5_input], model, tokenizer).detach().numpy()[0]
                
                response = agris.query.near_vector(
                    near_vector=embed_vector, limit=n
                )

            finally:
                client.close()
        
        else:
            print('Invalid embedding model. Please check embedding_model in the config file')
            sys.exit()
            
        keywords_list = [o.properties for o in response.objects]
        return keywords_list


def test():
    test_path = './tests/vector_searcher/input.json'
    with open(test_path, 'r') as f:
        keywords = json.load(f)
    
    vector_searcher = VectorSearcher(connection_name='Agriculture' ,embedding_model='intfloat/multilingual-e5-large')
    keywords_list = vector_searcher.search(keywords)
    print(keywords_list)


if __name__ == '__main__':
    test()
