"""
キーワード群からベクトル検索により関連情報を取得
JSONでキーワードを受け取り、検索結果をJSONもしくはJSONのリストで返す
"""
import polars as pl
import json

class DummyVectorSearcher:
    def __init__(self):
        pass

    def search(self, keywords:dict, n=10)->list:
        '''
        Input: Key words dict with black
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


def test():
    test_path = './tests/vector_searcher/input.json'
    with open(test_path, 'r') as f:
        keywords = json.load(f)
    
    vector_searcher = DummyVectorSearcher()
    result = vector_searcher.search(keywords)
    print(result)


if __name__ == '__main__':
    test()