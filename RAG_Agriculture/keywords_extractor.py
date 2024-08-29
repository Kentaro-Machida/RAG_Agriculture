"""
ユーザーから入力された質問文からLLMでキーワードを抜き出すためのモジュール
"""
class DummyKeywordExtractor:
    def __init__(self):
        pass

    def extract_keywords(self, question: str) -> dict:
        return {
            "PURPOSE": "発芽安定化",
            "ACTION": "",
            "TARGET": "種子",
            "SUBTARGET": "",
            "LOCATION": "水田",
            "METHOD": "",
            "MATERIAL": "",
            "CROP_EXAMPLE": "イネ",
            "REGION": "",
            "CONDITION": "",
            "TASK_NAME": "",
            "URI": ""
        }
    

def test():
    # テキストデータの読み込み
    text_path = './tests/keywords_extractor/input.txt'
    with open(text_path, 'r') as f:
        question = f.read()
        
    keyword_extractor = DummyKeywordExtractor()
    keywords = keyword_extractor.extract_keywords(question)
    print(keywords)


if __name__ == '__main__':
    test()
