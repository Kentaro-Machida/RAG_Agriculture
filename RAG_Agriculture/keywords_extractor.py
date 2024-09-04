"""
ユーザーから入力された質問文からLLMでキーワードを抜き出すためのモジュール
"""
from openai import OpenAI

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


class KeywordExtractor:
    def __init__(self, config:dict):
        self.initial_prompt = config['initial_prompt']
        self.retrieve_llm = config['retrieve_llm']

    def extract_keywords(self, question: str) -> dict:
        if self.retrieve_llm == 'GPT-3.5-turbo':
            client = OpenAI()

            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.initial_prompt},
                {"role": "user", "content": "日本の首都を漢字2文字で教えてください。"}
            ]
            )

            answer = completion.choices[0].message
        
        return extracted_keywords_dict
        


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
