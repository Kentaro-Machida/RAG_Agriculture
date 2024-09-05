"""
ユーザーから入力された質問文からLLMでキーワードを抜き出すためのモジュール
"""
import sys
from openai import OpenAI
from .utils.data_load import load_text, str2json

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
    def __init__(
            self,
            extract_keywords_prompt_path:str,
            retrieve_llm:str
            ):
        self.extract_keywords_prompt = load_text(extract_keywords_prompt_path)
        self.retrieve_llm = retrieve_llm

    def extract_keywords(self, question: str) -> dict:
        if self.retrieve_llm == 'gpt-3.5-turbo':
            client = OpenAI()

            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.extract_keywords_prompt},
                {"role": "user", "content": question}
            ]
            )

            answer = completion.choices[0].message.content
            extracted_keywords_dict = str2json(answer)
        
        else:
            print('Please set the correct LLM model to retrieve_llm in config file for keyword extraction.')
            sys.exit()

        return extracted_keywords_dict
        


def test():
    # テキストデータの読み込み
    text_path = './tests/keywords_extractor/input2.txt'
    with open(text_path, 'r') as f:
        question = f.read()
        
    keyword_extractor = KeywordExtractor(extract_keywords_prompt_path='./prompts/extract_keywords_prompt.txt',retrieve_llm='GPT-3.5-turbo')
    keywords = keyword_extractor.extract_keywords(question)
    print(type(keywords))
    print(keywords)


if __name__ == '__main__':
    test()
