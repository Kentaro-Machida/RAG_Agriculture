"""
キーワード群からLLMを用いてユーザーへの回答を生成するモジュール
"""
import json
import sys
from openai import OpenAI
from .utils.data_load import load_text, json2str

class DummyAnswerGenerator:
    def __init__(self):
        pass

    def generate_answer(self, keywords: dict) -> str:
        return "稲の種子を病気や害虫から守るためには、\
            「鉄コーティング」という方法が有効です。\
            この方法では、種子に鉄をまぶすことで、水田での発芽を安定化させ、\
            病気や害虫からの保護が期待されます。\
            鉄コーティングされた種子は、環境ストレスに対しても強くなり、\
            健全な発芽と生育が期待されます。関連URI: http://cavoc.org/aao/ns/4/A51"


class AnswerGenerator:
    def __init__(self, generate_answer_prompt_path:str, generate_llm:str):
        self.generate_answer_prompt = load_text(generate_answer_prompt_path)
        self.generate_llm = generate_llm

    def generate_answer(self, keywords: dict, question:str) -> str:
        keywords_str = json2str(keywords)
        if self.generate_llm == 'gpt-3.5-turbo':
            client = OpenAI()

            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.generate_answer_prompt},
                {"role": "user", "content": keywords_str},
                {"role": "user", "content": question}
            ]
            )

            answer = completion.choices[0].message.content
        else:
            print('Please set the correct LLM model to generate_llm in config file for answer generation.')
            sys.exit()
        
        return answer


def test():
    # キーワード辞書の読み込み
    test_path = './tests/answer_generator/input2.json'
    keywords = json.load(open(test_path, 'r'))

    question_path = './tests/answer_generator/input2.txt'
    question = load_text(question_path)

    answer_generator = AnswerGenerator(
        generate_answer_prompt_path='./prompts/generate_answer_prompt.txt',
        generate_llm='gpt-3.5-turbo'
    )
    answer = answer_generator.generate_answer(keywords, question)
    print(answer)


if __name__ == '__main__':
    test()
