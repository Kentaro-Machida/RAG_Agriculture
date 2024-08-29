"""
キーワード群からLLMを用いてユーザーへの回答を生成するモジュール
"""
import json

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
    

def test():
    # キーワード辞書の読み込み
    test_path = './tests/answer_generator/input.json'
    with open(test_path, 'r') as f:
        keywords = json.load(f)

    answer_generator = DummyAnswerGenerator()
    answer = answer_generator.generate_answer(keywords)
    print(answer)


if __name__ == '__main__':
    test()
