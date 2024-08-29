"""
実行したら対話モードに入り、農業系の質問をすると、
知識グラフを参照した答えを返すプログラム
"""
from RAG_Agriculture import answer_generator as ag, keywords_extractor as ke, vector_searcher as vs


def test():
    # テキストデータの読み込み
    text_path = './tests/main/input.txt'
    with open(text_path, 'r') as f:
        question = f.read()
    
    print("Question:", question)
    keyword_extractor = ke.DummyKeywordExtractor()
    keywords = keyword_extractor.extract_keywords(question)
    
    vector_searcher = vs.DummyVectorSearcher()
    result = vector_searcher.search(keywords)
    
    answer_generator = ag.DummyAnswerGenerator()
    answer = answer_generator.generate_answer(result)
    
    print("Answer:", answer)


if __name__ == '__main__':
    test()