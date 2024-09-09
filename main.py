"""
実行したら対話モードに入り、農業系の質問をすると、
ベクトルDBから関連する情報を取得し、回答を生成して返す。
"""
import requests
from RAG_Agriculture import answer_generator as ag, keywords_extractor as ke, vector_searcher as vs
from RAG_Agriculture.utils.data_load import load_json, add_lang_to_promptpath
from langdetect import detect

# # load config.json
# config = load_json('./config.json')
# retrieve_llm = config['retrieve_llm']
# extract_keywords_prompt_path = config['extract_keywords_prompt_path']
# generate_llm = config['generate_llm']
# generate_answer_prompt_path = config['generate_answer_prompt_path']
# embedding_model = config['embedding_model']
# search_num = config['search_num']

def main():

    # Input text by conversation system
    question = input("Please input your question: ")
    lang = detect(question)

    # Extract keywords from the question
    keyword_extractor = ke.KeywordExtractor(extract_keywords_prompt_path, retrieve_llm)
    keywords = keyword_extractor.extract_keywords(question)

    # Search the knowledge graph
    vector_searcher = vs.VectorSearcher(embedding_model, connection_name=config['weaviate']['schema']['class'])
    keywords_list = vector_searcher.search(keywords, n=search_num)

    # Generate answer
    answer_generator = ag.AnswerGenerator(generate_answer_prompt_path, generate_llm)
    answer = answer_generator.generate_answer(keywords_list, question)

    return answer


def test():
    # load config.json
    config = load_json('./config.json')
    retrieve_llm = config['retrieve_llm']
    extract_keywords_prompt_path = config['extract_keywords_prompt_path']
    generate_llm = config['generate_llm']
    generate_answer_prompt_path = config['generate_answer_prompt_path']
    embedding_model = config['embedding_model']
    search_num = config['search_num']

    # Input text by conversation system
    question = input("Please input your question: ")
    lang = detect(question)
    if lang != 'ja':
        extract_keywords_prompt_path = add_lang_to_promptpath(extract_keywords_prompt_path, lang)
        generate_answer_prompt_path = add_lang_to_promptpath(generate_answer_prompt_path, lang)

    print("-------------------")
    print("Question:", question)
    print("Language:", lang)

    # Extract keywords from the question
    keyword_extractor = ke.KeywordExtractor(extract_keywords_prompt_path, retrieve_llm)
    keywords = keyword_extractor.extract_keywords(question)
    print("-------------------")
    print("Key words extraction model: ",retrieve_llm)
    print("Extracted keywords:", keywords)
    
    # Translate keywords if the language is not Japanese
    if lang != 'ja':
        print(f"Translate from {lang} to ja.")
        url = 'http://localhost:5001/translate'
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json={"keywords": keywords, "lang": lang})
        if response.status_code == 200:
            keywords = response.json()  # JSONレスポンスを辞書型として取得
            print(keywords)
        else:
            print(f"Error: {response.status_code}")

    # Search the knowledge graph
    vector_searcher = vs.VectorSearcher(embedding_model, connection_name=config['weaviate']['schema']['class'])
    print("-------------------")
    print("Embedding model:", embedding_model)
    print("Connection name:", config['weaviate']['schema']['class'])
    keywords_list = vector_searcher.search(keywords, n=search_num)
    print("Vector search reslts:", keywords_list)

    # Generate answer
    answer_generator = ag.AnswerGenerator(generate_answer_prompt_path, generate_llm)
    answer = answer_generator.generate_answer(keywords_list, question)

    print("-------------------")
    print("Answer generation model:", generate_llm)
    print("Answer:", answer)
    

if __name__ == '__main__':
    test()
