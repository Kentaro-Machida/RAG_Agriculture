"""
実行したら対話モードに入り、農業系の質問をすると、
ベクトルDBから関連する情報を取得し、回答を生成して返す。
"""
import requests
from RAG_Agriculture import answer_generator as ag, keywords_extractor as ke, vector_searcher as vs
from RAG_Agriculture.utils.data_load import load_json, add_lang_to_promptpath
from RAG_Agriculture.utils.text_preprocess import count_layers
from langdetect import detect


def main(question: str, config: dict):

    
    retrieve_llm = config['retrieve_llm']
    extract_keywords_prompt_path = config['extract_keywords_prompt_path']
    generate_llm = config['generate_llm']
    generate_answer_prompt_path = config['generate_answer_prompt_path']
    embedding_model = config['embedding_model']
    search_num = config['search_num']

    lang = detect(question)
    if lang != 'ja':
        extract_keywords_prompt_path = add_lang_to_promptpath(extract_keywords_prompt_path, lang)
        generate_answer_prompt_path = add_lang_to_promptpath(generate_answer_prompt_path, lang)

    # Extract keywords from the question
    keyword_extractor = ke.KeywordExtractor(extract_keywords_prompt_path, retrieve_llm)
    keywords = keyword_extractor.extract_keywords(question)

    # Translate keywords if the language is not Japanese
    if lang != 'ja':
        print(f"Translate from {lang} to ja.")
        url = config['to_ja_translator_url']
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json={"keywords": keywords, "lang": lang})
        if response.status_code == 200:
            keywords = response.json()  # JSONレスポンスを辞書型として取得
            print(keywords)
        else:
            print(f"Error: {response.status_code}")

    # Search the knowledge graph
    vector_searcher = vs.VectorSearcher(embedding_model, connection_name=config['weaviate']['schema']['class'])
    keywords_list = vector_searcher.search(keywords, n=search_num)

    # Translate keywords back to the original language
    if lang != 'ja':
        print("-------------------")
        print(f"Translate from ja to {lang}.")
        url = config['from_ja_translator_url']
        headers = {'Content-Type': 'application/json'}
        for i, keywords in enumerate(keywords_list):
            response = requests.post(url, headers=headers, json={"keywords": keywords, "lang": lang})
            if response.status_code == 200:
                trans_keywords = response.json()  # JSONレスポンスを辞書型として取得
                keywords_list[i] = trans_keywords
            else:
                print(f"Error: {response.status_code}")    

    # Generate answer
    answer_generator = ag.AnswerGenerator(generate_answer_prompt_path, generate_llm)
    answer = answer_generator.generate_answer(keywords_list, question)

    return answer


def test(question:str, config:dict)->dict:
    
    process_log_dict = {}
    process_log_dict['question'] = question

    retrieve_llm = config['retrieve_llm']
    extract_keywords_prompt_path = config['extract_keywords_prompt_path']
    generate_llm = config['generate_llm']
    generate_answer_prompt_path = config['generate_answer_prompt_path']
    embedding_model = config['embedding_model']
    search_num = config['search_num']

    lang = detect(question)
    if lang != 'ja':
        extract_keywords_prompt_path = add_lang_to_promptpath(extract_keywords_prompt_path, lang)
        generate_answer_prompt_path = add_lang_to_promptpath(generate_answer_prompt_path, lang)
    process_log_dict['language'] = lang

    print("-------------------")
    print("Question:", question)
    print("Language:", lang)

    # Without RAG
    if config['without_rag']:
        print("--- Without RAG test ---")
        # Generate answer
        answer_generator = ag.AnswerGenerator(generate_answer_prompt_path, generate_llm)
        answer = answer_generator.generate_answer_without_rag(question)

        process_log_dict['Answer'] = answer
        print("-------------------")
        print("Answer generation model:", generate_llm)
        print("Answer:", answer)

        return process_log_dict

    # Extract keywords from the question
    keyword_extractor = ke.KeywordExtractor(extract_keywords_prompt_path, retrieve_llm)
    keywords = keyword_extractor.extract_keywords(question)
    process_log_dict['keywords'] = keywords
    print("-------------------")
    print("Key words extraction model: ",retrieve_llm)
    print("Extracted keywords:", keywords)
    
    # Translate keywords if the language is not Japanese
    if lang != 'ja':
        print(f"Translate from {lang} to ja.")
        url = 'http://localhost:5001/translate/to_ja'
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json={"keywords": keywords, "lang": lang})
        if response.status_code == 200:
            keywords = response.json()  # JSONレスポンスを辞書型として取得
            process_log_dict['translated_keywords'] = keywords
            print(keywords)
        else:
            print(f"Error: {response.status_code}")

    # Search the knowledge graph
    task_name_list = []
    layer_list = []
    vector_searcher = vs.VectorSearcher(embedding_model, connection_name=config['weaviate']['schema']['class'])
    print("-------------------")
    print("Embedding model:", embedding_model)
    print("Connection name:", config['weaviate']['schema']['class'])
    keywords_list = vector_searcher.search(keywords, n=search_num)
    for i, keywords in enumerate(keywords_list):
        task_name_list.append(keywords['task_name'])
        layer_list.append(count_layers(keywords))
        print(f"Search result {i+1}: {keywords['task_name']}")
        print(f"Number of layers: {count_layers(keywords)}")

    process_log_dict['Search result: task_name_list'] = task_name_list
    process_log_dict['Search result: layer_list'] = layer_list

    # Translate keywords back to the original language
    if lang != 'ja':
        print("-------------------")
        print(f"Translate from ja to {lang}.")
        url = 'http://localhost:5001/translate/from_ja'
        headers = {'Content-Type': 'application/json'}
        for i, keywords in enumerate(keywords_list):
            response = requests.post(url, headers=headers, json={"keywords": keywords, "lang": lang})
            if response.status_code == 200:
                trans_keywords = response.json()  # JSONレスポンスを辞書型として取得
                keywords_list[i] = trans_keywords
            else:
                print(f"Error: {response.status_code}")
        task_name_list = []
        layer_list = []
        for i, keywords in enumerate(keywords_list):
            task_name_list.append(keywords['task_name'])
            layer_list.append(count_layers(keywords))
            print(f"Translated search result {i+1}: {keywords['task_name']}")
            print(f"Number of layers: {count_layers(keywords)}")
        
        process_log_dict['Translated search result: task_name_list'] = task_name_list
        process_log_dict['Translated search result: layer_list'] = layer_list


    # Generate answer
    answer_generator = ag.AnswerGenerator(generate_answer_prompt_path, generate_llm)
    answer = answer_generator.generate_answer(keywords_list, question)

    process_log_dict['Answer'] = answer
    print("-------------------")
    print("Answer generation model:", generate_llm)
    print("Answer:", answer)
    
    return process_log_dict
    

if __name__ == '__main__':
    # load config.json
    config = load_json('./config.json')
    # Input text by conversation system
    question = input("Please input your question: ")
    
    test(question, config)
    # main(question, config)
