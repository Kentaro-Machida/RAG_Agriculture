import sys 

def keywords2text(keywords: dict) -> str:
    '''
    Convert keywords dict to text with space separated.
    '''
    text = ''
    keywords = dict(sorted(keywords.items(), key=lambda x: x[0]))
    for key, value in keywords.items():
        text += key + ': ' + value + ' '
    
    return text


def count_layers(keywords: dict) -> int:
    '''
    Count the number of layers in the keywords dict.
    '''
    target_list = ['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth']
    count = 0
    for target in target_list:
        if keywords[target] != '':
            count += 1
    return count


def mE5_preprocess(text:str, target:str)->str:
    '''
    Preprocess text for mE5 embedding model.
    target: 'query' or 'answer'
    '''
    if target == 'query':
        text = 'query: ' + text
    elif target == 'answer':
        text = 'passage: ' + text
    else:
        print('Please set the correct target for mE5 preprocessing.')
        sys.exit()
    
    return text

def mE5_postprocess(text:str)->str:
    '''
    Postprocess text for mE5 embedding model.
    '''
    text = text.split('passage: ')[-1]
    return text

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def mE5_preprocess_instruct(query:str)->str:
    '''
    Preprocess mE5-instruction embedding model.
    This instruction is for NFCorpus. 
    Please check the detailed instruction in this URL:
    https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
    '''
    instruction = 'Given a question, retrieve relevant documents that best answer the question'
    query = get_detailed_instruct(instruction, query)

    return query

def test():
    keywords = {
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
        "URI": "",
        "first": "hello",
        "second": "",
        "third": "こんにちは",
        "fourth": "",
        "fifth": "",
        "sixth": "",
        "seventh": "",
        "eighth": "",
        "ninth": "",
        "tenth": ""
    }
    print('------keywords2text------')
    print(keywords2text(keywords))

    print('------count_layers------')
    print(count_layers(keywords))

    query = '鉄コーディングとはなんですか？'
    print('------mE5_preprocess------')
    print(mE5_preprocess(query, 'query'))

    retrieval = '鉄コーティングとは、種子に鉄をまぶすことで、水田での発芽を安定化させ、病気や害虫からの保護が期待される方法です。'
    print('------mE5_preprocess------')
    print(mE5_preprocess(retrieval, 'answer'))

    print('------mE5_postprocess------')
    retrieved = 'passage: 鉄コーティングとは、種子に鉄をまぶすことで、水田での発芽を安定化させ、病気や害虫からの保護が期待される方法です。'
    print(mE5_postprocess(retrieved))

    print('------mE5_preprocess_instruct------')
    query = '鉄コーディングとはなんですか？'
    print(mE5_preprocess_instruct(query))

if __name__ == '__main__':
    test()    
