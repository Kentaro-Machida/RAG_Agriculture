import json

def load_text(text_path:str)->str:
    '''
    Load text from a file
    '''

    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def load_json(json_path:str)->dict:
    '''
    Load json from a file
    '''

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def str2json(json_str:str)->dict:
    '''
    Convert string to json from LLM output.
    Json data should be converted between <SOJ> and <EOJ>
    '''

    # <SOJ>と<>EOJ>で囲まれた部分を取り出す
    json_str = json_str.split('<SOJ>')[-1].split('<EOJ>')[0]
    json_data = json.loads(json_str)
    return json_data


def json2str(json_data:dict)->str:
    '''
    Convert json to string for LLM input.
    Json daa should be converted between <SOJ> and <EOJ>
    '''

    json_str = '<SOJ>' + json.dumps(json_data, ensure_ascii=False) + '<EOJ>'
    return json_str


def add_lang_to_promptpath(prompt_path: str, lang: str)->str:
    # 拡張子 .txt がファイルパスの最後にあるかを確認
    print(type(prompt_path))
    if prompt_path.endswith('.txt'):
        # .txt の前の部分と、後ろの拡張子部分に分ける
        base_path = prompt_path[:-4]  # .txt の前の部分
        extension = prompt_path[-4:]  # ".txt"
        
        # 新しいファイル名を作成
        new_prompt_path = f"{base_path}_{lang}{extension}"
        
        return new_prompt_path
    else:
        # .txt で終わっていない場合は、そのまま返す
        return prompt_path
