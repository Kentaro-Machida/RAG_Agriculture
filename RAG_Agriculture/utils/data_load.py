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

    json_str = '<SOJ>' + json.dumps(json_data) + '<EOJ>'
    return json_str


