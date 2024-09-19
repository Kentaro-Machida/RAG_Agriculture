from googletrans import Translator
from langdetect import detect

def detect_language(text:str)->str:
    """テキストの言語を検出する"""
    if not text.strip():
        return None
    return detect(text)


def translate_to_japanese(keywords_dict:dict, src_lang:str)->dict:
    """他言語を日本語に翻訳する処理"""
    translator = Translator()
    pass_keys = ['aao_id', 'agrovoc', 'naropedia']
    for k, v in keywords_dict.items():
        if k in pass_keys:
            continue
        try:
            if v and not v.isspace():
                v = translator.translate(v, src=src_lang, dest='ja').text
                keywords_dict[k] = v
        except Exception as e:
            print(f"Error: {e}")
            print(f"Failed to translate {v} to Japanese.")
    return keywords_dict


def translate_to_target_language(keywords_dict:dict, dest_lang:str)-> dict:
    """日本語を他言語に翻訳する処理"""
    translator = Translator()
    pass_keys = ['aao_id', 'agrovoc', 'naropedia']
    for k, v in keywords_dict.items():
        if k in pass_keys:
            continue
        try:
            if v and not v.isspace():
                v = translator.translate(v, src='ja', dest=dest_lang).text
                keywords_dict[k] = v
        except Exception as e:
            print(f"Error: {e}")
            print(f"Failed to translate {v} to {dest_lang}.")
    return keywords_dict


if __name__ == '__main__':
    # テスト用のデータ
    keywords = {
        'keyword1': 'apple',
        'keyword2': 'banana',
        'keyword3': 'orange',
        'keyword4': "https://met",
        'keyword5': '',
        'keyword6': ' ',
        'aao_id': 'Hello',
        'agrovoc': 'World',
        'naropedia': '!'
    }
    lang = 'en'

    # 英語 → 日本語への翻訳
    print("------Translate to Japanese.------")
    print(keywords)
    translated_keywords = translate_to_japanese(keywords, lang)
    print(translated_keywords)
    
    print("------Translate from Japanese.------")
    re_translated_keywords = translate_to_target_language(translated_keywords, lang)
    print(re_translated_keywords)
