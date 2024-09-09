import json
from googletrans import Translator
from langdetect import detect

translator = Translator()

def detect_language(text):
    """テキストの言語を検出する"""
    if not text.strip():
        return None
    return detect(text)

def translate_to_japanese(keywords_dict, src_lang):
    """他言語を日本語に翻訳する処理"""
    for k, v in keywords_dict.items():
        if v and not v.isspace():
            v = translator.translate(v, src=src_lang, dest='ja').text
            keywords_dict[k] = v
    return keywords_dict

def translate_to_target_language(keywords_dict, dest_lang):
    """日本語を他言語に翻訳する処理"""
    for k, v in keywords_dict.items():
        if v and not v.isspace():
            v = translator.translate(v, src='ja', dest=dest_lang).text
            keywords_dict[k] = v
    return keywords_dict