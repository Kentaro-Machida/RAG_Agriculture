from flask import Flask, request
import json
from utils.translate_module import  translate_to_japanese, translate_to_target_language

app = Flask(__name__)

@app.route('/translate/to_ja', methods=['POST'])
def translate_to_ja():
    """他言語を日本語に翻訳するエンドポイント"""
    posted_json = request.json
    keywords_dict = posted_json['keywords']
    question_lang = posted_json['lang']

    # キーワードの結合して言語検出
    joined_text = ', '.join(v for v in keywords_dict.values() if v and not v.isspace())
    
    print('Detected language: ', question_lang)
    
    # 他言語 → 日本語への翻訳
    translated_keywords = translate_to_japanese(keywords_dict, question_lang)
    
    # 翻訳結果を返す
    return app.response_class(
        response=json.dumps(translated_keywords, ensure_ascii=False),
        mimetype='application/json'
    )


@app.route('/translate/from_ja', methods=['POST'])
def translate_from_ja():
    """日本語を他言語に翻訳するエンドポイント"""
    posted_json = request.json
    keywords_dict = posted_json['keywords']
    question_lang = posted_json['lang']

    # キーワードの結合して言語検出（今回は日本語なのでスキップしても良い）
    src_lang = 'ja'
    
    print('Translate from Japanese to:', question_lang)
    
    # 日本語 → 他言語への翻訳
    translated_keywords = translate_to_target_language(keywords_dict, question_lang)
    
    # 翻訳結果を返す
    return app.response_class(
        response=json.dumps(translated_keywords, ensure_ascii=False),
        mimetype='application/json'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)