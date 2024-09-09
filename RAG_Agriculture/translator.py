"""
与えられたキーワードを翻訳するための関数を提供するモジュール
WeaviateやOpenAIAPIとの環境コンフリクトが発生するのでFlaskサーバーとして実装
JSON形式でPostされたデータのvalueの言語を翻訳し、その結果をJSON形式で返す
"""
import sys
from flask import Flask, request
import json
from googletrans import Translator
from langdetect import detect

app = Flask(__name__)
translator = Translator()

@app.route('/translate', methods=['POST'])
def translate_text():
    joined_text = ''
    # リクエストから翻訳するテキストを取得
    posted_json = request.json
    keywords_dict = posted_json['keywords']
    print(keywords_dict)
    question_lang = posted_json['lang']
    print(question_lang)

    # 言語の判定
    for v in keywords_dict.values():
        joined_text += v + ', '

    src_lang = detect(joined_text)
    print('keywords: ',joined_text)
    print('src: ',src_lang)
    if src_lang == 'ja':
        dest_lang = question_lang
    elif src_lang == question_lang:
        dest_lang = 'ja'
    else:
        print('Invalid source language. Please check the source language.')
        print('Detected question language:', question_lang)
        print('Detected values language:', src_lang)
        sys.exit()

    for k, v in keywords_dict.items():
        # v が空欄やスペース、特殊文字などの場合は翻訳しない
        if v == '' or v.isspace():
            continue
        
        v = translator.translate(v, src=src_lang, dest=dest_lang).text
        keywords_dict[k] = v
    
    # 翻訳結果を返す
    return app.response_class(
        response=json.dumps(keywords_dict, ensure_ascii=False),
        mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
