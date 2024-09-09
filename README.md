# RAG_Agriculture
Retrieval augmented generation using LLM for Agriculture knowledge graph for Japanese.

![Architecture Diagram](images/architecture_diagram.png)

## config.json
さまざまなファイルの設定が書き込んであるファイル。`main.py`と`prepare_vector_db.py`で読み込む。

```json
{
    "retrieve_llm":"gpt-3.5-turbo",  # 質問文からキーワードを抜き出すLLM
    "extract_keywords_prompt_path":"./prompts/extract_keywords_prompt.txt",  # キーワードを抜き出すLLMの初期プロンプトへのパス
    "generate_llm":"gpt-3.5-turbo",  # 検索結果キーワードと質問文から解答を生成するLLM
    "generate_answer_prompt_path":"./prompts/generate_answer_prompt.txt",  # generate_llmの初期プロンプトへのパス
    "embedding_model":"openai",  # ベクトル検索に使用する埋め込みモデル
    "search_num":5,  # ベクトル検索でとってくるオブジェクト数
    # weaviate の設定
    "weaviate":{
        # keyはwaviate collectionのプロパティ名、valueはデータセットにおける対応するカラム名
        "target_columns":{
            "task_name": "作業名",
            "purpose": "目的(ja)",
            "action": "行為(ja)",
            "target": "対象(ja)",
            "sub_target": "副対象(ja)",
            "location": "場所(ja)",
            "method": "手段(ja)",
            "material": "機資材(ja)",
            "crop_example": "生産対象(ja)",
            "season": "時期(ja)",
            "condition": "作業条件(ja)"
        },
        # ベクトルDBとして使用する農作業オントロジーデータへのパス
        "data_path": "./datasets/aao_mini_sample.csv",
        # weaviate で collection を作成する際の設定
        "schema":{
            "class": "Agriculuture",
            "description": "農作業基本オントロジーに関するデータ。URL:https://cavoc.org/aao/ns/4/A1.html",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-contextionary": {
                    "vectorizeClassName": true   
                }
            },
            "properties":[
                {
                    "name": "task_name",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-contextionary": {
                            # プロパティ名をベクトルの埋め込みに使用する場合true
                            "vectorizePropertyName": true
                        }
                    }
                },
                {
                    "name": "purpose",
                    "dataType": ["text"]
                },
                ......
            ]
        }
    }
}
```
それぞれのモデルの選択肢は以下の通り。
* retrieve_llm: gpt-3.5-turbo
* embedding_model: openai, intfloat/multilingual-e5-large
* generate_llm: gpt-3.5-turbo

## 使い方
全てベースディレクトリ（このREADME.md と同じディレクトリ）からの実行を前提とする。
農作業オントロジーとLLMを使用した対話システムを使用するいは以下の流れで進めていく。

### OpenAI APIの準備
OpenAIのAPIを使用する場合、OpenAIのサイトでアカウントを作り、APIの契約をする必要がある。デポジット性で使用前にお金を払っておく必要があることに注意。契約後、OpenAIのAPIキーを自身の環境で環境変数として以下のように設定。

```
export OPENAI_API_KEY={Your API key}
```

### Weaviate の Docker を立てるコマンド
Weaviate のベクトルDBを配置するDockerコンテナを用意。

```
docker compose up
```

### 農作業オントロジーデータをベクトルDBにセット
以下のコマンドで、csv形式の農作業オントロジーデータをベクトルDBにセットする。OpenAIの埋め込みモデルを使用する場合は以下のスクリプトを実行。
```
python prepare_vector_db.py
```

Hugging face で公開されているOSS埋め込みモデルを使用する際は以下のスクリプトを実行。
```
python prepare_vector_db_oss.py
```

### translatorの起動
日本語以外での検索を可能にするためのtranslaatorを起動する必要がある。以下のコマンドでflaskによるサーバーが立ち上がる。中身は、googletransという無料の翻訳機。
```
python RAG_Agriculture/translator.py
```

### 質問システムの起動
以下のコマンドで、RAG質問システムが起動

```
python main.py
```

## 各モジュールの簡単な説明
RAG_Agriculuture ディレクトリの中にモジュールが格納されている。

### keywords_extractor
質問文から農作業オントロジーデータの項目に合わせてkeywardを抜き出すモジュール
* 入力：質問文（str）
* 出力：キーワード辞書（dict）

### vector_searcher
キーワード辞書を使用して、ベクトル検索を行い、ヒットした上位のオブジェクトをリスト形式で返すモジュール
* 入力：キーワード辞書（dict）
* 出力：ヒットしたオブジェクトのリスト（list）

### answer_generator
検索システムにより抜き出してきた情報と質問文を組み合わせて解答を生成するモジュール
* 入力：質問文（str）、ヒットしたオブジェクトのリスト（list）
* 出力：生成データ

### utils
モジュール共通で使用するクラスや関数をまとめて置いてある。

## prompts
各LLMに与えるための初期プロンプト。

### extract_keywords_prompt.txt
質問を入力とし、そこからキーワードを抽出するLLMのためのプロンプト。
日本語で書かれている。

### generate_answer_prompt.txt
RAGにより検索して返ってきたドキュメントに加えて、回答生成用のLLMに与えるためのプロンプト。日本語で書かれている。

### extract_keywords_prompt_{language}.txt, generate_answer_prompt_{language}.txt
上記2プロンプトの日本語以外のもの。
{language}語で書かれている。言語の命名規則に従わないとエラーが発生。
命名規則はgoogletransライブラリの言語の表記に準拠。
日本語で使用する分には作成する必要はない。
* 英語：{language}=en
* ベトナム語：{language}=vi

