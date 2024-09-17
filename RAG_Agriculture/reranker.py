"""
検索結果のリストを受け取り、再ランキングを行う
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .utils.data_load import json2str
import copy

class Reranker:

    def __init__(self, model_name:str):
        # MPSまたはCUDAが利用可能か確認
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
            )
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        

    def _text_preprocess(self, keywords_list:list) -> list[str]:
        """
        Inputs:
            keywords_list: list, キーワードリスト
        Outputs:
            preprocessed_keywords_list: list, 前処理済みキーワードリスト
        """
        target_keys = [
            'task_name',
            'purpose',
            'action',
            'target',
            'sub_target',
            'crop_example',
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "location",
            "method",
            "material",
            "reading",
            "notation",
            "english",
            "concept",
            "condition"
            ]
        # target_keysに含まれるキーのみを抽出
        for i, keywords in enumerate(keywords_list):
            keywords_list[i] = {key: keywords[key] for key in target_keys if key in keywords}

        preprocessed_keywords_list = [json2str(keywords) for keywords in keywords_list]

        return preprocessed_keywords_list


    def rerank(self, question:str, search_results: list[dict]) -> tuple:
        """
        Inputs:
            question: str, 質問
            search_results: list, 検索結果(dict)のリスト
        Outputs:
            sorted_results: list[dict], 再ランキングされた検索結果のリスト
            scores: list, 再ランキングされた検索結果のスコアのリスト
        """
        tmp_search_results = copy.deepcopy(search_results)
        processed_keywords_list = self._text_preprocess(search_results)
        pairs = [(question, result) for result in processed_keywords_list]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().to('cpu')

        sorted_results = [result for _, result in sorted(zip(scores, tmp_search_results), key=lambda x: x[0], reverse=True)]
        scores = [score.item() for score in sorted(scores, reverse=True)]

        return sorted_results, scores


if __name__ == '__main__':
    """
    テストコード
    _text_preprocessのtarget_keysに含まれるキーのうちtask_name, purpose以外はコメントアウトする必要あり
    """
    model_name = "Alibaba-NLP/gte-multilingual-reranker-base"
    reranker = Reranker(model_name)
    question = '除草作業として適切なのはどれですか？'
    search_results = [
        {"task_name": "基本農作業", "purpose": "", "test": "test"},
        {"task_name": "直播", "purpose": "", "test": "test2"},
        {"task_name": "機械除草", "purpose": "除草", "test": "test3"},
    ]
    sorted_results, scores = reranker.rerank(question, search_results)
    print(sorted_results)
    print(scores)