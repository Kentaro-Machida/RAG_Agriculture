import numpy as np
from sklearn.metrics import ndcg_score

def apk(actual, predicted, k=10):
    """
    https://github.com/benhamner/Metrics/tree/master/Python/ml_metrics
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    https://github.com/benhamner/Metrics/tree/master/Python/ml_metrics
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def ndcg_from_taskname(predicted_task_names:list[str], gt_task_names:list[str], k=20):
    """
    predicted_task_names: 
        ランキングアルゴリズムによって予測されたタスク名のリスト
        若いindexは高いスコアを持つ、つまり優先的にレコメンド
    gt_task_names:
        正解のタスク名のリスト。正解さえ含まれていれば順番は評価値には関係がない
    k:
        上位k個についてNDGCを計算
    """
    gt_list = []
    for task_name in predicted_task_names:
        if task_name in gt_task_names:
            gt_list.append(1)
        else:
            gt_list.append(0)
    gt_np_array = np.array([gt_list])
    print(gt_np_array)
    pred_rank = [i for i in range(len(predicted_task_names), 0, -1)]
    pred_rank_np_array = np.array([pred_rank])

    return float(ndcg_score(gt_np_array, pred_rank_np_array, k=k))


def mean_ndcg_from_taskname(predicted_task_names_list:list[list[str]], gt_task_names_list:list[list[str]], k=20):
    """
    predicted_task_names_list: 
        ランキングアルゴリズムによって予測されたタスク名のリストのリスト
        若いindexは高いスコアを持つ、つまり優先的にレコメンド
    gt_task_names_list:
        正解のタスク名のリストのリスト。正解さえ含まれていれば順番は評価値には関係がない
    k:
        上位k個についてNDGCを計算
    """
    return np.mean([ndcg_from_taskname(predicted_task_names, gt_task_names, k) for predicted_task_names, gt_task_names in zip(predicted_task_names_list, gt_task_names_list)])


def recall_k_from_taskname(predicted_task_names:list[str], gt_task_names:list[str], k=20):
    """
    predicted_task_names_list: 
        ランキングアルゴリズムによって予測されたタスク名のリストのリスト
        若いindexは高いスコアを持つ、つまり優先的にレコメンド
    gt_task_names_list:
        正解のタスク名のリストのリスト。正解さえ含まれていれば順番は評価値には関係がない
    k:
        上位k個についてRecallを計算
    """
    gt_set = set(gt_task_names)
    pred_set = set(predicted_task_names[:k])
    return len(gt_set & pred_set) / len(gt_set)


def mrr(predicted_task_names:list[str], gt_task_names:list[str]):
    """
    predicted_task_names_list: 
        ランキングアルゴリズムによって予測されたタスク名のリストのリスト
        若いindexは高いスコアを持つ、つまり優先的にレコメンド
    gt_task_names_list:
        正解のタスク名のリストのリスト。正解さえ含まれていれば順番は評価値には関係がない
    """

    for i, task_name in enumerate(predicted_task_names):
        if task_name in gt_task_names:
            return 1 / (i+1)
        
    return 0


def all_metrics(predicted_task_names:list[str], gt_task_names:list[str], k=20):
    return {
        f'ndcp@{k}': ndcg_from_taskname(predicted_task_names, gt_task_names, k),
        f'recall@{k}': recall_k_from_taskname(predicted_task_names, gt_task_names, k),
        'mrr': mrr(predicted_task_names, gt_task_names)
    }


if __name__ == '__main__':
    print('----- test1 -----')
    pred_task_names = ["task5", "task3", "task4", "task1", "task2", "task6", "task7", "task8", "task9", "task10"]
    gt_task_names = ["task1", "task2", "task3", "task100", "task10"]
    print(ndcg_from_taskname(pred_task_names, gt_task_names, k=5))
    print(recall_k_from_taskname(pred_task_names, gt_task_names, k=5))
    print(mrr(pred_task_names, gt_task_names))

    print('----- test2 -----')
    pred_task_names = ["task1", "task3", "task4", "task5", "task2", "task6", "task7", "task8", "task9", "task10"]
    gt_task_names = ["task1", "task2", "task3", "task100", "task10"]
    print(ndcg_from_taskname(pred_task_names, gt_task_names, k=5))
    print(recall_k_from_taskname(pred_task_names, gt_task_names, k=5))
    print(mrr(pred_task_names, gt_task_names))

    print('----- test3 -----')
    pred_task_names = ["task1", "task3", "task4", "task5", "task6", "task2", "task7", "task8", "task9", "task10"]
    gt_task_names = ["task1", "task2", "task3", "task100", "task10"]
    print(ndcg_from_taskname(pred_task_names, gt_task_names, k=5))
    print(recall_k_from_taskname(pred_task_names, gt_task_names, k=5))
    print(mrr(pred_task_names, gt_task_names))

    print('----- test4 -----')
    pred_task_names = ["task1", "task3", "task4", "task5", "task6", "task2", "task7", "task8", "task9", "task10"]
    gt_task_names = ["task1", "task2", "task3", "task100", "task10"]
    print(all_metrics(pred_task_names, gt_task_names, k=5))

    print('----- test5 -----')
    pred_task_names = [
        "A514",
        "A531",
        "A511",
        "A510",
        "A509",
        "A503",
        "A494",
        "A512",
        "A337",
        "A513",
        "A485",
        "A 6",
        "A62",
        "A330",
        "A294",
        "A231",
        "A303",
        "A484",
        "A470",
        "A308",
        "A293",
        "A84",
        "A134",
        "A350",
        "Sub -7",
        "Sub -4",
        "A358",
        "A132",
        "A327",
        "A323"
    ]
    gt_task_names = [
        "A514",
        "A515",
        "A516",
        "A517",
        "A518"
    ]
    print(all_metrics(pred_task_names, gt_task_names, k=10))

    print('----- test6 -----')
    pred_task_names = ["A16","A51"]
    gt_task_names = [
        "A46",
        "A39",
        "A40",
        "A55",
        "A477",
        "A549",
        "A41",
        "A45",
        "A44",
        "A54",
        "Sub -8",
        "A43",
        "A49",
        "A48",
        "Sub -7",
        "A434",
        "A101",
        "Sub -4",
        "A56",
        "A443",
        "A52",
        "A110",
        "A53",
        "A330",
        "A112",
        "A92",
        "A93",
        "A80",
        "A468",
        "A109"
    ]
    print(all_metrics(pred_task_names, gt_task_names, k=10))