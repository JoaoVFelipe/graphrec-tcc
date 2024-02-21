import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def dcg(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg(y_true, y_score, k=10):
    dcg_score = dcg(y_true, y_score, k)
    best_dcg = dcg(y_true, y_true, k)
    if best_dcg != 0:
        return  (float(dcg_score) / float(best_dcg))
    return dcg_score

def get_groups(qids):
        prev_qid = None
        prev_limit = 0
        total = 0

        for i, qid in enumerate(qids):
            total += 1
            if qid != prev_qid:
                if i != prev_limit:
                    yield (prev_qid, prev_limit, i)
                prev_qid = qid
                prev_limit = i

        if prev_limit != total:
            yield (prev_qid, prev_limit, total)

def check_qids(qids):
        seen_qids = set()
        prev_qid = None

        for qid in qids:
            assert qid is not None
            if qid != prev_qid:
                if qid in seen_qids:
                    raise ValueError('Samples must be grouped by qid.')
                seen_qids.add(qid)
                prev_qid = qid

        return len(seen_qids)


def queries_ndcg(y_true, y_score, qids, k = 10):
    query_groups = np.array([(qid, a, b, np.arange(a, b))
                                 for qid, a, b in get_groups(qids)],
                                dtype=np.object)

    n_queries = check_qids(qids)
    queries_ndcg = np.zeros(n_queries)

    for qidx, (qid, a, b, _) in enumerate(query_groups):
        # scores = model.predict(X.iloc[a:b])
        queries_ndcg[qidx] = ndcg(y_true.iloc[a:b], y_score[a:b], k)
        # result_list[qidx] = [[qidx], np.array(y_true.iloc[a:b]).flatten(),  np.array(y_score[a:b]).flatten()]
    return queries_ndcg

# Function to calculate Average Precision (AP) for a single query
def calculate_ap(retrieved, relevant):
    # Initialize variables
    precision_at_k = []
    num_relevant = len(relevant)
    num_retrieved = 0
    num_correct = 0
    total_sum = 0
    
    # Calculate precision at each position
    for i, item in enumerate(retrieved):
        if item in relevant.values:
            num_correct += 1
            total_sum = total_sum + num_correct
            precision_at_k.append(num_correct / (i + 1))
            num_retrieved += 1
    
    # Calculate Average Precision (AP)
    if num_relevant == 0:
        return 0  # If there are no relevant items for the query, AP is 0
    else:
        return sum(precision_at_k) / num_relevant

def mean_ap(result_df, positive_value):
    map_values = list()

    groupby_user = result_df.groupby('user')

    for user_id in groupby_user.groups.keys():
        user_df = groupby_user.get_group(user_id)

        relevant_itens = user_df.loc[user_df['rate'] == positive_value]['item']
        predictions = user_df.sort_values(by='prediction', ascending=False)
        ap = calculate_ap(predictions['item'], relevant_itens)
        map_values.append(ap)
        
    mean_ap = sum(map_values) / len(groupby_user.groups.keys())
    return mean_ap
