# sequential.py
import pandas as pd
from flask import render_template

# Simple implementations of defaultdict and Counter using standard dict
class Counter(dict):
    def __missing__(self, key):
        return 0

class DefaultDict(dict):
    def __init__(self, default_factory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory
    def __getitem__(self, key):
        if key not in self:
            self[key] = self.default_factory()
        return dict.__getitem__(self, key)

# ------------------- Helper functions -------------------
def build_transitions(df):
    transitions = DefaultDict(Counter)
    for uid, group in df.groupby('user_id'):
        seq = list(group.sort_values('timestamp')['item_id'])
        for i in range(len(seq)-1):
            a = seq[i]; b = seq[i+1]
            transitions[a][b] += 1
    # normalize
    P = {}
    for a, ctr in transitions.items():
        total = float(sum(ctr.values())) + 1e-8
        P[a] = {b: cnt/total for b, cnt in ctr.items()}
    return P


def user_scores_from_lastk(last_k_items, P, decay=0.7):
    scores = {}
    for idx, item in enumerate(last_k_items):
        w = decay**idx
        if item in P:
            for j, p in P[item].items():
                scores[j] = scores.get(j, 0) + w * p
    return scores


def aggregate_group_scores(member_scores, agg='avg', member_weights=None):
    items = set().union(*[set(s.keys()) for s in member_scores]) if member_scores else set()
    if member_weights is None:
        member_weights = [1.0]*len(member_scores)
    total_w = sum(member_weights) if member_weights else 1.0
    agg_scores = {}
    for it in items:
        vals = [s.get(it,0.0) for s in member_scores]
        if agg=='avg':
            agg_scores[it] = sum(w*v for w,v in zip(member_weights, vals)) / total_w
        elif agg in ('min','least_misery'):
            agg_scores[it] = min(vals)
        elif agg in ('max','dictator'):
            agg_scores[it] = max(vals)
        elif agg=='median':
            agg_scores[it] = sorted(vals)[len(vals)//2]
        else:
            agg_scores[it] = sum(vals)/len(vals)
    return agg_scores


def recommend_for_group(group_member_ids, df, P, last_k=3, decay=0.7, agg='avg', topk=10, exclude_seen=True):
    member_scores = []
    member_weights = []
    for uid in group_member_ids:
        his = df[df.user_id==uid].sort_values('timestamp')['item_id'].tolist()
        lastk = his[-last_k:][::-1] if len(his)>0 else []
        s = user_scores_from_lastk(lastk, P, decay=decay)
        member_scores.append(s)
        member_weights.append(1.0)
    agg_scores = aggregate_group_scores(member_scores, agg=agg, member_weights=member_weights)
    if exclude_seen:
        seen = set(df[df.user_id.isin(group_member_ids)]['item_id'].tolist())
        for it in list(agg_scores.keys()):
            if it in seen:
                del agg_scores[it]
    ranked = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:topk]


def sequential(df):
    """Generate group sequential recommendations and render them in sequential.html."""
    group = df['user_id'].unique()[:3].tolist()  # pick first 3 users from dataset
    P = build_transitions(df)
    recs = recommend_for_group(group, df, P, last_k=3, decay=0.7, agg='avg', topk=10, exclude_seen=False)
    return render_template('sequential.html', recommendations=recs)