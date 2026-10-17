"""Shared evaluation metrics; PR-AUC uses trapezoidal integration, not AP."""
from sklearn.metrics import auc, precision_recall_curve


def pr_auc_score(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score


def score_metrics(frame):
    y, score = frame.label.astype(int), frame.score
    return dict(n_sites=len(frame), n_positive=int(y.sum()), prevalence=float(y.mean()),
                average_precision=float(average_precision_score(y,score)),
                pr_auc=pr_auc_score(y,score), roc_auc=float(roc_auc_score(y,score)),
                brier=float(brier_score_loss(y,score)),
                precision_at_0_5=float(precision_score(y,score >= .5,zero_division=0)),
                recall_at_0_5=float(recall_score(y,score >= .5,zero_division=0)))
