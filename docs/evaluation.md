# Evaluation

Use `model.evaluate()` to score a query on labeled data. Each record contains
a `state` and a `label`. Integer labels are zero-based indices into the query's
options.

```python
from social_jev import Query
from social_jev.metrics import f1

query = Query(
    text="Does this invention process speech or audio?",
    options=["Yes", "No"],
)
evaluation = model.evaluate(
    query=query,
    data=[
        {"state": "A method for recognizing spoken commands.", "label": 0},
        {"state": "A method for detecting defects in photographs.", "label": 1},
    ],
    metric=f1,
)
print(evaluation.score)
```

SocialJev ships `accuracy` and binary `f1` in `social_jev.metrics`. Evaluation
defaults to accuracy, the fraction of predictions that match the integer
reference label. Pass a custom callable as `metric=` that takes the batch of
predicted probability vectors in query option order and the corresponding
labels, and returns a scalar score. Binary F1 treats option index `0` as positive.

The result contains the batch `score`, one `Result` per record in `predictions`,
and the corresponding ground-truth `labels`. Evaluation uses your configured
inference backend and is available with the base install.
