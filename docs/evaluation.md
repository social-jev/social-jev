# Evaluation

Use `model.evaluate()` to score a query on labeled data. Each record contains
a `state` and a `label`. Integer labels are zero-based indices into the query's
options. Distribution labels give one target probability per option, in the
same order.

```python
from social_jev import Query
from social_jev.metrics import f1

query = Query(
    text="Does this invention process speech or audio?",
    options=["No", "Yes"],
)
evaluation = model.evaluate(
    query=query,
    data=[
        {"state": "A method for recognizing spoken commands.", "label": 1},
        {"state": "A method for detecting defects in photographs.", "label": 0},
    ],
    metric=f1,
)
print(evaluation.score)
```

SocialJev ships `accuracy` and binary `f1` in `social_jev.metrics`. Evaluation
defaults to accuracy. Pass any callable as `metric=` that takes the batch of
predicted probability vectors in query option order and the corresponding
labels, and returns a scalar score. Binary F1 treats option index `1` as positive.

The result contains the batch `score`, one `Result` per record in `predictions`,
and the corresponding ground-truth `labels`. Evaluation uses your configured
inference backend and is available with the base install.
