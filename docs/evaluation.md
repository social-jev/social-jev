# Evaluation

`model.evaluate()` measures how well SocialJev's predictions match reference labels for a query and its answer options. It returns the metric score and the individual predictions.

## Evaluate a query

This example evaluates whether SocialJev can identify support for expanding bus service. Each record pairs a passage with its reference answer. The labels are zero-based indices into the query's options, so `0` means `Supports` and `1` means `Opposes`.

```python
from social_jev import Query, SocialJev

model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview")
query = Query(
    text="Does the author support expanding bus service?",
    options=["Supports", "Opposes"],
)
examples = [
    {"state": "The city should add more bus routes.", "label": 0},
    {"state": "The council should cancel the expansion.", "label": 1},
]

evaluation = model.evaluate(query=query, data=examples)
print(evaluation.score)
print(evaluation.predictions[0].choice)
```

## Default metrics

SocialJev ships two built-in metrics, accuracy and binary F1. By default, `model.evaluate()` reports accuracy. Accuracy is the fraction of documents where the model's top choice matches the reference label. The label is the zero-based index of the correct option.

Binary F1 combines precision and recall for the first option. In this example, precision is the fraction of predicted `Supports` answers that are correct, and recall is the fraction of reference `Supports` answers the model predicts correctly. Binary F1 uses integer labels and is selected by passing `metric=f1`.

```python
from social_jev.metrics import f1

evaluation = model.evaluate(query=query, data=examples, metric=f1)
```

To count `Opposes` as positive, put it first and remap the labels, or pass a custom metric that treats index `1` as positive.

## API reference

### `SocialJev.evaluate(*, query, data, metric=accuracy)`

- `query` is the query with answer options applied to every document.
- `data` supplies records with `state` and `label`.
- `metric` returns one float or integer score for the batch. A custom metric can apply a different scoring rule.

`predictions[i][j]` is the predicted probability of `query.options[j]` for document `i`, and `labels[i]` is that document's reference label. For instance, `predictions[0][1]` is the probability the model assigns to `Opposes` for the first document, and a value of `0` in `labels[0]` means the reference answer is `Supports`.

```python
from social_jev.evaluation import Label


def my_metric(predictions: list[list[float]], labels: list[Label]) -> float | int: ...


evaluation = model.evaluate(query=query, data=examples, metric=my_metric)
```

`EvaluationResult` contains `score`, `predictions`, and `labels`. Predictions and labels follow document order.
