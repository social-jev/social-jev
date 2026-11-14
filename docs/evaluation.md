# Evaluation

`model.evaluate()` measures how well SocialJev's predictions match reference labels for a query and its answer options. It returns the metric score and the individual predictions.

## Evaluate a query

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

An integer label is a zero-based option index. A distribution label such as `[0.75, 0.25]` gives one reference weight per option in the same order.

## Default metrics

Accuracy gives each prediction one point when its highest-probability option matches the integer label. For a distribution label, it gives the prediction the reference weight at the selected index.

```python
from social_jev.metrics import f1

evaluation = model.evaluate(query=query, data=examples, metric=f1)
```

Binary F1 accepts integer labels and treats the first option as positive. With the options above, predicting `Supports` for a document labeled `Opposes` is a false positive. Predicting `Opposes` for a document labeled `Supports` is a false negative. To count `Opposes` as positive, put it first and remap the labels, or pass a custom metric that treats index `1` as positive.

## API reference

### `SocialJev.evaluate(*, query, data, metric=accuracy)`

- `query` is the query with answer options applied to every document.
- `data` supplies records with `state` and `label`.
- `metric` returns one float or integer score for the batch.

`predictions[i][j]` is the predicted probability of `query.options[j]` for document `i`, and `labels[i]` is that document's reference label. For instance, `predictions[0][1]` is the first document's probability of `Opposes`, while `labels[0] = 0` means its reference answer is `Supports`.

```python
from social_jev.evaluation import Label


def my_metric(predictions: list[list[float]], labels: list[Label]) -> float | int: ...


evaluation = model.evaluate(query=query, data=examples, metric=my_metric)
```

`EvaluationResult` contains `score`, `predictions`, and `labels`. Predictions and labels follow document order.
