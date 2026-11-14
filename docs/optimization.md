# Optimization

`model.optimize()` searches for wording that improves SocialJev's predictions on a labeled task. AsyncGEPA runs the search. A reflection model proposes new wording for the query and its answer options, and SocialJev's predictions are scored against reference labels to compare candidates. Each option keeps its meaning and position, so the same reference labels apply throughout.

## Optimize a query

This example runs SocialJev on vLLM to make predictions for each candidate. GLM-5.3 Flash, accessed through OpenRouter, proposes rewrites. The default reflector requires `OPENROUTER_API_KEY` in the environment.

```bash
uv pip install 'social-jev[optim,vllm]'
```

!!! important
    The default asynchronous optimization loop requires vLLM. Standard synchronous GEPA works with Transformers using `social-jev[optim]`. See [Customizing optimization](#customizing-optimization) for the synchronous loop.

The search uses training examples to propose rewrites and validation examples to compare them. After the search, the script scores the original and selected queries on the same held-out test set. A training record can include an `explanation` for the reflection model, as in the example where the author and the mayor take different positions.

```python
from social_jev import Query, SocialJev

model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview", backend="vllm", serve=True)
query = Query(
    text="Does the author support expanding bus service?",
    options=["Supports", "Opposes"],
)
trainset = [
    {"state": "The city should add more bus routes.", "label": 0},
    {
        "state": "The mayor wants more routes, but I think the plan should be cancelled.",
        "label": 1,
        "explanation": "The author opposes the expansion. The mayor supports it.",
    },
]
valset = [
    {"state": "I welcome the plan to extend bus service.", "label": 0},
    {"state": "Adding more routes would be a waste of money.", "label": 1},
]
testset = [
    {"state": "More bus routes would help commuters.", "label": 0},
    {"state": "The council should leave the bus network as it is.", "label": 1},
]

try:
    optimization = model.optimize(
        query=query, trainset=trainset, valset=valset, max_metric_calls=50
    )
    print(optimization.query.text)
    print(optimization.query.options)

    for candidate in (query, optimization.query):
        evaluation = model.evaluate(query=candidate, data=testset)
        print(candidate.text, evaluation.score)
finally:
    model.stop()
```

The `max_metric_calls` budget counts each candidate scored on one example as one evaluation. That budget applies only to candidate scoring.

## Customizing optimization

Keyword arguments on `model.optimize()` configure the search. SocialJev uses the reflection and rollout settings to configure those components and forwards the remaining optimizer settings to AsyncGEPA. See the [GEPA optimizer reference] and [AsyncGEPA continuous rollouts].

### Reflection

GLM-5.3 Flash proposes rewrites by default, with requests routed through OpenRouter. A supported provider/model string or a compatible GEPA language-model object supplied as `reflection_lm` replaces that default. A `reflection_strategy` can implement reflection using any model or service it can call, including a local model. These alternatives use their own credentials and work without `OPENROUTER_API_KEY`.

For example, `reflection_lm="openai/gpt-5.1"` uses OpenAI credentials. `reflection_strategy=my_reflection_strategy` uses the supplied strategy. `custom_candidate_proposer` can replace the proposal step directly.

### Scheduling and sampling

The scheduler determines when evaluations start. The sampler selects which existing candidates and training examples to use for new proposals. With continuous rollouts, the default scheduler can start new evaluations as others finish, including while reflection is in progress. `max_in_flight_rollouts=16` limits concurrent evaluations.

Batched GEPA uses `scheduler=None` and `sampling_strategy=IndependentSampling(4)` to select four parent-and-minibatch pairs per iteration. The sampler comes from `async_gepa.strategies.proposal_sampling`.

Classic GEPA uses `scheduler=None` with the default single-proposal sampler. Both iteration-based configurations work with Transformers or vLLM. This example uses Transformers.

```python
model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview")
optimization = model.optimize(
    query=query,
    trainset=trainset,
    valset=valset,
    max_metric_calls=50,
    scheduler=None,
)
```

## API reference

### `SocialJev.optimize(*, query, trainset, valset=None, metric=None, **kwargs)`

`query` supplies the initial wording and answer options. `trainset` and `valset` accept lists or GEPA data loaders. Each record contains `state` and `label`, with an optional `explanation`. When `valset` is omitted, GEPA uses the training set for validation.

The default metric gives each prediction a score of 1.0 if its top option matches the integer reference label and 0.0 otherwise. A custom metric receives [probability vectors and reference labels](evaluation.md#api-reference) one example at a time. It returns a one-element list of scores and optional feedback for reflection.

```python
from social_jev.evaluation import Label


def my_metric(
    predictions: list[list[float]],
    labels: list[Label],
) -> tuple[list[float], str | None]: ...


optimization = model.optimize(
    query=query,
    trainset=trainset,
    valset=valset,
    max_metric_calls=50,
    metric=my_metric,
)
```

`OptimizationResult.query` is the selected SocialJev query. `OptimizationResult.result` is the AsyncGEPA search result.

[GEPA optimizer reference]: https://gepa-ai.github.io/gepa/api/core/optimize/
[AsyncGEPA continuous rollouts]: https://github.com/DGoettlich/async-gepa/tree/continuous-rollouts
