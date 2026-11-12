# Optimization

If you have a labeled dataset and SocialJev's predictions fall short, use
`model.optimize()` to search for better wording for your query and its answer
options. The model weights stay fixed. SocialJev runs the predictions and
scores them, while AsyncGEPA uses feedback to propose rewrites and validation
scores to choose among them.

## Setup

```bash
uv add 'social-jev[optim]'
```

Set `OPENROUTER_API_KEY` to use the default reflector, GLM 5.3 Flash via
OpenRouter and Instructor. Reflection sends sampled training passages, labels,
predictions, and feedback to OpenRouter and can incur API charges. The default
continuous scheduler requires a model configured for async vLLM inference.

## Prepare your data and run optimization

Start with the query and ordered options you already use. Split your labeled
dataset into training examples for reflection, validation examples for
selecting a rewrite, and a held-out test set for comparing the original and
optimized queries. The example below shows the record format with a few rows.
Use your own splits for a real run.

```python
from social_jev import Query

query = Query(
    text="Does the author support or oppose expanding bus service?",
    options=["Supports", "Opposes"],
)
trainset = [
    {"state": "The city should add more bus routes.", "label": 0},
    {
        "state": "The council should scrap the expansion.",
        "label": 1,
        "explanation": "The author calls for cancelling the expansion.",
    },
]
valset = [
    {"state": "I welcome the plan to extend bus service.", "label": 0},
    {"state": "Adding more routes would be a waste of money.", "label": 1},
]

optimization = model.optimize(
    query=query,
    trainset=trainset,
    valset=valset,
    max_metric_calls=50,
)

print(optimization.query.text)
print(optimization.query.options)
```

Here `model` is your configured SocialJev instance. Pass lists or GEPA
`DataLoader` instances for `trainset` and `valset`. SocialJev forwards them
directly to AsyncGEPA. Each record needs a `state` and `label`, with an optional
`explanation` for reflection. Integer labels are zero-based option indices, so
`1` means the second option. Distribution labels give the target probabilities
in option order.

## Compare the result with your baseline

Evaluate both queries against the same held-out examples with `model.evaluate()`.
Each example supplies a `state` and a zero-based `label`.

```python
from social_jev.metrics import f1

for name, candidate in [("Baseline", query), ("Optimized", optimization.query)]:
    evaluation = model.evaluate(query=candidate, examples=testset, metric=f1)
    print(f"{name}: {evaluation.score:.1%}")
```

The default metric is accuracy. The example uses binary F1 with the second
option as the positive class. A custom metric receives the whole batch of
probability vectors in query option order and the corresponding labels, and
returns one score. The result also contains `predictions` and `labels` in
input order.

## What happens during optimization

1. **Evaluate a candidate on training examples.** GEPA samples a minibatch.
   SocialJev runs the candidate on those examples and scores the predictions.
2. **Propose a rewrite.** GEPA passes the current query and sampled feedback
   to the reflector. The default prompt asks for general coding rules in the
   query text and short option labels. Instructor validates the proposed query
   and retries invalid responses, such as rewrites with the wrong option count.
3. **Compare and validate.** GEPA tests the rewrite on the same training
   examples. Rewrites that pass can proceed to validation. If you omit `valset`,
   GEPA reuses `trainset`. One metric call is one candidate evaluated on one
   example. Reflection calls are separate from `max_metric_calls`.
4. **Read the result.** `optimization.query` is a SocialJev `Query` with the
   winning text and ordered options, ready to use with `model.query()`.
   `optimization.result` is the original AsyncGEPA `GEPAResult`, with admitted
   candidates, parent links, validation scores, and evaluation counts.

## Choose a budget and scoring rule

- **Evaluation budget.** Set `max_metric_calls` to control how many
  candidate-example evaluations the search can spend. The initial query and
  validation evaluations count toward that budget. Start small to check that
  scoring and reflection work, then increase it to give GEPA room to test
  more rewrites. The budget excludes reflection requests.
- **Validation set size.** Use a validation set that represents the cases you
  care about. A larger set gives GEPA more evidence for selecting rewrites and
  uses more of the budget each time it validates a candidate.
- **Reflection minibatch size.** Set `reflection_minibatch_size` to change how
  many training examples GEPA samples for a reflection step. Increase it when
  the reflector needs to see more varied failures in one step. Larger
  minibatches use more evaluations and produce longer reflection prompts.

- **Scoring.** The default metric uses the most probable option. It returns
  1 for a correct hard label and 0 otherwise. For a distribution label, it
  returns the target probability of the predicted option. A custom `metric`
  receives a list containing one prediction vector and a list containing one
  label. It returns a one-element score list and optional feedback text.
- **Reflection.** Pass `reflection_prompt_template=` to replace the template,
  `reflection_lm=` to use GEPA's built-in proposer with another model, or
  `custom_candidate_proposer=` to supply your own proposer. Candidate rewrites
  must preserve the original option count and the meaning of each option index.

## Control asynchronous rollouts

SocialJev enables AsyncGEPA's continuous rollouts by default. A rollout is one
query candidate evaluated on one example. As rollouts finish, the scheduler
can start more while other rollouts or reflection requests are still running.
This can keep a self-hosted vLLM model busy when examples take different amounts
of time or the reflector is generating a rewrite.

- `max_in_flight_rollouts` (default 64) limits concurrent candidate-example
  evaluations. Lower it when the inference server is saturated or requests
  queue too long.
- `max_in_flight_proposals` (default 8) limits the number of proposals in
  progress. Lower it to reduce overlapping work and pressure on the reflector.
- `max_proposal_age` (default 3) limits how many candidate-pool updates may
  happen while a proposal is in progress. Lower it if you want proposals based
  on more recent search state.
- Pass `scheduler=None` to use the standard GEPA loop. Pass a scheduler
  instance to replace the default continuous scheduler.

SocialJev forwards the remaining optimization arguments, including
`sampling_strategy`, to AsyncGEPA. See the [GEPA optimize reference] for the
full list of shared parameters and the [AsyncGEPA package] and its
[continuous rollout settings] for the asynchronous extension.

[GEPA optimize reference]: https://gepa-ai.github.io/gepa/api/core/optimize/
[AsyncGEPA package]: https://github.com/DGoettlich/async-gepa/tree/continuous-rollouts
[continuous rollout settings]: https://github.com/DGoettlich/async-gepa/blob/continuous-rollouts/src/async_gepa/core/continuous_rollouts.py
