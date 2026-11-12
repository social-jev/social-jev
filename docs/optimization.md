# Optimization

Use `model.optimize()` to improve an existing query and its answer options
from labeled examples. SocialJev evaluates the candidates with your model.
AsyncGEPA samples training examples, proposes rewrites with a reflection model,
and uses validation scores to track the best candidate.

```bash
uv add 'social-jev[optim]'
```

Set `OPENROUTER_API_KEY` to use the default reflector, GLM 5.3 Flash via
OpenRouter and Instructor. Reflection sends sampled training passages, labels,
predictions, and feedback to OpenRouter and can incur API charges. The default
continuous scheduler requires a model configured for async vLLM inference.

## Run an optimization

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

## How it works

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

## Change the defaults

- **Scoring.** The default metric uses the most probable option. It returns
  1 for a correct hard label and 0 otherwise. For a distribution label, it
  returns the target probability of the predicted option. A custom `metric`
  receives a list containing one prediction vector and a list containing one
  label. It returns a one-element score list and optional feedback text.
- **Reflection.** Pass `reflection_prompt_template=` to replace the template,
  `reflection_lm=` to use GEPA's built-in proposer with another model, or
  `custom_candidate_proposer=` to supply your own proposer. Candidate rewrites
  must preserve the original option count and the meaning of each option index.
- **Scheduling.** Pass `max_in_flight_rollouts=` to limit concurrent evaluations.
  An explicit `scheduler=` replaces the default. `scheduler=None` selects GEPA's
  standard loop. Arguments such as `sampling_strategy=` go directly to AsyncGEPA.
