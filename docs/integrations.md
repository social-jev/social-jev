# Rented GPUs

SocialJev runs inference on Modal, Hugging Face Jobs, or Vast.ai through a provider context. `SocialJev.from_pretrained()` starts the remote worker and loads the model there. The worker tokenizes requests, runs inference, and returns the results to your script. The calling script loads the dataset and handles saving or uploading results. Exiting the context releases the worker.

## Setup

```bash
# Works with Transformers and vLLM on the remote GPU
uv pip install 'social-jev[compute]'
```

### Modal

`modal setup` connects the local environment to a Modal workspace. [Modal setup](https://modal.com/docs/guide/getting-started)

### Hugging Face Jobs

`hf auth login` authenticates the local client. The token needs permission to start and manage Jobs. [Hugging Face Jobs setup](https://huggingface.co/docs/hub/jobs-configuration)

### Vast.ai

The local client needs a Vast API key and an SSH key registered with Vast. [Vast API and SSH setup](https://github.com/vast-ai/docs/blob/main/api-reference/openapi.yaml)

## Run inference

```python
from social_jev import Query, Request, SocialJev
from social_jev.integrations import Modal

request = Request(
    state="The council should add more bus routes.",
    queries=[
        Query(
            text="Does the author support expanding bus service?",
            options=["Supports", "Opposes"],
        ),
    ],
)

with Modal(gpu="H100!"):
    model = SocialJev.from_pretrained(
        "social-jev/social-jev-0.6B-preview", backend="vllm"
    )
    results = model.query(request)

print(results[0][0].choice)
```

The same query code works with `HuggingFaceJobs()` or `VastAI()`. With vLLM, `aquery()` also lets the calling script submit overlapping calls when the provider configuration allows them.

## Environment variables and secrets

If the remote worker needs access to a private model, it needs a token with permission to download it.

**Modal** can attach a workspace secret named `hub-access` that contains `HF_TOKEN`. [Modal secrets](https://modal.com/docs/guide/secrets)

```python
import modal
from social_jev.integrations import Modal

provider = Modal(secrets=[modal.Secret.from_name("hub-access")])
```

**Hugging Face Jobs** can copy a local token into the job's secrets. Jobs sets `HF_TOKEN` before the remote Python process starts. [Jobs secrets](https://huggingface.co/docs/hub/jobs-configuration)

```python
import os
from social_jev.integrations import HuggingFaceJobs

provider = HuggingFaceJobs(secrets={"HF_TOKEN": os.environ["HF_READ_TOKEN"]})
```

**Vast.ai** injects variables such as `HF_TOKEN` from **Account Settings → Environment Variables** into every container the account launches. The rental's `env` option can supply an ordinary variable for one container. [Vast container environment](https://github.com/vast-ai/docs/blob/main/guides/instances/docker-environment.mdx)

```python
from social_jev.integrations import VastAI

provider = VastAI(env="-e HF_HUB_VERBOSITY=warning")
```

## API reference

Provider constructors accept settings for the rented worker. Model and inference engine options go on `SocialJev.from_pretrained()`.

### `Modal(**kwargs)`

`max_inputs` limits simultaneous calls per container and defaults to `1`. Options such as `gpu`, `image`, `memory`, and `timeout` go to Modal. [Modal worker options](https://modal.com/docs/reference/modal.App#cls) · [Input concurrency](https://modal.com/docs/guide/concurrent-inputs)

```python
provider = Modal(gpu="H100!", max_inputs=8, timeout=1800)
```

### `HuggingFaceJobs(**kwargs)`

Options such as `flavor`, `image`, `timeout`, and `secrets` go to Hugging Face Jobs. SocialJev supplies the worker command and sets up an authenticated endpoint. [Jobs configuration](https://huggingface.co/docs/hub/jobs-configuration)

```python
provider = HuggingFaceJobs(flavor="h200", timeout="30m")
```

### `VastAI(*, max_hourly_price=None, startup_timeout=600, **kwargs)`

`gpu`, `num_gpus`, and `max_hourly_price` constrain offer selection. `startup_timeout` sets how many seconds SocialJev waits for the worker. Rental options such as `image` and `env` go to Vast. `disk` also affects offer selection. [Vast instance API](https://github.com/vast-ai/docs/blob/main/api-reference/openapi.yaml)

```python
provider = VastAI(gpu="H100_SXM", max_hourly_price=3.0)
```
