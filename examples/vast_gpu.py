"""Rent a GPU on Vast.ai, process requests, and destroy the instance on exit."""

from social_jev import Query, Request, SocialJev
from social_jev.integrations import VastAI

if __name__ == "__main__":
    request = Request(
        state="The council expanded the bus network.",
        queries=[Query(text="Which policy area is discussed?", options=["Transport", "Education"])],
    )
    with VastAI(gpu="H100_SXM", max_hourly_price=2.0):
        model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview", backend="vllm")
        print(model.query(request)[0][0].choice)
