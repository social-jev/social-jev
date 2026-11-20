"""Run one query on a Hugging Face GPU Job."""

from social_jev import Query, Request, SocialJev
from social_jev.integrations import HuggingFaceJobs

if __name__ == "__main__":
    request = Request(
        state="The council expanded the bus network.",
        queries=[Query(text="Which policy area is discussed?", options=["Transport", "Education"])],
    )
    with HuggingFaceJobs():
        model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview", backend="vllm")
        print(model.query(request)[0][0].choice)
