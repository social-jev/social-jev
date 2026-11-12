"""Run one query on Modal with social-jev[compute] and a configured Modal account."""

from social_jev import Query, Request, SocialJev
from social_jev.integrations import Modal

if __name__ == "__main__":
    request = Request(
        state="The council expanded the bus network.",
        queries=[Query(text="Which policy area is discussed?", options=["Transport", "Education"])],
    )

    with Modal(gpu="L40S"):
        model = SocialJev.from_pretrained("social-jev/social-jev-0.6B-preview")
        print(model.query(request)[0][0].choice)
