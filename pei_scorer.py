from certainty_estimator.predict_certainty import CertaintyEstimator


class PeiScorer:
    def __init__(self, use_cuda=False):
        self.model = CertaintyEstimator(
            "sentence-level",
            cuda=use_cuda
        )

    def score(self, sentences):
        return self.model.predict(sentences)


if __name__ == "__main__":
    scorer = PeiScorer()

    examples = [
        "The answer is 0.57.",
        "The calculation appears to give a value around 0.57.",
        "This calculation could be interpreted as giving a value around 0.57."
    ]

    scores = scorer.score(examples)

    for text, score in zip(examples, scores):
        print(f"{score:.3f}\t{text}")