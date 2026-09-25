import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_ID = "pedropei/sentence-level-certainty"


class PeiScorer:
    def __init__(self, use_cuda=False):
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID
        ).to(self.device)

        self.model.eval()

    def _pad(self, tokens, max_len=60):
        if len(tokens) >= max_len:
            return tokens[:max_len]

        return tokens + [self.tokenizer.pad_token_id] * (
            max_len - len(tokens)
        )

    def _encode(self, texts):
        encoded = []

        for text in texts:
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=60,
                truncation=True,
            )

            encoded.append(self._pad(tokens))

        return torch.tensor(
            encoded,
            dtype=torch.long,
            device=self.device
        )

    def score(self, sentences, batch_size=128):
        if isinstance(sentences, str):
            sentences = [sentences]

        all_scores = []

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            input_ids = self._encode(batch)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)

            scores = (
                outputs.logits
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

            all_scores.extend(scores.tolist())

        return all_scores


if __name__ == "__main__":
    scorer = PeiScorer(use_cuda=False)

    examples = [
        "The answer is 0.57.",
        "The calculation appears to give a value around 0.57.",
        "This calculation could be interpreted as giving a value around 0.57."
    ]

    scores = scorer.score(examples)

    print("\nPei & Jurgens certainty scores")
    print("--------------------------------")

    for text, score in zip(examples, scores):
        print(f"{score:.3f}\t{text}")