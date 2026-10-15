"""LM Evaluation Harness adapter for AgentGate-Base-142M (multiple-choice tasks)."""

import torch
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from src.model import AgentGate, AgentGateConfig


class AgentGateLM(TemplateLM):
    def __init__(self, checkpoint, tokenizer_path, device="cuda"):
        super().__init__()

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu.")
        self._device = torch.device(device)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._eot_id = self.tokenizer.token_to_id("<|endoftext|>")
        if self._eot_id is None:
            raise ValueError("Tokenizer is missing <|endoftext|>.")

        self.model = AgentGate(AgentGateConfig())
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.to(self._device).eval()
        self.max_length = self.model.config.context_length

    @property
    def eot_token_id(self):
        return self._eot_id

    def tok_encode(self, string, add_special_tokens=False, **kwargs):
        return self.tokenizer.encode(
            string, add_special_tokens=bool(add_special_tokens)
        ).ids

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, **kwargs):
        results = []

        for _, context_ids, continuation_ids in tqdm(
            requests, desc="Scoring candidates", unit="candidate"
        ):
            n = len(continuation_ids)

            if not context_ids or not n:
                raise ValueError("Context and continuation must contain tokens.")
            if n > self.max_length:
                raise ValueError("Continuation exceeds model context length.")

            ids = (context_ids + continuation_ids)[-(self.max_length + 1) :]

            inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=self._device)
            targets = torch.tensor([ids[1:]], dtype=torch.long, device=self._device)

            logits, _ = self.model(inputs)

            candidate_logits = logits[:, -n:, :].float()
            candidate_targets = targets[:, -n:]

            log_probs = F.log_softmax(candidate_logits, dim=-1)
            token_scores = log_probs.gather(
                -1, candidate_targets.unsqueeze(-1)
            ).squeeze(-1)

            is_greedy = (candidate_logits.argmax(-1) == candidate_targets).all().item()

            results.append((float(token_scores.sum().item()), bool(is_greedy)))

        return results

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError("Rolling log-likelihood is not supported.")

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError("Generation is not supported.")
