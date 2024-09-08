import torch
import torch.nn.functional as F
from torch import Tensor


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def text_embedding(text: list[str], model, tokenizer) -> Tensor:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    embeddings = average_pool(last_hidden_states, attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings
