from typing import Dict, Optional

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

import torch

@Model.register("bbc")
class BBCModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder) -> None:
        super().__init__(vocab)

        self.vocab = vocab

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        n_classes = vocab.get_vocab_size("labels")
        self.linear = torch.nn.Linear(encoder.get_output_dim(), n_classes)

        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            'acc' : CategoricalAccuracy()
        }

    def forward(self, 
        text: Dict[str, torch.tensor],
        category: Optional[torch.tensor] = None) -> Dict[str, torch.Tensor]:

        embedded_text = self.text_field_embedder(text)

        mask = get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)

        logits = self.linear(encoded_text)

        output = {
            'logits': logits
        }

        if category is not None:
            output['loss'] = self.loss(logits, category)
            for metric in self.metrics.values():
                metric(logits, category)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {name: metric.get_metric(reset)
                for name, metric in self.metrics.items()}
