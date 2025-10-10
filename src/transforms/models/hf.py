from typing import Mapping, Any

from transformers import PreTrainedModel

from src.transforms.base import TransformWrapper


class HFModelTransform(TransformWrapper):
    def __init__(
        self,
        model: PreTrainedModel,
        transform_kwargs: Mapping[str, Any] = {},
        mapping: Mapping[str, str] = {},
    ):
        super().__init__(self.model_forward, transform_kwargs, mapping)
        self.model = model

    def model_forward(self, **kwargs) -> Mapping[str, Any]:
        output = self.model(**kwargs)
        return dict(output)
