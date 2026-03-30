from __future__ import annotations
from complai.models.multimodal.load_model import load_model

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class MultimodalModelConfig:
    model_name: str = "CMAlign"
    device: str | None = None
    max_new_tokens: int = 64
    use_fp16: bool = False


class CMAlignModel:
    """
    Plain Python wrapper around your image-captioning model.

    This is NOT the Inspect provider. It is your actual model class.
    The solver will instantiate this class and call generate_from_image().
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        dataset: str | None = None,
        config: MultimodalModelConfig | None = None,
    ) -> None:
        self.config = config or MultimodalModelConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

        if self.config.device is not None:
            self.device = self.config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_components(
        cls,
        model_name: str = "CMAlign",
        dataset: str = "Flickr30k",
    ) -> "CMAlignModel":
        """
        Factory method.

        """
        config = MultimodalModelConfig(model_name=model_name)


        model, tokenizer = load_model(dataset=dataset)



        return cls(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
        )

    def preprocess_image(self, image: Any) -> Any:
        """
        Convert the image into the format expected by your model.

        Supports:
        - PIL.Image
        - torch.Tensor
        - numpy-like arrays if your code already handles them elsewhere
        """
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            # Keep on CPU until just before inference unless your model expects GPU input.
            return image

        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Expected PIL.Image.Image or torch.Tensor."
        )

    def generate_from_image(
        self,
        image: Any,
        dataset: Any | None = None,
        tokenizer: Any | None = None,
    ) -> str:
        """
        This is the function you said you want to use.

        Your original signature:
            model.generate_from_image(image, dataset, tokenizer)

        We preserve that shape here.
        """

        # if image.dim() == 3:
        #     image = image.unsqueeze(0)



        image = self.preprocess_image(image)
        dataset = dataset if dataset is not None else self.dataset
        tokenizer = tokenizer if tokenizer is not None else self.tokenizer

        with torch.no_grad():
            output = self.model.generate_from_image(image, dataset, tokenizer)


        return self._normalize_output(output)

    def _normalize_output(self, output: Any) -> str:
        """
        Convert different output formats into a plain caption string.
        """
        if isinstance(output, str):
            return output.strip()

        if isinstance(output, list):
            if len(output) == 0:
                return ""
            first = output[0]
            if isinstance(first, str):
                return first.strip()
            return str(first).strip()

        if isinstance(output, torch.Tensor):
            # Typical case for token IDs
            if self.tokenizer is None:
                return str(output)
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            return decoded.strip()

        return str(output).strip()



