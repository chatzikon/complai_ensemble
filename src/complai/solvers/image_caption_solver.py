from __future__ import annotations

from inspect_ai.solver import solver
from inspect_ai.model import ModelOutput
from complai.models.multimodal.multimodal_model import CMAlignModel


_MODEL_CACHE = {}


def _get_model(dataset_name: str) -> CMAlignModel:
    if dataset_name not in _MODEL_CACHE:
        _MODEL_CACHE[dataset_name] = CMAlignModel.from_components(
            model_name="my_model",
            dataset=dataset_name,
        )
    return _MODEL_CACHE[dataset_name]

@solver
def image_caption_solver():
    async def solve(state, generate):


        image = state.metadata["image"]

        dataset_name = state.metadata.get("dataset_name", "Flickr30k")

        model = _get_model(dataset_name)


        caption = model.generate_from_image(image)

        state.output = ModelOutput(completion=str(caption))
        return state

    return solve