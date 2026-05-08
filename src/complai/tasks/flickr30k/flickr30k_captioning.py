from inspect_ai import task, Task
from inspect_ai.dataset import Sample


from complai.datasets.flickr30k_dataset import Flickr30kDataset
from complai.scorers.caption_metrics import caption_sample
from complai.solvers.image_caption_solver import image_caption_solver


@task(technical_requirement="Robustness and Predictability")
def flickr30k_captioning():
    torch_dataset = Flickr30kDataset(split="test")
    # Convert to InspectAI Dataset format
    samples = []

    for item in torch_dataset:
        samples.append(
            Sample(
                input="Describe the image",
                target=item["caption"],
                metadata={
                    "image": item["image"],
                    "dataset_name": "Flickr30k",
                },
            )
        )

    return Task(
        dataset=samples,
        solver=image_caption_solver(),
        scorer=caption_sample(),
    )





