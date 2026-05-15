from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

import os
#os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"]='1'
CHECKPOINT_DIR = os.getenv("HF_HOME", "/.cache/huggingface/")


class Flickr30kDataset(Dataset):

    def __init__(self, split="test"):

        self.dataset = load_dataset(
            "AnyModal/flickr30k",
            token=os.environ.get("HF_TOKEN"),
            cache_dir="HF_HOME",
            split=f"{split}[:100]"
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        #image = item["image"].convert("RGB")
        image = item["image"]
        image = self.transform(image)

        caption = item["alt_text"][0]

        return {
            "image": image,
            "caption": caption
        }