from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class Flickr30kDataset(Dataset):

    def __init__(self, split="test"):

        self.dataset = load_dataset(
            "AnyModal/flickr30k",
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