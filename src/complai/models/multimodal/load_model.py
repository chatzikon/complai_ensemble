import torch
from complai.models.multimodal.vae import MultimodalVAE
from transformers import  AutoTokenizer


class Tokenizer:
    def __init__(self, max_length, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.max_length=max_length

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

def load_model(dataset):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")

    if dataset == 'Flickr30k':
        model_checkpoint=('/home/chatziko/PycharmProjects/PythonProject/checkpoints/final_model_kl_coef_1.0_lr_0.01_latent_dim_64_scheme_a.pt')
        num_attributes = 32
        tokenizer = Tokenizer(32, AutoTokenizer.from_pretrained("facebook/bart-base"))
        vocab_size = tokenizer.tokenizer.vocab_size
        latent_dim = 64

    elif dataset == "CelebAMask-HQ":
        model_checkpoint = ('/home/chatziko/PycharmProjects/PythonProject/checkpoints/final_model_kl_coef_1.pt')
        num_attributes = 10
        tokenizer=None
        vocab_size = 0
        latent_dim = 256


    checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)


    model = MultimodalVAE(device,tokenizer, vocab_size, 'a', dataset,
        latent_dim=latent_dim, num_attributes=num_attributes,
        temperature=1.0).to(device)


    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    return model, tokenizer
