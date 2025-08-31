from tiktoken import Encoding
from torch import Tensor, tensor, device
from torch.cuda import is_available
from torch.utils.data import Dataset

from src.settings import Settings


class GPTDataset(Dataset):

    input_ids: list[Tensor]
    target_ids: list[Tensor]

    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int):
        device("cuda" if is_available() else "cpu")
        self.input_ids = list()
        self.target_ids = list()

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={Settings.end_of_tokens})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(tensor(input_chunk))
            self.target_ids.append(tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
