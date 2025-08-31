from os import listdir
from tiktoken import get_encoding, Encoding
from torch import Tensor, tensor
from torch.cuda import is_available
from torch.utils.data import DataLoader
from tqdm import tqdm
from re import sub

from src.settings import Settings
from src.dataload.dataset import GPTDataset


class Tokenizer:

    encondings: list[str]
    tokenizer: Encoding

    def __init__(self, encondings: list[str]) -> None:
        self.encondings = encondings
        self.tokenizer = get_encoding("gpt2")

    def __read_content_path(self, file_path: str) -> str:
        for enconding in self.encondings:
            try:
                with open(file_path, "r", encoding=enconding) as f:
                    return f.read()
            except UnicodeDecodeError:
                print(
                    f"Warning: UnicodeDecoreError encountered. File: {file_path}, Encode: {enconding}"
                )
        raise RuntimeError("Encondings on Tokenizer not valid")

    def __create_dataloader(
        self,
        txt: str,
        batch_size: int,
        max_length: int,
        stride: int,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ) -> DataLoader:
        dataset = GPTDataset(txt, self.tokenizer, max_length, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return dataloader

    def create_dataloaders(
        self,
        data: str,
        train_ratio: float,
        batch_size: int,
        max_length: int,
        stride: int,
        num_workers=0,
    ) -> tuple[DataLoader, DataLoader]:
        split_idx = int(train_ratio * len(data))

        train_loader = self.__create_dataloader(
            data[:split_idx],
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = self.__create_dataloader(
            data[split_idx:],
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            drop_last=False,
            shuffle=False,
            num_workers=num_workers,
        )

        return train_loader, val_loader

    def generate_data(
        self,
        data_set_dir: str,
        target_path: str,
    ) -> None:
        current_content = list()

        file_paths = listdir(data_set_dir)

        for file_path in tqdm(file_paths):
            content_brute = self.__read_content_path(f"{data_set_dir}{file_path}")
            content_single_blank = sub(r"\n\s*\n", "\n\n", content_brute)
            current_content.append(content_single_blank)

        if current_content:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(Settings.end_of_tokens.join(current_content))

    def text_to_token_ids(
        self, text: str, device_type: str = "cuda" if is_available() else "cpu"
    ) -> Tensor:
        enconded = self.tokenizer.encode(text)
        enconded_tensor = tensor(enconded).unsqueeze(0)
        return enconded_tensor.to(device_type)

    def token_ids_to_text(self, token_ids: Tensor) -> str:
        flat = token_ids.squeeze(0).tolist()
        texts = self.tokenizer.decode(flat)
        return "".join(texts)
