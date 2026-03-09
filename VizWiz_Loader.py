import torch
from torchvision.transforms import v2
from PIL import Image

import os
import json
from typing import Callable, Dict, List, Sequence

from torch.utils.data import DataLoader

class VizWizLoader(torch.utils.data.Dataset):
    def __init__(self, strFolder: str, strAnnotationPath: str, fDataPercentage: float = 1.0,
                 tTransform: Callable[[torch.tensor], torch.tensor] = None) -> None:
        '''
        strFolder: path to unzip'd folder of VizWiz images
        strLabelPath: path to .json file containing the annotations
        fDataPercentage: percentage of available samples to use. Must be normalized between 0.0 and 1.0. Default: 1.0
        tTransform: optional place to connect PyTorch image transformations. Default: converts images to 3x224x224 tensors
        For the train and val splits, returns tuples of the form:
            (image, question text, binary label, answer texts)
        Otherwise, returns tuples of the form:
            (image, question text)
        '''
        self.strFolder = strFolder
        if self.strFolder[-1] != "/": self.strFolder += "/"
        self.tTransform = tTransform
        vecPaths = os.listdir(self.strFolder)
        self.strPrefix = vecPaths[0].split("_")[1]

        with open(strAnnotationPath, "r") as f:
            self.vecAnnos = json.load(f)
        
        self.iN = int(fDataPercentage * len(self.vecAnnos))

        if self.tTransform is None:
            self.tTransform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True), v2.RandomCrop(224)])
            self.tTransformUndersized = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True), v2.Resize((224, 224))])

        return

    def __len__(self) -> int: return self.iN

    def __getitem__(self, idx: int) -> tuple:
        if idx > self.iN:
            print(f"Error! Tried to access index {idx} but only {self.iN} samples are available")
            return None
        
        strPath = self.strFolder + self.vecAnnos[idx]["image"]
        imX = Image.open(strPath)
        w, h = imX.size
        if w >= 224 and h >= 224: tX = self.tTransform(imX)
        else: tX = self.tTransformUndersized(imX)

        if self.strPrefix == "test":
            return tX, self.vecAnnos[idx]["question"]
        else:
            return tX, self.vecAnnos[idx]["question"], self.vecAnnos[idx]["answerable"], self.vecAnnos[idx]["answers"]


def build_simple_tokenizer(question_texts: Sequence[str]) -> Callable[[str, int], List[int]]:
    """Builds a whitespace tokenizer with fixed ids: 0=PAD, 1=UNK."""
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    for text in question_texts:
        for token in str(text).lower().strip().split():
            if token and token not in vocab:
                vocab[token] = len(vocab)

    def tokenize(text: str, max_len: int) -> List[int]:
        tokens = str(text).lower().strip().split()
        ids = [vocab.get(token, 1) for token in tokens][:max_len]
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        return ids

    return tokenize


class VizWizDatasetWithText(torch.utils.data.Dataset):
    """Wraps VizWiz samples and adds fixed-length tokenized questions."""

    def __init__(
        self,
        base_dataset: VizWizLoader,
        text_tokenizer: Callable[[str, int], List[int]],
        max_len: int = 32,
    ) -> None:
        self.base_dataset = base_dataset
        self.text_tokenizer = text_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.base_dataset[idx]

        if len(sample) == 2:
            image, question_text = sample
            question_text = str(question_text).lower().strip()
            question_ids = self.text_tokenizer(question_text, self.max_len)
            return {
                "image": image,
                "question_ids": question_ids,
            }

        image, question_text, answerable_label, answers_list = sample
        question_text = str(question_text).lower().strip()
        question_ids = self.text_tokenizer(question_text, self.max_len)

        canonical_answer = ""
        for answer_item in answers_list:
            if isinstance(answer_item, dict):
                candidate = str(answer_item.get("answer", "")).lower().strip()
            else:
                candidate = str(answer_item).lower().strip()
            if candidate:
                canonical_answer = candidate
                break

        return {
            "image": image,
            "question_ids": question_ids,
            "answerable": int(answerable_label),
            "target_answer": canonical_answer,
        }


if __name__ == "__main__":
    train_images_folder = "path/to/VizWiz/train/"
    train_annotations_path = "path/to/Annotations/train.json"

    if os.path.isdir(train_images_folder) and os.path.isfile(train_annotations_path):
        train_base = VizWizLoader(strFolder=train_images_folder, strAnnotationPath=train_annotations_path)
        train_questions = [str(train_base[i][1]).lower().strip() for i in range(len(train_base))]
        tokenizer = build_simple_tokenizer(train_questions)
        train_dataset = VizWizDatasetWithText(
            base_dataset=train_base,
            text_tokenizer=tokenizer,
            max_len=32,
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)