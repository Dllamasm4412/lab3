import torch
from torchvision.transforms import v2
from PIL import Image

import os
import json
from typing import Callable

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