import os
import json
from PIL import Image
from torch.utils.data import Dataset
import random

class OSMTextImageDataset(Dataset):
    def __init__(self, image_root_dirs, json_file_paths, preprocessor=None, processor=None):
        self.image_root_dirs = image_root_dirs if isinstance(image_root_dirs, list) else [image_root_dirs]
        self.json_file_paths = json_file_paths if isinstance(json_file_paths, list) else [json_file_paths]
        self.preprocessor = preprocessor
        self.processor = processor

        self.image = []
        self.text = []
        self.img2txt = {}
        self.txt2img = {}

        for json_file_path, image_root_dir in zip(self.json_file_paths, self.image_root_dirs):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            """
            for idx, item in enumerate(json_data):
                image_name = os.path.basename(item['bev_image'])
                caption = item['caption']

                self.image.append(os.path.join(image_root_dir, image_name))
                self.text.append(caption)
                
                self.img2txt[len(self.image) - 1] = [len(self.text) - 1]
                self.txt2img[len(self.text) - 1] = len(self.image) - 1
            """
            for idx, (image_name, caption) in enumerate(json_data.items()):

                self.image.append(os.path.join(image_root_dir, image_name))
                self.text.append(caption)

                self.img2txt[len(self.image) - 1] = [len(self.text) - 1]
                self.txt2img[len(self.text) - 1] = len(self.image) - 1

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        image_ = Image.open(self.image[idx]).convert("RGB")
        text = self.text[idx]

        if self.preprocessor:
            image, text_id = self.preprocessor(image_, text)

        if self.processor:
            output = self.processor(image_, text, return_tensors="pt", max_length=40, padding='max_length', truncation=True)
            for key, value in output.items():
                value = value.squeeze(0)
            return output
        
        
        return {
            'image': image,
            'text': text_id,
            'original_text': text,
            'image_path': self.image[idx],
            'idx': idx
        }


class JsonTextImageDataset(Dataset):
    """Dataset for the train/test json format provided in the user request.

    Each entry in the json file is a dictionary with at least the following keys:

    - ``image``: relative path to the image file
    - ``image_id``: identifier in the form ``<building>/<filename>``
    - ``caption``: either a string or a list of strings

    This dataset exposes fields compatible with the evaluation functions
    defined in :mod:`evaluate.py` such as ``image``, ``text``, ``img2txt`` and
    ``txt2img``.  In addition it provides ``img2building`` which maps each image
    index to a building index extracted from ``image_id``.
    """

    def __init__(self, json_path, image_root_dir="", preprocessor=None, processor=None):
        self.preprocessor = preprocessor
        self.processor = processor
        self.image_root_dir = image_root_dir

        with open(json_path, "r") as f:
            data = json.load(f)

        self.image = []
        self.text = []
        self.img2txt = {}
        self.txt2img = {}
        self.img2building = []

        building_map = {}

        for idx, item in enumerate(data):
            img_path = item.get("image", item.get("image_id"))
            if self.image_root_dir and not os.path.isabs(img_path):
                img_path = os.path.join(self.image_root_dir, img_path)
            caption = item.get("caption", "")

            if isinstance(caption, list):
                caption = " ".join(caption)

            self.image.append(img_path)
            self.text.append(caption)

            self.img2txt[idx] = [idx]
            self.txt2img[idx] = idx

            building_id = str(item.get("image_id", "")).split("/")[0]
            bidx = building_map.setdefault(building_id, len(building_map))
            self.img2building.append(bidx)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        image_ = Image.open(self.image[idx]).convert("RGB")
        text = self.text[idx]

        image, text_id = None, None

        if self.preprocessor:
            image, text_id = self.preprocessor(image_, text)

        if self.processor:
            output = self.processor(image_, text, return_tensors="pt",
                                   max_length=40, padding="max_length",
                                   truncation=True)
            for key, value in output.items():
                output[key] = value.squeeze(0)
            return output

        return {
            "image": image,
            "text": text_id,
            "original_text": text,
            "image_path": self.image[idx],
            "idx": idx,
        }


def load_text_image_dataset(json_path, image_root_dir="", preprocessor=None, processor=None):
    """Utility to choose the appropriate dataset class based on JSON structure."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return JsonTextImageDataset(json_path, image_root_dir=image_root_dir or "", preprocessor=preprocessor, processor=processor)
    else:
        return OSMTextImageDataset(image_root_dir, json_path, preprocessor=preprocessor, processor=processor)
