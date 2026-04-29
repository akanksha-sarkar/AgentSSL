# Classification Agent Task Specification

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset. Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**.

You have access to a small training set, a validation set and a set of unlabelled datapoints. Your output will be evaluated using a standardized evaluation script which will return a validation metric as the feedback, but note that the priority is test set performance. 

### Dataset specification
This challenge focused on semi-supervised fine-grained classification where we provide labeled data of the target classes and unlabeled data from both target and non-target classes. The data is obtained from iNaturalist, a community driven project aimed at collecting observations of biodiversity. 

All the images are stored in JPEG format and have a maximum dimension of 300px. 

The training set consists of:
- **Labeled data**: **810** fine-grained species. Every image in this file is paired with a `category_id`. Each **category** entry includes full taxonomy, including **kingdom** and **phylum** (and finer ranks: class, order, family, genus, species). This is the taxonomic information available *from the annotation file* for the labeled set.
- **Unlabeled data**: a large COCO `images` list. In this repository each record has **`id` and `file_name` only**; there are **no** per-image kingdom, phylum, species, or in-class / out-of-class fields in the JSON, and the dataloader only exposes `file_name` and `image_id` for unlabeled images. 

### Guidelines 

You are allowed to use the labeled and unlabeled images in the training set in any way they like. You are not allowed to use images, labels, or pre-trained models from previous iNaturalist competitions (including iNat-17, iNat-18, iNat-19, iNat-21, and Semi-Aves) or any other datasets. However, you are allowed to use ImageNet-1k (not 21k) pre-trained model (e.g., pre-trained models from torchvision). If you cannot verify the training set of a model, we strongly urge you to not use them because of the potential overlap of class labels. The general rule is that participants should only use the provided images and annotations (except for the ImageNet-1k pre-trained models) to train a model.


## Program Interface (Required)

You must implement the following class in your program:

```python

class ClassificationAgent:
    def __init__(self, num_classes):
        self.transform = pass # class must include self.transform
        self.model = pass # The model you predict with.
    def fit(self, img_dir, train_ann_file, unlabel_ann_file):
    """
    Train or initialize the classification model.

    Inputs:
        img_dir: str
            Root directory containing ALL images (both labeled and unlabeled).
            Expected structure:
                img_dir/
                    <image_1>.jpg
                    <image_2>.jpg
                    ...
            Note:
                - Both labeled and unlabeled images are stored in this same directory.
                - Image file names must match those referenced in the annotation files.

        train_ann_file: str
            COCO-format JSON file containing labeled training data.
            Required fields:
                - "images": [{ "id": int, "file_name": str }, ...]
                - "annotations": [{ "image_id": int, "category_id": int }, ...]
                - "categories": [{ "id": int, "name": str }, ...]

        unlabel_ann_file: str
            COCO-format JSON file specifying unlabeled images.
            Required fields:
                - "images": [{ "id": int, "file_name": str }, ...]
                - "categories": [{ "id": int, "name": str }, ...]

            Notes:
                - This file MUST NOT contain "annotations".
                - These images do not have ground-truth labels.
    Returns:
        Optional. May return None.
    """
    pass

    def predict(self, image_batch):
        """
        Predict labels for a batch of PIL images.

        Input:
            image_batch: torch.Tensor

        Returns:
            logits: torch.Tensor
            features: torch.Tensor
            predictions: list[dict] Must have "category_id" key as the label. 
        """
        pass

```

## Usage

The program you write will be used in the following way:
```python
    program = load_program(program_path)

    print("Loaded program...")
    agent = program.ClassificationAgent(num_classes=num_classes)
    print("Created agent...")
    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
    ...
    image = agent.transform(raw_image_tensor)
    logits, feat, predictions = agent.predict(image)
    ....
```