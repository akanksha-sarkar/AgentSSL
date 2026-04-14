# Classification Agent Task Specification
### TESTING

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset.  
Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a limited training set, a validation set and a set of unlabelled datapoints. CLIP and DINO-style pretrained backbones are likely to perform well on this classification task. Parameter-efficient methods such as LoRA (accessed through the peft library) or adaptformer may be particularly effective for adapting these large models under limited supervision.

Your output will be evaluated using a standardized evaluation script which will return a set of unsupervised proxy metrics (ami, ari, v_measure, fmi, silhouette) as the feedback, but note that the priority is test set performance. At the end of the evolution, the final program will be evaluated on the validation set. 

### Dataset specification 
A benchmark dataset of aerial RGB images depicting diverse land-use and scene categories. It contains 45 classes (e.g., urban areas, transportation infrastructure, natural landscapes). Each image is 256x256 pixels, and the dataset is balanced across classes. 

---

## Program Interface (Required)

You must implement the following class in your program:

```python

class ClassificationAgent:
    def __init__(self):
        self.transform = pass # class must include self.transform

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
            predictions: list[dict]
        """
        pass

```
