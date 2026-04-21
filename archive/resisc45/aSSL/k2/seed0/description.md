# Classification Agent Task Specification
### TESTING

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset.  
Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a limited training set, a validation set and a set of unlabelled datapoints.

Your output will be evaluated using a standardized evaluation script that computes classification metrics (e.g., top-1 accuracy).

### Dataset specification 
A benchmark dataset of aerial RGB images depicting diverse land-use and scene categories. It contains 45 classes (e.g., urban areas, transportation infrastructure, natural landscapes). Each image is 256x256 pixels, and the dataset is balanced across classes. 

### Metric specification 
The feedback metric is Adjusted Mutual Information (AMI), an unsupervised clustering consistency metric. **No ground truth labels are used.** 
AMI measures how well the model’s softmax predictions agree with the geometric structure of its own feature space — i.e., internal representation consistency.

Range:
	- 1 → perfect agreement between predictions and feature clusters
	- 0 → no better than chance
	- < 0 → worse than chance

In practice: a high AMI means the model’s predicted classes align with natural groupings in its feature space, which correlates with learning useful representations.

---

## Program Interface (Required)

You must implement the following class in your program:

```python

class ClassificationAgent:
    def __init__(self):
        pass

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