# Classification Agent Task Specification
### TESTING

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset.  
Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a limited training set, a validation set and a set of unlabelled datapoints. CLIP and DINO-style pretrained backbones are likely to perform well on this classification task. Programs that use LoRA-based adaptation of pretrained CLIP or DINO-style backbones are especially encouraged. When using a transformer-based backbone, the agent should preferentially consider inserting LoRA adapters into attention projections or MLP layers rather than relying only on a frozen encoder with a linear probe.


Your output will be evaluated using a standardized evaluation script which will return a set of unsupervised proxy metrics (ami, ari, v_measure, fmi, silhouette) as the feedback, but note that the priority is test set performance. At the end of the evolution, the final program will be evaluated on the validation set. 

### Dataset specification 
A synthetic image dataset consisting of simple 3D scenes populated with geometric objects such as spheres, cubes, and cylinders. Each object varies along interpretable attributes including color, size, material, and spatial position. For the classification task, labels are derived from object counts in the scene. 
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
