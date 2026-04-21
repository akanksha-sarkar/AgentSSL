# Classification Agent Task Specification

## Objective

You are an AI agent tasked with implementing a **semi-supervised image classification program**.

Your goal is to:

* Train a classifier using **limited labeled data + abundant unlabeled data**
* Predict labels for all validation images

You may use pretrained backbones (e.g., CLIP, DINO) and parameter-efficient fine-tuning (e.g., LoRA, AdaptFormer).

---

## Evaluation

* During development, you receive **unsupervised proxy feedback** (e.g., adjusted mutual information).
* Final evaluation is based on **validation set classification accuracy**.

---

## Dataset

* Aerial RGB images (256×256)
* 45 balanced classes (land-use / scene categories)

---

## Task Structure

You will design a semi-supervised learning (SSL) pipeline using modular components.

Typical SSL components include:

* **Pseudo-labeling**
* **Data augmentation**
* **Filtering / confidence selection**
* **Consistency regularization**
* **Loss design**

These components are **intentionally flexible** — you are free to implement them in any way.

---

## Required Interface

You must implement the following classes:

```python
class LabelGenerator:
    def fit(self, model, labeled_data):
        """Optional: initialize or pretrain label generator."""

    def generate(self, model, x):
        """
        Generate pseudo-labels.
        Requirements:
            - Use torch.no_grad()
            - Return pseudo_labels and metadata
        """

    def update(self, student_model, step):
        """Optional: EMA update, ensemble refresh, etc."""


class Augmentation:
    def generate_views(self, x):
        """
        Generate one or more views of input.
        Example outputs:
            {"weak": x_w, "strong": x_s}
        """


class Filter:
    def apply(self, pseudo_labels, metadata):
        """
        Select reliable pseudo-labels.
        Examples:
            - confidence thresholding
            - probabilistic filtering
            - no filtering
        """

class SSL_Loss:
    def supervised(self, preds, y_true):
        """Supervised loss"""

    def unsupervised(self, preds, targets, mask):
        """Unsupervised (pseudo-label) loss"""
    
    def consistency(self, model, label_gen, views, pseudo_labels=None):
        """Optional consistency regularization loss."""

    def total(self, L_sup, L_unsup, L_cons):
        """Combine losses"""


class SSL_Algorithm:
    def __init__(self):
        self.label_gen = LabelGenerator()
        self.aug = Augmentation()
        self.filter = Filter()
        self.ssl = SSL_Loss()
        self.model = None
        self.transform = None # Transform
    def ssl_step(self, model, batch_l, batch_u, step):
        """
        Perform one SSL training step using all components.
        """

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

    def predict(self, image_batch):
        """
        Predict labels for a batch.

        Returns:
            logits: torch.Tensor
            features: torch.Tensor
            predictions: list[dict], use "category_id" as a key
        """
```

---

## Key Expectations

* Use unlabeled data effectively
* Design a **coherent SSL pipeline**, not isolated tricks
* Ensure components interact correctly (e.g., augmentation → pseudo-labeling → filtering → loss)
* Prioritize final classification performance over proxy metrics
