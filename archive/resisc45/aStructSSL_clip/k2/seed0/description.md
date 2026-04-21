# Classification Agent Task Specification

## Objective

You are an AI agent tasked with implementing a **semi-supervised image classification program**.

Your goal is to:

* Train a classifier using **limited labeled data + abundant unlabeled data**
* Predict labels for all validation images

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
However, the input to this pipeline will be a fixed backbone which is described below. 

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
    def __init__(self, backbone):
        self.label_gen = LabelGenerator()
        self.aug = Augmentation()
        self.filter = Filter()
        self.ssl = SSL_Loss()
        self.backbone = backbone
        self.model = None # Use self.backbone
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

## Usage

The program you write will be used in the following way:
```python
    program = load_program(program_path)

    import timm

    backbone = timm.create_model(
        "vit_base_patch16_clip_224",
        pretrained=True,
        num_classes=45
    )
    print("Loaded program...")
    agent = program.SSL_Algorithm(backbone=backbone)
    print("Created agent...")
    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
```

## Backbone (Provided)

As you can see above, a pretrained clip backbone will be provided to your agent as:

    self.backbone

Requirements:
- You MUST use this backbone for feature extraction
- You MUST NOT instantiate a new backbone
- You may add additional layers (e.g., classification head, adapters, LoRA)

The backbone outputs features via:
    features = self.backbone.forward_features(x)

## Success Threshold

A simple LoRA few-shot supervised fine-tuning of the provided backbone using only labeled data gets **80% classification accuracy**. If you are not able to beat this that is concerning.
