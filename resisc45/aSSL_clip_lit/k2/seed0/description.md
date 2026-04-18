# Classification Agent Task Specification
### TESTING

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset.  
Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a limited training set, a validation set and a set of unlabelled datapoints. 

Your output will be evaluated using a standardized evaluation script which will return an unsupervised metric (adjusted mutual information) as the feedback. At the end of the evolution, the final program will be evaluated on the validation set. 

### Dataset specification 
A benchmark dataset of aerial RGB images depicting diverse land-use and scene categories. It contains 45 classes (e.g., urban areas, transportation infrastructure, natural landscapes). Each image is 256x256 pixels, and the dataset is balanced across classes. 

---

## Program Interface (Required)

You must implement the following class in your program:

```python

class ClassificationAgent:
    def __init__(self, backbone):
        self.backbone = backbone
        self.transform = None

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
    agent = program.ClassificationAgent(backbone=backbone)
    print("Created agent...")
    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
```

## Backbone (Provided)

As you can see above, a pretrained clip backbone will be provided to your agent as:

    self.backbone

Requirements:
- You MUST use this backbone for feature extraction
- You MUST NOT instantiate a new backbone
- You may add additional layers and finetune any parts (e.g., classification head, adapters, LoRA)

The backbone outputs features via:
    features = self.backbone.forward_features(x)

## Success Threshold

A simple LoRA few-shot supervised fine-tuning of the provided backbone using only labeled data gets **80% classification accuracy**. If you are not able to beat this that is concerning.

## Relevant Literature

SSL is a learning paradigm associated with constructing models that use both labeled and unlabeled data. SSL methods can improve learning performance by using additional unlabeled instances compared to supervised learning algorithms, which can use only labeled data. We want to focus on the classification task and avoid transductive learning based methods as we want to learn a strong classifier. 

### CONSISTENCY REGULARIZATION
In this section, we introduce the consistency regularization methods for semi-supervised deep learning. In these methods, a consistency regularization term is applied to the final loss function to specify the prior constraints assumed by researchers. Consistency regularization is based on the manifold assumption or the smoothness assumption, and describes a category of methods that the realistic perturbations of the data points should not change the output of the
model. Consequently, consistency regularization can be regarded to find a smooth manifold on which the dataset lies by leveraging the unlabeled data. Some examples of these methods include, 
Student-Teacher, PI-Model, Ladder Network, Mean-Teacher, etc.

### PSEUDOLABELING METHODS
The pseudo-labeling methods differ from the consistency
regularization methods in that the consistency regularization methods usually rely on consistency constraint of rich data transformations. In contrast, pseudo-labeling methods rely on the high confidence of pseudo-labels, which can be added to the training data set as labeled data. There are two main patterns, one is to improve the performance of the whole framework based on the disagreement of views or multiple networks, and the other is self-training, in particular, the success of self-supervised learning in unsupervised domain makes some self-training self-supervised methods realized. Some examples of relevant methods are disagreement-based models or self-training. 

### HYBRID METHODS

These methods combine ideas from the above-mentioned methods such as pseudo-label, consistency regularization and entropy minimization for performance improvement. Some examples include, MixMatch, FixMatch, SoftMatch, FlexMatch, etc. These methods are generally considered to be SOTA.

### SSL with Foundation Models

While previous methods trained models from scratch, several recent works have explored the effects of large pre-trained foundation models on SSL.

Paper 1 Findings: 
Semi-supervised learning (SSL) enhances model performance by leveraging abundant unlabeled data alongside limited labeled data. As vision foundation models
(VFMs) become central to modern vision applications, this paper revisits SSL in
the context of these powerful pre-trained models. We conduct a systematic study
on tasks where frozen VFMs underperform and reveal several key insights when
fine-tuning them. First, parameter-efficient fine-tuning (PEFT) using only labeled
data often surpasses traditional SSL methods—even without access to unlabeled
data. Second, pseudo-labels generated by PEFT models offer valuable supervisory
signals for unlabeled data, and different PEFT techniques yield complementary
pseudo-labels. These findings motivate a simple yet effective SSL baseline for the
VFM era: ensemble pseudo-labeling across diverse PEFT methods and VFM backbones. Extensive experiments validate the effectiveness of this approach, offering
actionable insights into SSL with VFMs and paving the way for more scalable and
robust semi-supervised learning in the foundation model era.

Paper 2 Findings:
Semi-supervised learning (SSL) has witnessed remarkable progress, resulting in the emergence of
numerous method variations. However, practitioners often encounter challenges when attempting to deploy these methods due to their subpar
performance. In this paper, we present a novel
SSL approach named FINESSL that significantly
addresses this limitation by adapting pre-trained
foundation models. We identify the aggregated
biases and cognitive deviation problems inherent
in foundation models, and propose a simple yet
effective solution by imposing balanced margin
softmax and decoupled label smoothing. Through
extensive experiments, we demonstrate that FINESSL sets a new state of the art for SSL on
multiple benchmark datasets, reduces the training cost by over six times, and can seamlessly
integrate various fine-tuning and modern SSL algorithms. 