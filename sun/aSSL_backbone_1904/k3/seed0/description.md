# Classification Agent Task Specification

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset. Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a very small training set, a validation set and a set of unlabelled datapoints. 

Your output will be evaluated using a standardized evaluation script which will return a set of unsupervised proxy metrics (ami, ari, v_measure, fmi, silhouette) as the feedback, but note that the priority is test set performance. At the end of the evolution, the final program will be evaluated on the validation set. 

### Dataset specification 
A large-scale scene recognition dataset containing images from hundreds of diverse environment categories, such as indoor spaces and outdoor landscapes. It emphasizes fine-grained distinctions between scenes (e.g., different types of rooms or natural settings) and contains total 397 classes.
Your training dataset contains 3 images per class for a total of just 1191 images. Your unlabeled dataset has 18659 datapoints. 
 
### Model Interface

You MUST use the provided model builder.

#### Step-by-step usage

1. Choose backbone:
    net_name ∈ {
        "timm/vit_base_patch16_clip_224.openai",   # CLIP
    }

2. Create PEFT config:
    peft_config = get_peft_config_fn({...})

PEFT options:
Using Lora: { "method_name": "lora_1", "lora_bottleneck": 4 or 16 or .. }
Using Adaptformer: {
                    "method_name": "adaptformer", 
                    "ft_mlp_module": "adapter",
                    "ft_mlp_mode": "parallel",
                    "ft_mlp_ln": "before",
                    "adapter_init": "lora_kaiming",
                    "adapter_bottleneck": 4 or 16,
                    "adapter_scaler": 0.1
                    }
Frozen Backbone: {"freeze_backbone": True} (default is False)

3. Set vit_config: 
    vit_config = {"drop_path_rate": 0 or 0.2}
4. Build model class:
    net_builder = net_builder_fn(net_name, peft_config, vit_config)

IMPORTANT: net_builder is a CLASS, not an instance.

5. Instantiate:
    self.model = net_builder(
        num_classes=num_classes,
        pretrained=True,
        pretrained_path=""
    ).to(device)

### Model Behavior (IMPORTANT)

The model is a ViT wrapper with the following behavior:

- `out = model(x)` returns:
    - `out["feat"]`: pooled feature vector of shape (B, 768)
    - `out["logits"]`: classification logits of shape (B, num_classes)

Notes:
- `feat` is already pooled (no need for CLS token extraction or pooling)
- `logits` are raw (no softmax applied)
- **IMPORTANT:** After instantiation, use trainable = [p for p in self.model.parameters() if p.requires_grad] to identify optimizable parameters. The model builder sets gradient flags to indicate what should be trained.

## Program Interface (Required)

You must implement the following class in your program:

```python

class ClassificationAgent:
    def __init__(self, net_builder_fn, get_peft_config_fn, num_classes):
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
    agent = program.ClassificationAgent(net_builder_fn=get_net_builder, get_peft_config=get_peft_config, num_classes=397)
    print("Created agent...")
    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
    ...
    image = agent.transform(raw_image_tensor)
    logits, feat, predictions = agent.predict(image)
    ....
```