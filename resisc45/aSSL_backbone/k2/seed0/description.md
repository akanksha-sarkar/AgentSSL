# Classification Agent Task Specification

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset. Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a very small training set, a validation set and a set of unlabelled datapoints. 

Your output will be evaluated using a standardized evaluation script which will return a set of unsupervised proxy metrics (ami, ari, v_measure, fmi, silhouette) as the feedback, but note that the priority is test set performance. At the end of the evolution, the final program will be evaluated on the validation set. 

### Dataset specification 
A benchmark dataset of aerial RGB images depicting diverse land-use and scene categories. It contains 45 classes (e.g., urban areas, transportation infrastructure, natural landscapes). Each image is 256x256 pixels and the dataset is balanced across classes. Your training dataset contains 2 images per class for a total of just 90 images. Your unlabeled dataset has 21915 datapoints. 

### Model Requirement and Tuning Library

You are required to ONLY use models through the interface below. You will also be provided a useful parameter-efficient finetuning (PEFT) library which can be set as shown below. 

You will receive two functions **net_builder_fn** and **get_peft_config_fn** as an input to your class.

    net_builder_fn(net_name, peft_config=None, vit_config=None):
    """
    built network according to network name
    return **class** of backbone network (not instance).

    Args
        net_name: Must be one of the two options.
        - CLIP-ViT-16-B: set net_name = "timm/vit_base_patch16_clip_224.openai"   
        - DINOv2-B: set net_name = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
    """ 

    get_peft_config_fn(peft_config)
    """
    Updates the default peft config with peft_config dictionary values. 
    These are the default values:
    _DEFAULT_PEFT_CONFIG = {
        "ft_attn_module": None,
        "ft_attn_mode": "parallel",
        "ft_attn_ln": "before",
        "ft_mlp_module": None,
        "ft_mlp_mode": "parallel",
        "ft_mlp_ln": "before",
        "adapter_bottleneck": 64,
        "adapter_init": "lora_kaiming",
        "adapter_scaler": 0.1,
        "convpass_bottleneck": 8,
        "convpass_xavier_init": False,
        "convpass_init": "lora_xavier",
        "convpass_scaler": 10,
        "vpt_mode": None,
        "vpt_num": 10,
        "vpt_layer": None,
        "vpt_dropout": 0.1,
        "vqt_num": 0,
        "ssf": False,
        "lora_bottleneck": 0,
        "fact_dim": 8,
        "fact_type": None,
        "fact_scaler": 1.0,
        "repadapter_bottleneck": 8,
        "repadapter_init": "lora_xavier",
        "repadapter_scaler": 1,
        "repadapter_group": 2,
        "bitfit": False,
        "attention_type": "full",
        "ln": False,
        "difffit": False,
        "freeze_backbone": False,
    }
    """

The required way to access these models is as follows. 
1. Choose from the provided backbones. 
- CLIP-ViT-16-B: set *net_name* = "timm/vit_base_patch16_clip_224.openai"
- DINOv2-B: set *net_name* = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
2. Set any PEFT configs to *peft_config* using **get_peft_config_fn**.
3. Set any ViT configs to *vit_config*.
4. Set *net_builder* with **net_builder_fn** (Remember it is a class not an instance).
5. Set *self.model* with **net_builder** as shown below:
```python
    self.model = net_builder(
                num_classes=num_classes,
                pretrained=True,
                pretrained_path="",
            ).to(device)
```

---

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
            predictions: list[dict]
        """
        pass

```

## Usage

The program you write will be used in the following way:
```python
    program = load_program(program_path)

    print("Loaded program...")
    agent = program.ClassificationAgent(net_builder_fn=get_net_builder, get_peft_config=get_peft_config)
    print("Created agent...")
    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
    ...
    image = agent.transform(raw_image_tensor)
    logits, feat, predictions = agent.predict(image)
    ....
```