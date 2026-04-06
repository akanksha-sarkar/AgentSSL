# Classification Agent Task Specification
### TESTING

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset.  
Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a limited training set, a validation set.

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

    def fit(self, train_img_dir, train_ann_file):
        '''
        Inputs:
            train_img_dir: str
                Directory containing training images.
            train_ann_file: str
                COCO-format JSON file with training labels.

        Behavior:
            Train or prepare the classifier using training data only.

        Returns:
            Optional. May return None.
        '''
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