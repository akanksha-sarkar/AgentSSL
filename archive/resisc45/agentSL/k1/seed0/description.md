# Classification Agent Task Specification
### TESTING

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset.  
Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a limited training set, a validation set.

Your output will be evaluated using a standardized evaluation script that computes classification metrics (e.g., top-1 accuracy).

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

    def predict(self, images):
        """
        Predict labels for a batch of images.

        Inputs:
            images: list[PIL.Image.Image]
                A list of images provided by the evaluator.

                The evaluator will call this function on batches of images.
                Your code should treat this as inference only.

        Returns:
            predictions: list[dict]
                A list of predictions, one per input image, in the SAME ORDER
                as the input list.

                Each prediction must be a dictionary of the form:
                    {
                        "category_id": int,   # required
                        "score": float        # optional but recommended
                    }

        Required fields (per prediction):
            - "category_id":
                The predicted class label as the ORIGINAL COCO category_id,
                not an internal class index unless they are identical.

        Optional fields (per prediction):
            - "score":
                A confidence score for the prediction. Higher means more
                confident.

        Example valid return:
            [
                {"category_id": 5, "score": 0.91},
                {"category_id": 2, "score": 0.77},
                {"category_id": 8, "score": 0.66},
            ]

        Notes:
            - The length of the returned list MUST equal len(images).
            - The i-th prediction must correspond to the i-th input image.
            - Do not shuffle or reorder inputs internally unless you restore order.
            - Do not assume access to filenames, paths, or annotations.
            - The evaluator handles dataset iteration and ground-truth comparison.

        Important:
            - Convert each PIL image to a tensor using your test/inference transform.
            - Stack inputs into a batch tensor before passing to the model.
            - If using internal label indices, map them back to COCO category_id.
            - This function must be side-effect free (no training, no state updates).
        """
        return [
            {"category_id": 0, "score": 1.0}
            for _ in images
        ]
```