# Classification Agent Task Specification

## Objective

You are an AI agent tasked with implementing a **classification program** for an image dataset.  
Your goal is to train (or otherwise derive) a classifier using the provided training dataset and produce **predicted labels for every image in the validation dataset**. 

You have access to a limited training set, a validation set, and a set of unlabelled dataset. 

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

    def predict(self, image):
        """
        Predict a label for exactly one image.

        Inputs:
            image: PIL.Image.Image
                A single image provided by the evaluator.

                The evaluator will call this function one image at a time.
                Your code should treat this as inference only.

        Returns:
            prediction: dict
                A dictionary of the form:
                    {
                        "category_id": int,   # required
                        "score": float        # optional but recommended
                    }

                Required fields:
                    - "category_id":
                        The predicted class label as the ORIGINAL COCO
                        category_id, not an internal class index unless they
                        are identical.

                Optional fields:
                    - "score":
                        A confidence score for the prediction. Higher means more
                        confident. This field is optional, but recommended.

        Example valid return:
            {
                "category_id": 5,
                "score": 0.91
            }

        Notes:
            - This function must return a prediction for exactly one image.
            - Do not expect access to the full validation set here.
            - Do not expect a filename, path, or annotation file here.
            - The evaluator handles validation-set iteration and ground-truth
              comparison outside your program.

        Important:
            - If your model expects tensors, convert the PIL image inside this
              function using your test/inference transform.
            - If you use an internal label index, convert it back to the
              original COCO category_id before returning.
        """
        return {
            "category_id": 0,
            "score": 1.0,
        }
```