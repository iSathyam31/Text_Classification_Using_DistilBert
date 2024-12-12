# DistilBERT Fine-Tuning on AG News Dataset

This project demonstrates the fine-tuning of the `DistilBERT` model on the AG News dataset for text classification, using the Hugging Face `Transformers` library.

## Overview

The AG News dataset consists of four classes of news articles:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

The goal is to fine-tune a `DistilBERT` model to classify these articles into one of the four classes. The fine-tuning process is performed using the Hugging Face `Trainer` API, which simplifies the process of training, evaluation, and saving the model.

## Model Architecture

- **Model**: `DistilBERT-base-uncased`
- **Optimizer**: AdamW
- **Loss Function**: Cross-entropy loss
- **Epochs**: 2
- **Learning Rate**: 2e-5
- **Batch Size**: 32

## Dataset

The AG News dataset is a collection of news articles categorized into four classes:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

You can access the dataset via the Hugging Face `datasets` library.

## Training Configuration

The training arguments are set as follows:

```python
training_args = TrainingArguments(
    output_dir="my_goodish_model",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
```

## Result
After training for 2 epochs, the following evaluation metrics were achieved:

| Epoch | Training Loss | Validation Loss | Accuracy  |
|-------|---------------|-----------------|-----------|
| 1     | 0.110000      | 0.174125        | 94.37%    |
| 2     | 0.094400      | 0.173345        | 94.63%    |

## Dependencies
The required dependencies for this project are:

* transformers
* datasets
* torch
* sklearn
* numpy

## How to Use
1. Clone the repository:
```
git clone https://github.com/iSathyam31/Text_Classification_Using_DistilBert.git
cd Text_Classification_Using_DistilBert
```
2. Install the required dependencies
3. Run the .ipynb

## Push to Hugging Face Hub
This model is automatically pushed to the Hugging Face Model Hub. You can find it here: [Model Hub Link](https://huggingface.co/iSathyam03/my_goodish_model).

## License
This project is licensed under the MIT License.