# Named Entity Recognition with Transformers

This project is a Named Entity Recognition (NER) model built with **Hugging Face Transformers** and **PyTorch**.

The model identifies named entities in text, such as:

* Person
* Organization
* Location
* Miscellaneous entities

## Project Overview

Named Entity Recognition is a token classification task where each token in a sentence receives a label.

Example:

```text
Barack Obama was born in Hawaii and worked for Microsoft.
```

Model output:

```text
Barack Obama → Person
Hawaii → Location
Microsoft → Organization
```

## Dataset

This project uses the **CoNLL-2003** dataset.

The dataset contains tokenized sentences with NER labels.

Main fields:

* `tokens`
* `ner_tags`

NER labels:

```text
O
B-PER
I-PER
B-ORG
I-ORG
B-LOC
I-LOC
B-MISC
I-MISC
```

## Task Type

This is a **Token Classification** task.

Unlike text classification, where one label is predicted for the whole sentence, NER predicts one label for each token.

## Model

The project uses a pretrained Transformer model:

```text
distilbert-base-uncased
```

The model was fine-tuned using:

```python
AutoModelForTokenClassification
```

## Tech Stack

* Python
* Hugging Face Datasets
* Hugging Face Transformers
* PyTorch
* Seqeval
* Google Colab

## Project Pipeline

```text
Dataset
→ Tokenization
→ Label Alignment
→ Token Classification Model
→ Training
→ Evaluation
→ Prediction
```

## Key Concepts Used

* Named Entity Recognition
* Token Classification
* BIO Tagging
* Tokenization
* Label Alignment
* Transformer Fine-Tuning
* Data Collator for Token Classification
* Seqeval Metrics
* Precision, Recall, F1 Score

## Evaluation Results

Final validation results:

| Metric          |    Value |
| --------------- | -------: |
| Training Loss   | 0.022743 |
| Validation Loss | 0.051239 |
| Precision       | 0.929942 |
| Recall          | 0.938236 |
| F1 Score        | 0.934071 |
| Accuracy        | 0.986508 |

The most important metric for this project is **F1 Score**, because NER datasets contain many non-entity tokens labeled as `O`.

Final F1 Score:

```text
0.934071
```

## Example Prediction

```python
from transformers import pipeline

ner_pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

ner_pipe("Barack Obama was born in Hawaii and worked for Microsoft.")
```

Expected output:

```text
Barack Obama → PER
Hawaii → LOC
Microsoft → ORG
```

## Model Saving

The trained model and tokenizer were saved using:

```python
trainer.save_model("ner_transformer_model")
tokenizer.save_pretrained("ner_transformer_model")
```

## What I Learned

Through this project, I learned how to:

* Work with token-level NLP tasks
* Use the CoNLL-2003 NER dataset
* Understand BIO tagging
* Align labels after tokenization
* Fine-tune a Transformer model for NER
* Evaluate NER models using precision, recall, and F1 score
* Build an end-to-end Hugging Face NLP pipeline

## Future Improvements

* Test larger Transformer models
* Add a Gradio demo interface
* Deploy the model on Hugging Face Spaces
* Compare DistilBERT with BERT and RoBERTa
* Analyze entity-level errors
* Try NER on custom text datasets

## Author

This project was built as part of my Machine Learning and NLP learning journey.
