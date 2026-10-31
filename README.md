# NLP-Based Mood Tracking and Emotion Classification

A transformer-based Natural Language Processing (NLP) system that analyzes daily journal entries, classifies emotional states using a pretrained DistilRoBERTa model, and records mood predictions for tracking emotional patterns over time.

The system combines **pretrained transformer inference, batch processing, confidence scoring, classification evaluation, and persistent mood logging** to transform unstructured journal text into structured emotional insights.

## Overview

Daily journal entries contain valuable linguistic signals about a person's emotional state. This project applies a pretrained transformer-based text classification model to identify emotions expressed in journal text and organize the predictions into a structured dataset.

The implementation uses `j-hartmann/emotion-english-distilroberta-base` from Hugging Face Transformers to perform emotion classification. It also evaluates predictions against labeled text data, visualizes classification errors, and maintains a CSV-based mood history.

### Key Features

- **Transformer-Based Emotion Classification:** Uses a pretrained DistilRoBERTa model to classify emotions expressed in natural-language text.
- **Batch Inference:** Processes text samples in configurable batches to support efficient dataset-level predictions.
- **Confidence Scoring:** Records the model's prediction score alongside each predicted emotion.
- **Model Evaluation:** Generates a classification report containing precision, recall, F1-score, and support.
- **Confusion Matrix Visualization:** Compares actual labels against predicted labels to examine classification performance.
- **Persistent Mood Logging:** Stores journal entries, predicted emotions, confidence scores, and entry dates in a CSV file.
- **Prediction Caching:** Saves dataset predictions to avoid repeating inference when a cached prediction file is available.
- **Mood Distribution Analysis:** Visualizes the frequency of predicted emotions across logged journal entries.

## Technical Architecture

The project follows a straightforward NLP inference and data-analysis pipeline.

```text
                  Labeled Text Dataset
                           |
                           v
                 Dataset Preprocessing
                           |
                           v
              Pretrained DistilRoBERTa
                           |
                           v
                    Batch Inference
                           |
                           v
                Emotion + Confidence
                           |
                +----------+----------+
                |                     |
                v                     v
        Model Evaluation       Prediction Cache
                |                     |
                v                     v
       Classification Report    predicted_emotions.csv
                |
                v
         Confusion Matrix


              Daily Journal Input
                       |
                       v
          Transformer-Based Inference
                       |
                       v
             Predicted Emotion
                       |
                       v
             Confidence Score
                       |
                       v
              Date-Stamped Log
                       |
                       v
                 mood_log.csv
                       |
                       v
            Mood Distribution Plot
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| NLP Model | DistilRoBERTa |
| Model Repository | Hugging Face Hub |
| Transformer Inference | Hugging Face Transformers |
| Deep Learning Framework | PyTorch |
| Data Processing | Pandas |
| Model Evaluation | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Progress Monitoring | tqdm |
| Data Persistence | CSV |

## Model Details

**Model:** [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

The project uses a pretrained DistilRoBERTa-based sequence classification model to predict the emotion most strongly associated with an input text.

The Hugging Face `pipeline` API handles tokenization, model inference, and conversion of model outputs into predicted labels and scores.

For each input journal entry, the system returns:

- **Predicted emotion:** The emotion label selected by the classifier.
- **Confidence score:** The model's score for the predicted label, rounded to three decimal places.
- **Date:** The date on which the journal entry is logged.

The implementation uses GPU acceleration when CUDA is available and falls back to CPU execution otherwise.

> **Note:** The prediction score represents the model's confidence-related output, not a clinically validated measurement of emotional state. Predictions may not fully capture context, sarcasm, mixed emotions, or subtle linguistic expressions.

## Dataset

The implementation loads a labeled text dataset from `train.txt`.

Each valid line must contain a text sample and its corresponding emotion label, separated by a semicolon.

**Expected format:**

```text
I feel happy today;joy
I am worried about the upcoming exam;fear
I feel frustrated with this situation;anger
I feel sad about what happened;sadness
```

The dataset loader splits each valid line at the first semicolon and stores the resulting values in a Pandas DataFrame.

The resulting dataset contains two columns:

| Column | Description |
|---|---|
| `text` | Input text sample |
| `label` | Reference emotion label |

**Important:** The reference labels must use the same naming convention as the model's predicted labels for meaningful classification evaluation. The examples above illustrate the file format; they are not a claim about the actual contents of the dataset.

## Installation

### Prerequisites

- Python 3.9 or later
- pip
- Internet access for downloading the pretrained model and tokenizer on the first run
- Optional: CUDA-compatible GPU for accelerated inference

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Replace the placeholders with your GitHub repository URL and directory name.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn transformers torch tqdm
```

Alternatively, create a `requirements.txt` file containing:

```text
pandas
scikit-learn
matplotlib
seaborn
transformers
torch
tqdm
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the labeled `train.txt` file in the project directory, alongside the Python script or notebook.

Ensure that the dataset follows the expected semicolon-separated format.

## Usage

### 1. Run the Classification Pipeline

Execute the notebook in Google Colab or run the Python script containing the implementation.

```bash
python mood_tracking.py
```

Replace `mood_tracking.py` with the actual Python filename if it differs.

The program loads the dataset, initializes the pretrained model, and generates emotion predictions in batches.

### 2. Evaluate Model Predictions

The implementation prints a classification report comparing the reference labels against the predicted labels.

The report includes:

- **Precision:** The proportion of predictions for a class that are correct.
- **Recall:** The proportion of reference samples for a class that are correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Support:** The number of reference samples belonging to each class.

A confusion matrix is also generated to visualize the distribution of correct and incorrect predictions across emotion classes.

### 3. Analyze a Daily Journal Entry

After dataset processing, the program prompts for a journal entry:

```text
How was your day?
```

Enter a natural-language description of your day. The classifier predicts the associated emotion and calculates the corresponding prediction score.

For example, an illustrative interaction might look like:

```text
Input:
I felt excited about completing my project today.

Predicted emotion:
joy

Confidence score:
0.XXX
```

The predicted label and score above are illustrative. Actual results depend on the model's output.

### 4. Review Mood History

The system appends each journal entry to `mood_log.csv`.

After logging, it generates a bar plot showing the frequency of predicted emotions in the available mood history.

This visualization provides a basic overview of the distribution of recorded emotional states.

## Implementation Details

### 1. Dataset Loading and Parsing

The `load_emotion_file()` function reads labeled text samples from a local file and converts them into a structured DataFrame.

This separates raw dataset ingestion from downstream inference and evaluation.

### 2. Transformer Initialization

The model and tokenizer are loaded using:

```python
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
```

The Hugging Face classification pipeline provides a unified interface for processing input text and obtaining emotion predictions.

### 3. Batch Inference

The `batch_predict()` function processes the input dataset in configurable batches.

```python
def batch_predict(texts, batch_size=32):
    preds, confs = [], []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        results = emotion_classifier(batch)

        for result in results:
            preds.append(result["label"])
            confs.append(round(result["score"], 3))

    return preds, confs
```

Batch processing reduces the overhead associated with making individual pipeline calls for every text sample and provides a progress indicator during inference.

### 4. Prediction Caching

The implementation stores dataset predictions in `predicted_emotions.csv`.

If the cache file exists, the program loads it instead of performing inference again. Otherwise, it generates predictions and saves the results.

This reduces repeated inference time when the cached predictions remain valid for the current dataset and model configuration.

**Implementation consideration:** The current cache logic does not verify whether the dataset or model has changed since the cache was generated. Regenerate the cache when the underlying data or model configuration changes.

### 5. Classification Evaluation

The project uses Scikit-learn's `classification_report()` and `confusion_matrix()` functions to compare the reference emotion labels with model predictions.

The resulting metrics help identify classes that the classifier predicts accurately and classes that may be confused with one another.

The evaluation is meaningful only when the reference labels are compatible with the model's output labels.

### 6. Mood Logging

The `log_mood_entry()` function accepts journal text and records the following fields:

| Field | Description |
|---|---|
| `date` | Date of the journal entry |
| `journal` | Original journal text |
| `predicted_mood` | Emotion predicted by the classifier |
| `confidence` | Prediction score rounded to three decimal places |

The data is persisted in `mood_log.csv`, allowing entries to be retained across program executions.

### 7. Mood Distribution Visualization

The `plot_mood_trends()` function reads the saved mood history and plots the frequency of each predicted emotion.

Despite the function's name, the current visualization shows **emotion frequency rather than chronological mood trends**. It does not yet plot daily changes, moving averages, or changes in emotion over time.

## Project Structure

The repository can be organized as follows:

```text
mood-tracking-nlp/
├── mood_tracking.ipynb
├── train.txt
├── requirements.txt
├── predicted_emotions.csv
├── mood_log.csv
└── README.md
```

- `mood_tracking.ipynb` — Notebook containing the model initialization, inference, evaluation, logging, and visualization code.
- `train.txt` — Labeled text dataset used for evaluation.
- `requirements.txt` — Python dependencies.
- `predicted_emotions.csv` — Cached dataset predictions generated during execution.
- `mood_log.csv` — Persisted daily journal entries and predicted emotions.
- `README.md` — Project documentation.

Generated CSV files do not need to be committed to GitHub unless they are intentionally included as sample outputs.

**Privacy recommendation:** Do not commit real journal entries or personal mood logs. Add `mood_log.csv` and other private data files to `.gitignore`.

## Limitations

- **English-language model:** The selected pretrained model is intended for English text.
- **Context sensitivity:** A single text classification result may not capture multiple emotions or the full context of a journal entry.
- **Label compatibility:** Evaluation depends on the reference dataset using compatible emotion labels.
- **Confidence interpretation:** A high prediction score does not guarantee that the predicted emotion is correct.
- **Basic trend analysis:** The current visualization summarizes emotion frequency rather than temporal changes.
- **CSV-based storage:** The implementation does not provide a database, authentication, encryption, or a web interface.
- **No clinical validation:** This is an NLP experimentation project, not a diagnostic or mental health assessment tool.

## Potential Improvements

Future development could extend the system through:

- **Temporal Mood Analysis:** Track daily emotion distributions and visualize changes across weeks or months.
- **Interactive Dashboard:** Build a Streamlit interface for journal submission, prediction display, and historical analysis.
- **Model Benchmarking:** Compare the pretrained DistilRoBERTa classifier with alternative transformer-based emotion classifiers.
- **Evaluation Improvements:** Validate label mappings, calculate macro and weighted metrics, and evaluate on a clearly defined held-out dataset.
- **Data Management:** Introduce a database with appropriate access controls for persistent mood records.
- **Inference Optimization:** Add configurable batch sizes, cache validation, and model inference optimizations.
- **Emotion Distribution Analysis:** Preserve the complete class probability distribution when supported by the pipeline, rather than recording only the selected class score.
- **Privacy Protection:** Implement secure storage, data deletion controls, and safeguards for sensitive journal text.

## Skills Demonstrated

- Natural Language Processing (NLP)
- Transformer-Based Text Classification
- Hugging Face Transformers
- PyTorch Model Inference
- Batch Processing and Prediction Caching
- Classification Metrics and Error Analysis
- Data Processing with Pandas
- Data Visualization with Matplotlib and Seaborn
- File-Based Data Persistence

## Disclaimer

This project is intended for educational and experimental purposes. Its predictions represent automated text classification outputs and should not be treated as definitive interpretations of a person's emotional state or as professional mental health advice.

## License

Add a license file if you intend to distribute the project as open source. Choose a license that matches your intended usage and distribution requirements.
