# Speech-to-Joy: Predicting Enjoyment from Speech

This repository contains the code and data for the research project "Speech-to-Joy," which focuses on predicting enjoyment levels in dyadic conversations using machine learning. The project leverages both audio and text modalities to train and evaluate various models, from baseline RNNs to state-of-the-art Large Language Models (LLMs).

## Project Overview

The primary goal of this project is to analyze multimodal signals from conversations to automatically predict self-reported enjoyment scores. We explore a range of techniques, including:

- **Audio Analysis**: Using raw audio features and advanced speech representations from models like HuBERT and wav2vec2.
- **Text Analysis**: Processing conversation transcripts to capture semantic and emotional content.
- **Multimodal Fusion**: Combining audio and text information to improve prediction accuracy.
- **LLM-based Prediction**: Leveraging the capabilities of modern LLMs (e.g., Gemini, Gemma, Claude) for zero-shot or few-shot enjoyment prediction.

The evaluation metric used across all models is the **Concordance Correlation Coefficient (CCC)**, which measures the agreement between predicted and ground-truth enjoyment scores.

## Repository Structure

The repository is organized as follows:

- **Jupyter Notebooks**: Core scripts for data processing, modeling, and analysis.
  - `data.ipynb`: Handles initial data loading, preprocessing, and organization.
  - `extract-annotations.ipynb`: Extracts annotations and timestamps from transcript files.
  - `ccc_audio.ipynb` / `ccc_audio_rnn.ipynb`: Models for audio-based enjoyment prediction using CCC.
  - `ccc_text.ipynb` / `ccc_text_rnn.ipynb`: Models for text-based enjoyment prediction using CCC.
  - `hubert.ipynb` / `wav2vec2.ipynb`: Feature extraction using HuBERT and wav2vec2 models.
  - `llms.ipynb`: Experiments with various Large Language Models for prediction.
  - `mean-pooling.ipynb`: Implements mean pooling for feature aggregation.
  - `plots.ipynb`: Generates visualizations and plots for results analysis.
  - `random.ipynb`: Implements a random baseline for comparison.
- **`data/`**: Contains raw and processed data, including audio files and transcripts.
- **`predictions/`**: Stores the prediction outputs from various models.
- **`backup-*/`**: Contains backups of predictions from different LLM experiments (e.g., Gemini, Gemma, Claude).
- **`y-enjoyment.csv`**: The ground-truth file containing self-reported enjoyment scores for each conversation.
- **`README.md`**: This file.

## Methodology

### 1. Data Preparation

The `data.ipynb` notebook is the starting point for preparing the dataset. It involves loading the audio files, transcripts, and the `y-enjoyment.csv` ground-truth labels.

### 2. Feature Extraction

- **Audio**: The `hubert.ipynb` and `wav2vec2.ipynb` notebooks are used to extract deep audio representations. These models convert raw audio into rich embeddings that capture complex speech characteristics.
- **Text**: Transcripts are processed to create text-based features for the text models.

### 3. Modeling

Several modeling approaches are implemented and evaluated:

- **Baseline Models**:
  - **Random Baseline** (`random.ipynb`): Predicts random scores to establish a lower bound for performance.
  - **RNN Models** (`ccc_audio_rnn.ipynb`, `ccc_text_rnn.ipynb`): Basic recurrent neural networks for sequence modeling on both audio and text.
- **Advanced Models**:
  - **HuBERT/wav2vec2-based Models**: These models use the extracted audio embeddings to predict enjoyment.
  - **Large Language Models (LLMs)** (`llms.ipynb`): Modern LLMs are prompted with conversation transcripts (and sometimes audio information) to generate enjoyment predictions. Experiments include models like Google's Gemini and Gemma, and Anthropic's Claude.

### 4. Evaluation

All models are evaluated using the Concordance Correlation Coefficient (CCC), which is well-suited for assessing agreement in continuous-valued predictions. The `plots.ipynb` notebook helps visualize and compare the performance of different models.

## How to Run

1. **Setup**: Ensure all dependencies listed in the environment are installed.
2. **Data Processing**: Run `data.ipynb` to prepare the dataset.
3. **Feature Extraction**: Run `hubert.ipynb` or `wav2vec2.ipynb` to generate audio embeddings.
4. **Training and Prediction**:
    - To run the baseline or RNN models, execute the corresponding `ccc_*.ipynb` notebooks.
    - To run LLM-based predictions, use the `llms.ipynb` notebook.
5. **Analysis**: Use `plots.ipynb` to generate performance plots and compare results.

## Dependencies

The project relies on several key Python libraries:

- **Core**: `numpy`, `pandas`, `scikit-learn`
- **Deep Learning**: `torch`, `transformers`
- **Audio Processing**: `librosa`, `pydub`, `soundfile`
- **Visualization**: `matplotlib`, `seaborn`
- **Jupyter**: For running the notebooks.
