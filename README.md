# Speech-to-Joy

A machine learning project focused on predicting enjoyment levels from speech using both audio features and transcript analysis.

## Project Overview

Speech-to-Joy is a research project that analyzes both audio and text data from conversations to predict participant enjoyment. The project uses various machine learning techniques including attention-based models to identify patterns in speech that correlate with self-reported enjoyment ratings.

The primary components of the project include:

- Audio processing and feature extraction
- Transcript analysis
- Various machine learning models for prediction
- Evaluation and visualization of results

## Repository Structure

- **11l-corrected-transcripts/**: Contains the transcripts of audio files with timestamps
- **data/**: Contains raw and processed audio data
- **user-self-reports/**: Contains self-reported enjoyment ratings from participants
- **backup-11-speaker-***: Backup files of various model outputs and predictions
- **Jupyter Notebooks**: Various notebooks for data processing, model training, and analysis:
  - `extract-annotations.ipynb`: Processes and extracts annotations from audio files
  - `attention_audio.ipynb`: Implements attention-based models for audio analysis
  - `attention_text.ipynb`: Implements attention-based models for text analysis
  - `hubert.ipynb`: Uses HuBERT (Hidden Unit BERT) models for audio processing
  - `llms.ipynb`: Experiments with Large Language Models
  - `mean-pooling.ipynb`: Implements mean pooling techniques
  - `plots.ipynb`: Generates visualizations of results
  - `data.ipynb`: Data preprocessing and organization
  - `random.ipynb`: Random baseline model implementations

## Data Processing

The project processes audio files through several steps:

1. Extracting annotations from SRT files
2. Trimming audio to focus on relevant segments
3. Voice Activity Detection (VAD) to filter out silence
4. Feature extraction from audio
5. Preparation of text data from transcripts

## Models

Several machine learning approaches are implemented:

- Attention-based models for both audio and text
- HuBERT (Hidden Unit BERT) for audio processing
- Mean pooling techniques
- Random baseline models for comparison

## Model Configurations

The attention-based models use configurations like:

```python
CONFIG = {
    'learning_rate': 1e-3,        
    'num_epochs': 120,            
    'batch_size': 37,             
    'attn_hidden_dim': 1,        
    'fc_hidden_dim': 2048,
    'weight_decay': 1e-2,         
    'dropout_rate': 0.4,          
    'use_dropout': True,          
    'device': torch.device("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
}
```

## Results

The results and predictions from different models are stored in:

- `predictions-audio-attention.csv`
- `predictions-text-attention.csv`
- `fused_predictions_avg.csv`

The `y-enjoyment.csv` file contains the ground truth enjoyment ratings.

## Usage

To use this codebase:

1. **Data Preparation**:
   - Run `data.ipynb` to organize and preprocess the data

2. **Feature Extraction**:
   - Run `extract-annotations.ipynb` to process the audio files and transcripts

3. **Model Training**:
   - Run the specific model notebooks (e.g., `attention_audio.ipynb`, `attention_text.ipynb`) to train and evaluate models

4. **Analysis**:
   - Use `plots.ipynb` to visualize results and compare model performance

## Dependencies

The project relies on several Python libraries:

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- SciPy
- webrtcvad (for Voice Activity Detection)
- Matplotlib (for visualization)
- Audio processing libraries (like pydub)

## Notes

This project appears to be focused on analyzing conversations between participants and two conversational agents named "Alice" and "Clara", with the goal of predicting how enjoyable the conversations were based on participant responses.

The data consists of both audio recordings and their corresponding transcripts, with timestamps available for synchronization. The self-reported enjoyment ratings are used as ground truth for model training and evaluation.
