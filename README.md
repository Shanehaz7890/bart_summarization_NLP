# BART Text Summarization Project

This project implements a complete BART (Bidirectional and Auto-Regressive Transformers) text summarization pipeline following the methodology outlined in the midterm presentation slides. The system fine-tunes BART on the CNN/DailyMail dataset and evaluates performance using ROUGE scores.

## Project Overview

BART is a transformer-based model developed by Facebook (Meta AI) in 2019. It combines a BERT-style encoder with a GPT-style decoder, making it strong at both understanding and generating text. This project uses BART for abstractive text summarization.

## Methodology

The project follows a 5-step pipeline:

1. **Collect and Select Dataset** - Uses the CNN/DailyMail dataset (300k news articles with human-written summaries)
2. **Preprocess Text** - Tokenization and cleaning of articles and summaries
3. **Fine-tune BART** - Configure and fine-tune BART with hyperparameters:
   - `max_length`: Maximum summary length (142 tokens)
   - `min_length`: Minimum summary length (56 tokens)
   - `temperature`: Sampling temperature (1.0)
   - `num_beams`: Beam search width (4)
   - `length_penalty`: Length penalty (2.0)
4. **Generate Summaries** - Use the fine-tuned model to generate summaries
5. **Evaluate** - Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) to assess performance

## Dataset

The CNN/DailyMail dataset is the standard dataset used for summarization models. It contains:
- 300k news articles with human-written summaries
- Split into training, validation, and test sets
- Version 2 and 3 (version 1 was for abstractive question answering)

The dataset files should be located in the `cnn_dailymail/` directory:
- `train.csv`
- `validation.csv`
- `test.csv`

Each CSV file contains columns: `id`, `article`, `highlights`

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training, but CPU will work for inference)

### Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure the CNN/DailyMail dataset is in the `cnn_dailymail/` directory

## Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook bart_summarization.ipynb
```

2. Run the cells sequentially:
   - **Section 1**: Setup and imports
   - **Section 2**: Dataset loading and exploration
   - **Section 3**: Text preprocessing
   - **Section 4**: BART fine-tuning (this may take several hours)
   - **Section 5**: Summary generation
   - **Section 6**: ROUGE evaluation

### Fine-tuning

The fine-tuning process will:
- Train for 3 epochs
- Use a batch size of 4 with gradient accumulation (effective batch size of 16)
- Save checkpoints after each epoch
- Save the final model to `./bart-cnn-dailymail/final_model`

**Note**: Fine-tuning can take several hours depending on your hardware. On a modern GPU, expect 3-6 hours for the full training set.

### Generating Summaries

After fine-tuning, you can generate summaries for new articles using the `generate_summary()` function. The notebook includes examples of generating summaries for the test set.

### Evaluation

The project uses ROUGE scores for evaluation:
- **ROUGE-1**: Measures overlap of unigrams (single words)
- **ROUGE-2**: Measures overlap of bigrams (word pairs)
- **ROUGE-L**: Measures longest common subsequence

Each metric provides precision, recall, and F-measure scores.

## Model Information

- **Base Model**: `facebook/bart-large-cnn` (pre-trained on CNN/DailyMail)
- **Model Size**: ~400M parameters
- **Input Length**: Up to 1024 tokens
- **Output Length**: Configurable (default: 56-142 tokens)

## Expected Outputs

After running the complete pipeline, you should see:
- Dataset statistics (article and summary length distributions)
- Training progress and loss metrics
- Sample generated summaries compared to reference summaries
- ROUGE evaluation scores (typically ROUGE-1 F1: 0.40-0.50, ROUGE-2 F1: 0.20-0.25, ROUGE-L F1: 0.35-0.45)

## File Structure

```
Final Pres/
├── cnn_dailymail/              # Dataset directory
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── bart_summarization.ipynb    # Main Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── bart-cnn-dailymail/         # Generated during training
    └── final_model/            # Fine-tuned model (after training)
```

## References

- **BART Paper**: Lewis, M., et al. (2019). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
- **CNN/DailyMail Dataset**: Hermann, K. M., et al. (2015). "Teaching Machines to Read and Comprehend"
- **ROUGE Metric**: Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries"

## Notes

- The model uses mixed precision training (FP16) if a GPU is available
- Training checkpoints are saved to allow resuming if interrupted
- The notebook can be run in sections - you can load a previously fine-tuned model instead of retraining
- For faster experimentation, you can reduce the dataset size or number of training epochs

## License

This project is for educational purposes as part of a Natural Language Processing course.

