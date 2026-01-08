# Spotify Review Sentiment Analysis using LSTM

## Overview
This project applies **Deep Learning** techniques to classify user reviews of the Spotify app into **Positive** or **Negative** sentiments. Unlike traditional machine learning models, this project utilizes **LSTM (Long Short-Term Memory)** networks, which are highly effective at understanding the context and sequence of words in text data.

## Dataset
* **Source:** Spotify App Reviews
* **File:** `data/Spotify_DATASET.csv`
* **Size:** ~10,000+ reviews
* **Features:**
    * `Review`: Text content of the user feedback.
    * `label`: Sentiment label (`POSITIVE` or `NEGATIVE`).

## Model Architecture
The solution is built using **TensorFlow** and includes the following pipeline:
1.  **Text Preprocessing:** Tokenization and padding sequences.
2.  **Embedding Layer:** Converts words into dense vectors of fixed size.
3.  **LSTM Layer:** Captures long-term dependencies in the text sequence.
4.  **Dense Layers:** Fully connected layers for final classification.
5.  **Output Layer:** Sigmoid activation function for binary classification (0 or 1).

## Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/citakamaliaa/spotify-review-lstm.git](https://github.com/citakamaliaa/spotify-review-lstm.git)
    cd spotify-review-lstm
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook**
    Open `notebooks/Spotify-Review-RNN-LSTM.ipynb` in Jupyter Notebook or Google Colab to see the training process and evaluation results.

## Results
The LSTM model was evaluated on a test set, comparing the **Actual Sentiment** vs. **Predicted Sentiment**.
*(See the notebook for the confusion matrix and accuracy plots)*.

## License
This project is licensed under the MIT License.
