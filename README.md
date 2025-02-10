# English to French Translation
# Seq2Seq Machine Translation using LSTM

This project implements a **Sequence-to-Sequence (Seq2Seq) model** for machine translation using **TensorFlow and Keras**. The model is trained on English-French sentence pairs and uses an **LSTM-based encoder-decoder architecture**.

---

## 📌 Features
- Implements **Seq2Seq model** using LSTMs for text translation.
- Uses **one-hot encoding** to preprocess text data.
- Supports **character-level tokenization**.
- **Trains a model** using the RMSprop optimizer and categorical cross-entropy loss.
- Implements **inference mode (sampling)** to generate translations.
- Saves and loads trained models for future inference.

---

## 📂 Dataset
The dataset used for training consists of English-French sentence pairs from the `fra.txt` file. The sentences are preprocessed and vectorized before training.

---

## 🔧 Requirements
To run this project, install the required dependencies using:
```bash
pip install tensorflow numpy
```

---

## 🚀 How to Run
### 1️⃣ Train the Model
Run the following command to train the model:
```bash
python train.py
```
This will:
- Load and preprocess the dataset
- Train the encoder-decoder model
- Save the trained model as `s2s_model.keras`

### 2️⃣ Perform Inference
Once the model is trained, you can generate translations by running:
```bash
python translate.py
```
This will:
- Load the trained model
- Take user input for an English sentence
- Generate a translated French sentence

---

## 🏗 Project Structure
```
├── fra.txt       # Dataset file (English-French sentence pairs)
├── train.py              # Training script
├── translate.py          # Inference script
├── s2s_model.keras       # Saved trained model
├── README.md             # Project documentation
```

---

## 🔥 Model Architecture
- **Encoder:** LSTM processes input sequences and stores its state.
- **Decoder:** LSTM generates output sequences based on encoder state.
- **Inference Mode:** Uses trained model to generate translations character-by-character.

---

## 📈 Training Details
- **Batch size:** `64`
- **Epochs:** `100`
- **Latent dimensions:** `256`
- **Optimizer:** `RMSprop`
- **Loss function:** `Categorical Crossentropy`

---

## 📝 Example Output
```
Input sentence: How are you?
Decoded sentence: Comment allez-vous?
```

---

## 🎯 Future Improvements
- Use **pre-trained word embeddings** (e.g., Word2Vec, GloVe) for better translation.
- Implement **attention mechanism** to improve translation accuracy.
- Train on a **larger dataset** for better generalization.

---

## 💡 References
- [TensorFlow Seq2Seq Documentation](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
- [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/)

---

## 🤝 Contributing
Feel free to contribute by improving the model, adding new features, or optimizing performance. Pull requests are welcome! 😊

---
