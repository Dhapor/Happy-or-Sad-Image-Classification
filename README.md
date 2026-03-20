# 😊 Happy or Sad Image Classifier

A real-time facial expression classifier built with TensorFlow and Streamlit. Upload a photo or use your webcam to classify faces as **Happy** or **Sad**, with a confidence score.

---

## Features

- Upload an image for instant classification
- Live webcam classification with on-screen prediction overlay
- Confidence score displayed alongside every prediction
- Clean, responsive Streamlit UI

---

## Project Structure

```
Happy-or-Sad-Image-Classification/
├── classif.py              # Main Streamlit app
├── requirements.txt        # Python dependencies
├── models/
│   └── imageclassifier2.h5 # Trained CNN model
├── data/                   # Training data (not tracked in git)
├── logs/                   # Training logs
├── Getting Started.ipynb   # Data prep and model training notebook
└── second_try.ipynb        # Improved training experiments
```

---

## Setup and Installation

**1. Clone the repo**
```bash
git clone https://github.com/Dhapor/Happy-or-Sad-Image-Classification.git
cd Happy-or-Sad-Image-Classification
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Make sure the model file is in place**

The trained model should be at `models/imageclassifier2.h5`. If you need to retrain it, run through the `Getting Started.ipynb` notebook.

**5. Run the app**
```bash
streamlit run classif.py
```

Then open `http://localhost:8501` in your browser.

---

## How the Model Works

- Input images are resized to **256x256 pixels** and normalised to [0, 1].
- A CNN outputs a value between 0 and 1.
  - Values closer to **0** = Happy
  - Values closer to **1** = Sad
- Confidence is derived from how far the prediction score is from 0.5.

---

## Requirements

- Python 3.8 - 3.11
- TensorFlow 2.15
- Streamlit 1.28
- OpenCV (headless)
- streamlit-webrtc

---

## Author

Built by **Datapsalm**
