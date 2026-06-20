# Face Mask Detection 😷

A deep learning pipeline that classifies face mask status — **worn correctly**, **worn incorrectly**, or **not worn at all** — from images or live webcam feed. Built with TensorFlow/Keras, MTCNN for face detection, and deployed as an interactive Streamlit web app.

## Demo

The app supports two input modes:
- **Upload a photo** — get instant classification with confidence scores
- **Live webcam** — real-time detection with bounding boxes drawn directly on the video feed

## How it works

1. **Face detection** — MTCNN locates faces in the input image/frame
2. **Classification** — each detected face is cropped, resized to 128×128, and passed through a fine-tuned EfficientNetB0 model
3. **Output** — a bounding box and label (Mask / Incorrect Mask / No Mask) with a confidence score is drawn for each face

## Model

- **Architecture:** EfficientNetB0 (pretrained on ImageNet), with a custom classification head, fine-tuned via transfer learning
- **Classes:** `with_mask`, `without_mask`, `incorrect_mask`
- **Training data:** combined from two Kaggle datasets to improve coverage of different mask-wearing patterns (e.g. nose exposed, mask below chin)

## Tech stack

- **TensorFlow / Keras** — model training and inference
- **MTCNN** — face detection
- **OpenCV, Pillow** — image preprocessing
- **Streamlit** — web app interface
- **streamlit-webrtc** — real-time webcam streaming in-browser

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project structure
├── app.py                       # Streamlit app

├── best_facemask_effnet.keras   # Trained model

├── requirements.txt             # Python dependencies

└── *.ipynb                      # Training notebooks (data prep, training, evaluation)

## Known limitations

- Classification confidence is lower in poor lighting conditions, since training data is biased toward bright, evenly-lit photos
- The "incorrect mask" class is the hardest to detect reliably, as it covers several visually distinct sub-patterns (nose exposed, mask below chin, etc.)

## License

This project is for educational purposes.
