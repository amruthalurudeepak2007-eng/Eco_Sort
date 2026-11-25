# EcoSort – Smart Waste Classification Web App

A Streamlit web application for an ML internal project. It accepts an image and,
after a trained model is installed, predicts one of six waste categories:
cardboard, glass, metal, paper, plastic, or trash.

## What is included
- `app.py`: dark-themed Streamlit interface, image upload, model inference,
  top-three predictions, confidence display, disposal tips, session history.
- `ml/train.py`: MobileNetV2 transfer-learning training pipeline.
- `requirements.txt`: Python dependencies.

## Important
The project ZIP does not contain a trained model. Real predictions require labelled
training images. The app deliberately shows a setup notice rather than inventing
a prediction when the model is absent.

## Setup
Recommended: Python 3.10 or 3.11.

1. Create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare a labelled image dataset with this exact structure:

```
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

A commonly used educational dataset is TrashNet. Confirm that its folder names
match the six labels above; rename folders if needed. Keep training and test data
separate when reporting final results.

4. Train:
   `python ml/train.py --data_dir dataset --epochs 8`

This creates:
- `models/waste_classifier.keras`
- `models/labels.json`

5. Launch the web application:
   `streamlit run app.py`

The browser opens the EcoSort interface. Upload a clear image containing one
waste item and select **Classify waste**.

## ML explanation for your internal
- Type: supervised multi-class image classification.
- Feature extractor: MobileNetV2 pretrained on ImageNet (transfer learning).
- Classifier head: global average pooling, dropout, dense softmax layer.
- Input: RGB image resized to 224 × 224.
- Normalization: pixel values mapped from 0–255 to -1–1 inside the model.
- Output: six class probabilities; the largest is shown as the predicted class.
- Validation: training script reports validation accuracy. For a fair final result,
  evaluate on a separate held-out test set and report precision, recall, F1-score
  and a confusion matrix. Do not claim an accuracy until you have run training/testing.

## Limitations
Lighting, background, angle, object overlap, class balance and dataset size affect
accuracy. The confidence score is not a guarantee. Disposal rules differ by location.
