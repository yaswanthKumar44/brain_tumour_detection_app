
# 🧠 Brain Tumour Detection Web App

An AI-powered Flask application for classifying brain MRI images into four tumour categories using a deep learning model (EfficientNetB3).

![Brain Tumour Detection](https://img.shields.io/badge/Brain%20Tumour%20Detection-EfficientNetB3-blue)

---

## 📸 Demo Preview

> Upload an MRI image and get an instant tumour type prediction with confidence score.
>https://www.linkedin.com/posts/yaswanth-kumar-peddagamalla-91443a288_ai-deeplearning-braintumourdetection-activity-7342469305499795457-qnnB?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEXU66ABDU8KUG62xAonIZM7Pv4L0HRaOVs
---

## 📂 Project Structure

```

brain\_tumour\_app/
├── app.py                     # Flask backend
├── templates/
│   └── index.html             # Bootstrap-enhanced UI
├── static/
│   └── uploads/               # Uploaded images
├── brain\_tumour\_best\_model.h5 # AI model (hosted externally)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

````

---

## 📦 Download Dataset & Model

🚨 Due to GitHub's 100MB file limit, model and dataset files are hosted externally:

👉 [📥 Download from Google Drive](https://drive.google.com/drive/folders/1ruIQL1c94TbVwUn_clVVd36WGpWE0I2C?usp=sharing)

- Place `brain_tumour_best_model.h5` in your project root directory.
- Dataset folder as needed for retraining or analysis.

---

## 🚀 Features

- 📤 Upload brain MRI images via web interface.
- 📈 Predict tumour type with confidence score.
- 🧠 Supports 4 classes:
  - **Glioma**
  - **Meningioma**
  - **Pituitary Tumour**
  - **No Tumour**
- ✨ Clean, animated Bootstrap 5 UI.
- 📊 Displays prediction result and uploaded image.
- 🔥 AI model built using EfficientNetB3 and Transfer Learning.

---

## 📊 AI Model Details

- **Architecture:** EfficientNetB3
- **Trained On:** Brain MRI dataset (available via Drive)
- **Accuracy:** 95.35% (Test Data)
- **Framework:** TensorFlow / Keras

**Features:**
- Data Augmentation: Rotation, flipping, zooming.
- Optimizers: Adam
- Loss Function: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Training Epochs: 30

---

## 📦 Dependencies

```bash
pip install -r requirements.txt
````

**Main Libraries:**

* Flask
* TensorFlow / Keras
* numpy
* Pillow
* Bootstrap (via CDN)

---

## 🖥️ Run Locally

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000/`

---

## 📈 Results

* **Test Accuracy:** 95.35%
* **Classification Report:** Precision, Recall, F1-Score
* **Confusion Matrix**


---

## 🎯 Impact

Early detection of brain tumours is crucial for improved outcomes. This AI-powered solution provides:

* 📉 Faster, automated MRI analysis
* 📈 High accuracy diagnostic assistance
* 💻 Scalable and lightweight deployment

---

## 📢 Author

**P. Yaswanth Kumar**
[GitHub Profile](https://github.com/yaswanthKumar44)

---

## 📜 License

Open-source project — fork, modify and contribute!

---

## 📌 Notes

* Files over 100MB are hosted via Google Drive (link above)
* Ensure `static/uploads/` directory exists before running the app.
* Grad-CAM explanation images can be optionally implemented.

---

## ⭐ Star This Repo

If you like this project — please ⭐ star this repository and share it!

---

