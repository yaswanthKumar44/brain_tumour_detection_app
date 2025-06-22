**Brain Tumour Detection Flask App with AI Model** 

---

```markdown
# 🧠 Brain Tumour Detection Web App

An AI-powered Flask web application for detecting brain tumours from MRI images using a deep learning model (EfficientNetB3). The system classifies images into one of four categories: **Glioma**, **Meningioma**, **Pituitary Tumour**, or **No Tumour**.

---

## 📸 Demo Preview

> 👉 Upload your brain MRI image and get instant tumour prediction with confidence score, tumour type information, and visual explanation.

---

## 📂 Project Structure

```

brain\_tumour\_app/
├── app.py                     # Flask application backend
├── templates/
│   └── index.html             # Frontend interface (Bootstrap enhanced)
├── static/
│   └── uploads/               # Folder for uploaded images
├── brain\_tumour\_best\_model.h5 # Pre-trained AI model (download externally)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

````

---

## 📥 Resources

### 📦 Download Dataset & Trained Model

Due to file size limitations on GitHub, the trained model and dataset are hosted on Google Drive.  
👉 [📥 Download Dataset and Model Folder](https://drive.google.com/drive/folders/1ruIQL1c94TbVwUn_clVVd36WGpWE0I2C?usp=sharing)

- Place the **dataset** inside `datasets/` directory.
- Place `brain_tumour_best_model.h5` inside your project root folder.

---

## 🚀 Features

- 📤 Upload MRI images via web interface
- 🧠 Predict tumour type with confidence score
- 📊 Classifies into:
  - Glioma
  - Meningioma
  - Pituitary Tumour
  - No Tumour
- 🎨 Clean animated Bootstrap frontend
- 📈 AI model built with EfficientNetB3 + Transfer Learning
- 🔥 Heatmap visual explanation support (Grad-CAM ready)

---

## 📊 AI Model Details

- **Architecture:** EfficientNetB3  
- **Trained on:** Custom Brain MRI dataset (Google Drive link above)
- **Accuracy:** Achieved **95.35% final test accuracy**
- **Framework:** TensorFlow / Keras

**Model Features:**
- Transfer learning from ImageNet
- Data augmentation (rotation, flipping, zoom)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Normalization to scale input images

---

## 📦 Dependencies

Install project dependencies using:

```bash
pip install -r requirements.txt
````

**Key Libraries:**

* Flask
* TensorFlow / Keras
* numpy
* Pillow
* Bootstrap 5 (CDN via HTML)

---

## 🖥️ Run the Application

After setting up dependencies and placing the model:

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to access the app.

---

## 📈 Results & Evaluation

* 📊 Final test accuracy: **95.35%**
* 📋 Classification report with precision, recall, and f1-score
* 📊 Confusion matrix visualization
* 📉 Accuracy and loss graphs over epochs
* 📌 Per-class performance metrics:

  * Glioma: 18.30%
  * Meningioma: 25.42%
  * No Tumour: 28.57%
  * Pituitary: 20.05%

---

## 🎯 Impact

Early detection of brain tumours is crucial in improving treatment outcomes. This AI system assists radiologists and healthcare professionals by providing:

* Faster diagnoses
* Improved consistency
* Automated second-opinion tool
  Ideal for deployment in low-resource hospitals and telemedicine solutions.

---

## 📢 Author

**P. Yaswanth Kumar**
GitHub: [yaswanthKumar44](https://github.com/yaswanthKumar44)

---

## 📜 License

Open-source project — feel free to fork, modify and contribute.

---

## 📌 Notes

* Model files >100MB are hosted externally (Drive link above)
* Ensure required folders like `static/uploads/` exist for image uploads
* For Grad-CAM visualizations, optionally enable the Grad-CAM section in `app.py`

---

## ⭐ Star This Repository

If you found this useful — please ⭐ star this repo and share it!

```

---
