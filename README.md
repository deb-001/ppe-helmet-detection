# 🛡️ PPE Helmet Detection System (Flask App)

## 📌 Overview

This project is a **Real-Time PPE Helmet Detection System** built using **Faster R-CNN (ResNet-50 FPN)** and deployed via a **Flask web application**.

It detects whether a person is wearing a helmet or not using:

* 📷 Live Webcam Feed
* 🖼️ Pre-captured Image Upload

---

## 🚀 Features

* ✅ Real-time helmet detection using webcam
* ✅ Upload and test pre-captured images
* ✅ Detects:

  * 👤 Person
  * 🧑 Head
  * ⛑️ Helmet
* ✅ Classifies:

  * 🟢 **Helmet Worn**
  * 🔴 **No Helmet**
* ✅ Optimized for both speed (webcam) and accuracy (image upload)
* ✅ Clean Flask-based UI

---

## 🧠 Model Details

* **Model:** Faster R-CNN
* **Backbone:** ResNet-50 with Feature Pyramid Network (FPN)
* **Framework:** PyTorch + TorchVision
* **Training Platform:** Kaggle

### 📦 Classes:

| Class ID | Label  |
| -------- | ------ |
| 1        | Person |
| 2        | Head   |
| 3        | Helmet |

---

## ⚙️ Tech Stack

* Python 🐍
* PyTorch 🔥
* TorchVision
* OpenCV
* Flask 🌐
* NumPy

---

## 📁 Project Structure

```
ppe_flask_app/
│
├── app.py                 # Main Flask application
├── utils_ppe.py          # Helper functions
├── requirements.txt      # Dependencies
├── README.md
├── .gitignore
│
├── templates/            # HTML files
│   └── index.html
│
├── static/               # CSS / JS / assets
│
└── models/               # Model files (NOT included in repo)
```

---

## 📥 Model Download

⚠️ Model files are not included due to size limitations.

👉 Download trained model from:
🔗 **[https://drive.google.com/drive/folders/1oWzy4nC8kkF7gHRmgSBlTrzWID5JaWfC?usp=sharing]**

After downloading, place the model inside:

```
models/
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ppe-helmet-detection.git
cd ppe-helmet-detection
```

---

### 2️⃣ Create environment (Anaconda recommended)

```bash
conda create -n ppe python=3.10
conda activate ppe
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the application

```bash
python app.py
```

---

### 5️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

## 🎯 Usage

### 📷 Webcam Detection

* Automatically starts when app loads
* Shows real-time detection
* 🟢 Green → Helmet Worn
* 🔴 Red → No Helmet

---

### 🖼️ Image Upload

* Upload any image
* Model detects helmets and displays results

---

## ⚠️ Known Issues

* Detection may vary with:

  * Poor lighting
  * Low-resolution images
  * Extreme angles
* Webcam performance depends on system hardware

---

## 🔮 Future Improvements

* 🔹 Deploy on cloud (Render / AWS / GCP)
* 🔹 Add alert system for safety violations
* 🔹 Improve model accuracy with larger dataset
* 🔹 Mobile app integration
* 🔹 Multi-PPE detection (vest, gloves, etc.)

---

## 👨‍💻 Author

**Debanjan Kauri**

---

## 📜 License

This project is licensed under the MIT License.

---
