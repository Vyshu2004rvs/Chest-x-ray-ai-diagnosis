# 🩺 Chest X-ray Multi-task Classification with Explainability (XAI)

A deep learning-powered project for classifying chest X-ray images into **COVID-19**, **Pneumonia**, or **Normal**, along with **severity prediction** and **Grad-CAM visual explanations** to highlight affected regions.

---

## 🔍 Features

- ✅ Multi-task model: Disease classification + Severity prediction
- 🎯 Model trained on COVID-19, Pneumonia & Normal X-ray images
- 🔥 Grad-CAM heatmaps to visualize prediction focus areas
- 🖥️ Frontend: Upload and view predictions with explanation
- 🚀 FastAPI + TensorFlow backend

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: FastAPI, TensorFlow
- **Model**: DenseNet121 with dual output heads
- **Explainability**: Grad-CAM visualization

---
## How to Use

Run the Final.ipynb File and Save the Model in the models Folder
To Run Backend:uvicorn main:app --reload,
Open index1.html in your browser.
or 
Can use the Gradio Code in the Final_Gradio_Code.ipynb

---
## Use responsibly for academic and healthcare research. Not intended for clinical use.

---
##🙌 Acknowledgements
Kaggle Chest X-ray COVID-19 & Pneumonia Dataset
