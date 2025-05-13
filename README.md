# Fake-currency-detection-system
A Streamlit app that uses deep learning to detect fake Indian currency notes of ₹50, ₹100, ₹500, and ₹2000 denominations. Built using ResNet50 and TensorFlow.
## 🚀 Features

- 📸 Upload currency note images
- 💡 Detect if the note is **real or fake**
- 🧠 Uses **ResNet50-based CNN models**
- 🌗 Toggle between **Dark Mode** and Light Mode
- 📜 Keeps detection history for current session
- 🔍 Suggests safety steps for suspected fake notes

---

## 🧠 Technology Stack

- Python  
- Streamlit  
- TensorFlow / Keras  
- ResNet50 (Transfer Learning)  
- PIL & NumPy for image preprocessing  

---

## 🗂️ Project Structure
.
├── app.py # Main Streamlit web app
├── 50rs Identification.ipynb # Model training notebook for ₹50 notes
├── 100rs identification.ipynb # Model training notebook for ₹100 notes
├── 500rs Identification.ipynb # Model training notebook for ₹500 notes
├── 2000rs Identiifcation.ipynb # Model training notebook for ₹2000 notes
├── model_50rs.keras # Trained model for ₹50
├── model_100rs.keras # Trained model for ₹100
├── model_500rs.keras # Trained model for ₹500
├── model_2000rs.keras # Trained model for ₹2000

## ▶️ How to Run

Make sure you have Python and required packages installed. Then run:
"python -m streamlit run app.py" in the command prompt.

## 🧪 How It Works

- Upload an image of a currency note.
- Select the denomination.
- The app preprocesses the image.
- A pretrained ResNet50 model predicts whether it’s Real or Fake.
- If fake, safety suggestions are displayed.
