
# 💼 Employee Attrition Prediction

## 📌 Project Description

This project implements an **Employee Attrition Prediction system** using Machine Learning.
The model predicts whether an employee is **likely to leave (Attrition = Yes)** or **stay (Attrition = No)** based on multiple work-related and personal attributes.

The project includes:

* Data analysis and model training
* A trained classification model
* A **Streamlit web application** for real-time attrition prediction

This project is developed as a **mini project** for academic learning and practical exposure to end-to-end machine learning workflows.

---

## 📁 Dataset Information

* **Dataset Name:** HR Employee Attrition Dataset
* **File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`

The dataset contains employee-related information such as:

* Demographic details
* Job role and department
* Salary and work experience
* Work-life balance and satisfaction levels
* Attrition status (target variable)

---

## 🛠️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib / Pickle
* Streamlit

---

## 📂 Project Structure

```
Attrition-prediction
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── employee-attrition.ipynb
├── attrition_model.joblib
├── employee-attrition.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Selvaganapathy-k/Attrition-prediction
cd Attrition-prediction
```

---

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🌐 Live Application

🔗 **Streamlit App URL:**
[https://attrition-prediction-r9ssk26dydjbnslvffe2ar.streamlit.app/](https://attrition-prediction-r9ssk26dydjbnslvffe2ar.streamlit.app/)

---

## 🔍 Model Details

* Problem Type: **Binary Classification**
* Target Variable: **Attrition (Yes / No)**
* Model saved using **Joblib / Pickle**
* Handles both numerical and categorical employee features

---

## 📈 Features

* Interactive and user-friendly interface
* Predicts employee attrition in real time
* Displays prediction probability
* Accepts multiple employee attributes as input

---

## 🎓 Learning Outcomes

* Understanding employee attrition problems
* Data preprocessing and feature handling
* Training and saving ML classification models
* Building and deploying ML apps using Streamlit
* Structuring complete ML projects on GitHub

---

## 📌 Notes

* Virtual environment folders (`venv`, `myvenv`) are not included in the repository.
* All dependencies are listed in `requirements.txt`.

---

## ✍️ Author

**Selvaganapathy K**
Computer Science Student

---

## 🏁 Conclusion

This project demonstrates how machine learning can be applied to HR analytics to predict employee attrition and support data-driven decision-making.
