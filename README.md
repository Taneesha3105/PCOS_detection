# PCOS Disease Detection  

### Team: Code Catalysts  

---

## About the Project  
Polycystic Ovary Syndrome (PCOS) is a common health issue among women and is often detected late due to diverse symptoms and dependency on clinical evaluation.  
Our project tries to simplify early screening using Machine Learning and Deep Learning.  

We built:  
- **Clinical data models** → Decision Tree, Logistic Regression, Random Forest (with SMOTE to fix class imbalance).  
- **Image models** → ResNet-18 and ResNet-50 trained on ovarian ultrasound images.  
- **Web App** → A Streamlit interface where a user can upload an image and get a prediction (powered by ResNet-50).  
- **Chatbot** → Integrated a Gemini-based chatbot that can answer basic PCOS-related queries inside the app.  

---

## Tech Stack  
- **Python** for programming  
- **Pandas, NumPy** for data handling  
- **Scikit-learn** for ML models  
- **PyTorch** for ResNet models  
- **Matplotlib, Seaborn** for visualizations  
- **Streamlit** for deployment  
- **Google GenAI API** (Gemini) for chatbot  

---

## System Flow  
1. **Clinical data pipeline** → ML models on features like BMI, cycle length, hormone levels.  
2. **Image pipeline** → Ultrasound images classified using ResNet-50.  
3. **Integration** → Best model wrapped in Streamlit.  
4. **User interaction** → Upload → Prediction → (optional) Chatbot help.  

---

## Why it Matters  
- **Early detection** → Helps reduce long-term complications like infertility and diabetes.  
- **Accessible** → Can be used outside hospitals with just a browser.  
- **Assistive** → Gives doctors an AI-based second opinion.  
- **Awareness** → Helps women understand and track PCOS risk better.  

---

## How to Run  
Clone this repo and install dependencies:  
```bash
git clone https://github.com/Taneesha3105/PCOS_detection
cd PCOS_detection
pip install -r requirements.txt
