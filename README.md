# 🌸 Iris Dataset Analysis and Classification

### 👩‍💻 Author
**Himani Agarwal**  
Roll No: PCE23AD024  
Poornima College of Engineering, Jaipur  
Department of Advance Computing  
Faculty: Appoorva Bansal  
Date: 05/11/2025

---

## 📊 1. Exploratory Data Analysis (EDA)

The Iris dataset contains **150 samples** of iris flowers from three species:
- Setosa  
- Versicolor  
- Virginica  

Each observation includes **4 features**:
- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

### 🔍 Key Insights
- **Petal features** show clear separation between species.  
- **Setosa** forms a distinct cluster.  
- **Versicolor** and **Virginica** slightly overlap.  
- **Petal Length ↔ Petal Width** correlation ≈ **0.96**  
- **Sepal Width** shows mild outliers.

---

## 🤖 2. Classification Models

| Model              | Accuracy | Kappa | Observation |
|--------------------|----------|--------|--------------|
| Decision Tree      | 93.33%   | 0.90   | Minor overlap between Versicolor & Virginica |
| SVM (Linear)       | 96.67%   | 0.95   | Better separation between species |
| KNN (k=5)          | 100%     | 1.00   | Perfect classification |

---

## ✅ 3. Conclusion
All models performed very well, but **KNN achieved 100% accuracy**, making it the best classifier.  
**Petal-based features** were most discriminative, while **sepal features** added support.

---

 4. Technologies Used
- Python 🐍  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- scikit-learn  

---

## ▶️ 5. Run Instructions
```bash
# Clone this repository
git clone https://github.com/<your-username>/Iris_Assignment.git
cd Iris_Assignment

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook iris_analysis.ipynb
