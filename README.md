
# 🛡️ DDoS Attack Classification Using Machine Learning

This project applies **supervised machine learning** techniques to classify **DDoS (Distributed Denial of Service) attacks** from network traffic data using three core models: **Logistic Regression**, **Random Forest**, and **Neural Network**.

---

## 📁 Dataset

- **Source**: [CICDDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)
- **Description**: Labeled network flow data containing both **benign** and **malicious (DDoS)** traffic.
- **Preprocessing Steps**:
  - Removed missing/null values
  - Encoded categorical target labels using `LabelEncoder`
  - Feature-target split (`X`, `y`)
  - Train-test split: 80% training, 20% testing

---

## 🤖 Machine Learning Models Used

| Model                | Accuracy | F1 Score | Precision | Recall  |
|---------------------|----------|----------|-----------|---------|
| Logistic Regression | 93.57%   | 94.60%   | 90.71%    | 98.83%  |
| Random Forest       | 99.99%   | 99.99%   | 100.00%   | 99.98%  |
| Neural Network      | 97.49%   | 97.84%   | 96.08%    | 99.67%  |

---

## 🔍 Evaluation Metrics

- **Accuracy**: Overall correctness of predictions  
- **Precision**: Correct positive predictions out of total predicted positives  
- **Recall**: Correct positive predictions out of all actual positives  
- **F1 Score**: Harmonic mean of precision and recall  
- **Confusion Matrix**: Shows true/false positives and negatives per class

---

## ✅ Best Performing Model

**Random Forest Classifier**
- Achieved **99.99% accuracy**
- **100% precision** (no false positives)
- **99.98% recall** (nearly all attacks detected)

This makes it ideal for **real-time DDoS detection systems** where high accuracy and low false alarms are critical.

---

## ⚙️ How It Works

1. Load & preprocess the CICDDoS2019 dataset
2. Train 3 models:
   - Logistic Regression
   - Random Forest
   - Neural Network (e.g., MLP)
3. Evaluate using standard classification metrics
4. Compare results and choose the best model

---

## 💻 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

If using a neural network (MLPClassifier or Keras/TensorFlow):

```

---

## 🧠 Future Improvements

- Feature scaling for better convergence in Logistic Regression and Neural Network
- Hyperparameter tuning using `GridSearchCV`
- Handling class imbalance using SMOTE or class weights
- Real-time streaming detection integration using Flask API or Kafka

---

## 📂 Project Structure

```
├── DDOS_attack_Classification_Using_Machine_Learning.ipynb
├── CICDDoS2019.csv
├── README.md
└── requirements.txt
```

---

## ✍️ Author

**Rashedul Albab**  
📧 albabahmed74@gmail.com
🔗 [Linkedin:https://www.linkedin.com/in/rashedul-albab-91a50b2ab/]

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
