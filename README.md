# Handwritten-Digits-Recognition
Handwritten Digits Recognition is a machine learning project that focuses on classifying grayscale images of handwritten digits (0 to 9) using the MNIST dataset. The project demonstrates the use of both classical ML models like Logistic Regression and SVM, as well as deep learning techniques using Convolutional Neural Networks (CNNs) with Keras.


# ✍️ Handwritten Digits Recognition using Machine Learning

This project implements a machine learning model to recognize handwritten digits (0–9) using the popular MNIST dataset. It showcases the application of image classification techniques using both traditional ML and deep learning models.

---

## 🧠 Objective

To accurately classify grayscale images of handwritten digits into one of the 10 digit classes using supervised learning.

---

## 📂 Dataset

- **Source:** [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/)
- **Size:** 70,000 total images (60,000 training + 10,000 testing)
- **Image Size:** 28x28 pixels, grayscale
- **Classes:** 10 (digits 0 to 9)

---

## 🛠️ Tools & Technologies

- Python
- NumPy, Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

---

## 🚀 Workflow

1. **Data Loading & Preprocessing**
   - Normalization of pixel values
   - Train-test split
   - Flattening or reshaping images as needed

2. **Model Training**
   - Logistic Regression (Baseline)
   - Support Vector Machine (SVM)
   - Convolutional Neural Network (CNN) using Keras

3. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - Visualizing predictions

4. **Model Improvement**
   - Hyperparameter tuning
   - Regularization
   - Dropout layers (for CNN)

---

## 📈 Results

| Model         | Accuracy |
|---------------|----------|
| Logistic Reg. | 92.3%    |
| SVM           | 96.7%    |
| CNN           | 99.1%    |

---

## 📸 Sample Outputs

### Confusion Matrix
![confusion matrix](images/confusion_matrix.png)

### Digit Prediction Examples
![predictions](images/sample_predictions.png)

---

## 📁 Folder Structure


---

## 💡 Future Enhancements

- Deploy model using Streamlit or Flask
- Use pretrained models or transfer learning (e.g., MobileNet for digit detection on images)
- Extend to multi-digit number recognition

---

## 🙋 Author

**Mohammed Faizal**  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License.
