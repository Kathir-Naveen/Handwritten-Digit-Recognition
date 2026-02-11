# ✍️ Handwritten Digit Recognition using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kathir-Naveen/Handwritten-Digit-Recognition/blob/main/PRCP_1002_HandwrittenDigits_Complete_Project.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive machine learning project that explores and compares **7 different classification algorithms** for recognizing handwritten digits (0-9) using the MNIST dataset. This project demonstrates the complete ML workflow from exploratory data analysis to model evaluation and real-world deployment considerations.

---

## 🎯 Project Overview

This project analyzes the **MNIST Handwritten Digit Dataset** and builds multiple classification models to identify handwritten numbers with high accuracy. Rather than focusing solely on achieving the highest accuracy, we evaluate models based on **production constraints** including speed, scalability, interpretability, and deployment feasibility.

### Why This Matters
Handwritten digit recognition powers real-world applications like:
- 📱 Mobile check deposit systems in banking apps
- 📮 Automated postal code recognition for mail sorting
- 📝 Digitizing handwritten forms and documents
- 🔢 Automatic data entry systems

---

## 📊 What's Inside This Repository

```
Handwritten-Digit-Recognition/
│
├── PRCP_1002_HandwrittenDigits_Complete_Project.ipynb    # Main project notebook
└── README.md                                              # Project documentation
```

### Notebook Contents
- **Exploratory Data Analysis (EDA)**: Visualizing digit patterns and distributions
- **Data Preprocessing**: Normalization and feature scaling
- **Model Training**: 7 classification algorithms implemented and compared
- **Performance Evaluation**: Accuracy, confusion matrices, and error analysis
- **Production Analysis**: Real-world deployment recommendations
- **Lessons Learned**: Challenges faced and solutions discovered

---

## 🚀 Quick Start

### Option 1: Open in Google Colab (Recommended)
Click the badge below to run the notebook instantly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kathir-Naveen/Handwritten-Digit-Recognition/blob/main/PRCP_1002_HandwrittenDigits_Complete_Project.ipynb)

**No installation required!** All dependencies are pre-installed in Colab.

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/Kathir-Naveen/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter

# Launch Jupyter Notebook
jupyter notebook PRCP_1002_HandwrittenDigits_Complete_Project.ipynb
```

---

## 📈 Models Compared

We trained and evaluated **7 different machine learning algorithms**:

| Model | Accuracy | Training Time | Best Use Case |
|-------|----------|---------------|---------------|
| **Random Forest** | **95.8%** | ~45 sec | ⭐ Best overall - Balanced accuracy & speed |
| SVM (RBF Kernel) | 95.1% | ~180 sec | High accuracy, slower training |
| Neural Network (MLP) | 94.7% | ~120 sec | Scalable with large datasets |
| Logistic Regression | 92.3% | ~15 sec | Fast baseline, good for quick tests |
| K-Nearest Neighbors | 91.8% | ~5 sec (train), slow inference | Small datasets only |
| Decision Tree | 87.6% | ~8 sec | Interpretable but lower accuracy |
| Naive Bayes | 83.2% | ~3 sec | Fast but assumes independence |

> **Note**: These results were obtained using 20,000 training samples and 4,000 test samples from the MNIST dataset.

---

## 🔍 Key Findings

### 1. Dataset Size Dramatically Affects Model Rankings
One of the most important discoveries:

| Dataset Size | Best Performing Model |
|--------------|----------------------|
| ~10,000 samples | SVM |
| ~20,000 samples | Random Forest |
| 70,000 samples (full MNIST) | Neural Network (MLP) |

**Takeaway**: The "best" model isn't universal—it depends heavily on dataset size, not just algorithm choice.

### 2. Random Forest Excels for Medium-Sized Datasets
- Achieved **95.8% accuracy** with minimal hyperparameter tuning
- Fast inference time suitable for real-time applications
- Robust to noisy inputs and requires minimal preprocessing
- Offers feature importance insights (useful for debugging)

### 3. Data Normalization is Critical
- **Before normalization**: ~60% accuracy across all models
- **After normalization** (0-255 → 0-1): ~95% accuracy
- Simple preprocessing step yielded **35% improvement**!

### 4. Common Confusion Patterns
The confusion matrix analysis revealed:
- **4 and 9**: Most frequently confused (similar shapes)
- **3 and 5**: Second most common confusion
- **7 and 1**: Occasionally misclassified
- **0**: Rarely misclassified (distinct circular shape)

This shows the models are learning genuine patterns, not memorizing data.

### 5. Production ML ≠ Research ML
For real-world deployment, consider:
- **Banking systems**: Prioritize accuracy (Random Forest/SVM)
- **Mobile apps**: Prioritize speed (Random Forest/optimized MLP)
- **Research**: Use ensemble methods (Random Forest + SVM + MLP)

---

## 🚧 Challenges Faced & Solutions

### Challenge 1: Poor Initial Performance (~60% Accuracy)
**Problem**: Early experiments yielded disappointingly low accuracy across all models.

**Root Cause**: Pixel values (0-255) were not normalized, causing numerical instability in gradient-based algorithms.

**Solution**: 
- Normalized pixel values to [0, 1] range
- Applied StandardScaler for zero-mean, unit-variance scaling
- **Result**: Accuracy jumped from 60% to 95%+

---

### Challenge 2: Insufficient Training Data
**Problem**: Initial experiments used only 10,000 samples, leading to unstable and inconsistent results.

**Root Cause**: Complex models like Random Forest require sufficient data to learn robust patterns.

**Solution**:
- Increased training set to 20,000 samples
- Implemented stratified sampling to maintain class balance
- **Result**: Random Forest performance improved from 89% to 95.8%

---

### Challenge 3: Neural Network Underfitting
**Problem**: MLP achieved only 70% accuracy despite being a powerful model.

**Root Cause**: `max_iter` parameter was set too low (20 iterations), causing premature convergence.

**Solution**:
- Increased `max_iter` to 100-200 iterations
- Added early stopping to prevent overfitting
- Adjusted learning rate schedule
- **Result**: MLP accuracy improved from 70% to 94.7%

---

### Challenge 4: Assumption Bias
**Problem**: Assumed SVM would always outperform Random Forest based on theoretical knowledge.

**Root Cause**: Relied on textbook rankings rather than empirical validation.

**Solution**:
- Adopted experimental mindset: "Test, don't assume"
- Benchmarked all models on the actual dataset
- **Result**: Random Forest outperformed SVM (95.8% vs 95.1%) with 4× faster training

---

### Challenge 5: Accuracy-Only Focus
**Problem**: Initially evaluated models solely on accuracy, ignoring deployment constraints.

**Root Cause**: Research mindset rather than production-oriented thinking.

**Solution**:
- Evaluated models on multiple criteria:
  - Accuracy
  - Training time
  - Inference speed
  - Memory footprint
  - Interpretability
- Created deployment scenarios matching real-world use cases
- **Result**: More informed model selection for different deployment contexts

---

## 🔮 Future Improvements

We've identified several enhancements for the next iteration:

### 1. Deep Learning Implementation
- [ ] Implement Convolutional Neural Network (CNN)
- [ ] Expected accuracy: 98-99%
- [ ] Compare classical ML vs deep learning trade-offs

### 2. Full Dataset Utilization
- [ ] Train on complete 70,000-sample MNIST dataset
- [ ] Implement 5-fold cross-validation for robust evaluation
- [ ] Test model stability across different data splits

### 3. Data Augmentation
- [ ] Apply rotation (±15 degrees)
- [ ] Add random shifts (±2 pixels)
- [ ] Introduce elastic deformations
- [ ] Goal: Improve model generalization

### 4. Ensemble Methods
- [ ] Build voting classifier (Random Forest + SVM + MLP)
- [ ] Implement soft voting with probability averaging
- [ ] Target accuracy: 97-98% with classical ML

### 5. Feature Engineering
- [ ] Extract domain-specific features:
  - Stroke density
  - Aspect ratio
  - Pixel cluster patterns
  - Edge detection features
- [ ] Combine with raw pixels for hybrid models

### 6. Production Deployment
- [ ] **Web Application**:
  - Frontend: HTML5 Canvas for digit drawing
  - Backend: Flask/FastAPI API
  - Real-time prediction with confidence scores
- [ ] **Model Optimization**:
  - Model compression and quantization
  - ONNX export for cross-platform deployment
- [ ] **API Development**:
  - RESTful endpoint for batch predictions
  - Docker containerization
  - CI/CD pipeline

### 7. Advanced Analysis
- [ ] Implement confidence calibration
- [ ] Add prediction uncertainty estimates
- [ ] Create interactive error analysis dashboard
- [ ] Test on real-world handwritten data (not just MNIST)

---

## 💡 Key Learnings

This project reinforced several important ML principles:

1. **Preprocessing Matters More Than Algorithms**
   - Simple normalization improved accuracy by 35%
   - Clean data beats complex models on dirty data

2. **Context is King**
   - No universally "best" algorithm
   - Model choice depends on dataset size, constraints, and deployment environment

3. **Always Validate Assumptions**
   - Theoretical knowledge ≠ empirical performance
   - Test everything on your actual data

4. **Production ML is Different from Research ML**
   - Accuracy is important but not the only metric
   - Speed, interpretability, and maintainability matter equally

5. **Confusion Matrix > Accuracy Score**
   - Reveals which digits are genuinely hard to distinguish
   - Helps identify systematic errors vs random mistakes

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Deep Learning**: TensorFlow/Keras (for MNIST dataset loading)
- **Environment**: Google Colab / Jupyter Notebook

---

## 📚 Dataset

**MNIST Handwritten Digit Database**
- **Source**: [Keras Datasets](https://keras.io/api/datasets/mnist/) / Original [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Size**: 70,000 grayscale images
  - Training set: 60,000 images
  - Test set: 10,000 images
- **Image Dimensions**: 28×28 pixels (784 features)
- **Classes**: 10 digits (0-9)
- **Format**: Grayscale pixel intensity values (0-255)

---

## 👥 Authors

**Vishnu K** & **NaveenKumar K**

*This project was completed as part of the DataMites Internship Program.*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DataMites** for providing the internship opportunity and project framework
- **MNIST Dataset** creators (Yann LeCun, Corinna Cortes, Christopher J.C. Burges)
- **Scikit-learn** community for excellent ML library and documentation
- **TensorFlow/Keras** team for easy dataset access

---

## 📞 Contact

Have questions or suggestions? Feel free to:
- Open an [issue](https://github.com/Kathir-Naveen/Handwritten-Digit-Recognition/issues)
- Submit a [pull request](https://github.com/Kathir-Naveen/Handwritten-Digit-Recognition/pulls)
- Connect on GitHub: [@Kathir-Naveen](https://github.com/Kathir-Naveen)

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a star! It helps others discover the project.

[![GitHub stars](https://img.shields.io/github/stars/Kathir-Naveen/Handwritten-Digit-Recognition?style=social)](https://github.com/Kathir-Naveen/Handwritten-Digit-Recognition/stargazers)

---

<div align="center">

**Built with ❤️ using Python and Machine Learning**

[🔝 Back to Top](#️-handwritten-digit-recognition-using-machine-learning)

</div>
