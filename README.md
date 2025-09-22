# Air Quality Forecasting with Deep Learning

## 📌 Project Overview
Air pollution, especially fine particulate matter (PM2.5), poses serious risks to public health and the environment. This project develops a **deep learning model** to forecast PM2.5 concentrations in Beijing using time-series data.  

We implement and compare different architectures, with the final optimized **ConvLSTM model** achieving strong performance on both validation and Kaggle test datasets.

---

## 📊 Dataset
- Source: Beijing PM2.5 dataset (UCI & Kaggle sources).
- Features: Meteorological data (temperature, pressure, humidity, wind speed, etc.) and PM2.5 concentrations.
- Target: PM2.5 concentration levels.
- Preprocessing:
  - Missing values imputed
  - Normalization applied
  - Train-validation-test split

---

## ⚙️ Methodology
The project followed a **machine learning pipeline**:

1. **Exploratory Data Analysis (EDA)**  
   - Trends, correlations, and seasonality studied  
   - Outliers and missing data addressed  

2. **Model Development**  
   - Baseline: Simple LSTM  
   - Optimized: Conv1D + Stacked BiLSTM + Attention + Dense layers  

   ### Final Model Architecture:
   - 1D Convolution (feature extraction)  
   - Batch Normalization  
   - Bidirectional LSTM layers  
   - Attention mechanism  
   - Global Average Pooling  
   - Dense + Dropout  
   - Output layer (linear activation)  

3. **Training & Optimization**  
   - Loss: Mean Squared Error (MSE)  
   - Optimizer: Adam (lr=3e-4)  
   - Regularization: Dropout & Early stopping  

---

## 📈 Results and Discussion
- **Validation RMSE**: **21.05**  
- **Kaggle Public Leaderboard RMSE**: **4256.7329**  
- Improvement: **16% reduction** compared to baseline.  

### Key Findings:
- Convolutional layers improved feature extraction.  
- Longer sequence lengths captured temporal context better.  
- Lower learning rates stabilized training.  
- Dropout + Early stopping prevented overfitting.  
- Stacked BiLSTMs overcame vanishing gradient challenges.  

### RMSE Formula:
\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

---

## ✅ Conclusion and Future Work
The project successfully developed a deep learning model capable of forecasting PM2.5 concentrations with high accuracy.  

**Future Work:**
- Integrate external datasets (traffic, holidays, industrial activity).  
- Explore advanced architectures (Transformers, Temporal Fusion Transformers).  
- Use Bayesian optimization for hyperparameter tuning.  
- Apply explainability techniques (SHAP, attention visualization).  

---

## 🛠️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/air-quality-forecasting.git
   cd air-quality-forecasting
   pip install -r requirements.txt
  jupyter notebook AirQuality_Forecasting.ipynb
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64)
References

S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.

A. Vaswani et al., “Attention is All You Need,” NeurIPS, 2017.

J. Brownlee, Deep Learning for Time Series Forecasting, Machine Learning Mastery, 2019.

Beijing PM2.5 Dataset, UCI Machine Learning Repository
.

Kaggle Air Quality Challenge, Kaggle
.

A. Graves, Supervised Sequence Labelling with Recurrent Neural Networks, Springer, 2012.

D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015.

Scikit-learn Developers, “scikit-learn Documentation,” 2023. [Online]. Available: https://scikit-learn.org/stable/

World Health Organization, “WHO Global Air Quality Guidelines,” Geneva, 2021.
