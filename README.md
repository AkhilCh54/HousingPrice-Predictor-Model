# HousingPrice-Predictor-Model
A machine learning project that predicts housing prices using the Ames Housing dataset. This model leverages the power of Random Forest Regression to estimate home prices based on various features from the dataset.

## 📊 Model Overview
- Algorithm Used: Random Forest Regressor
- Mean Squared Error (MSE): 710,691,363.28
- Approximate RMSE: $26,657
- R² Score (Accuracy): 91.36%

This model performs well for real estate valuation tasks, with low prediction error and high variance explanation.
## 🔧 Features & Preprocessing

- Dropped rows with missing values in key columns (`Lot Area`, `SalePrice`, `Lot Config`)
- Applied **one-hot encoding** to several categorical features (e.g., `Lot Config`, `MS Zoning`, `House Style`, etc.)
- Removed multicollinear and less relevant features (`BsmtFin SF 2`, `TotRms AbvGrd`, `Functional`, etc.)
- Scaled numeric features using **StandardScaler**

## 📈 Future Enhancements
- Implement additional models (e.g., Gradient Boosting, XGBoost)
- Add hyperparameter tuning
- Integrate feature selection and dimensionality reduction
- Deploy as a web app with Flask or Streamlit
