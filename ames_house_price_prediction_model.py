#Model used: Random Forest
#MSE       : 710691363.2811588 - The program predicts the value of houses with an error of approximately $26000
#R2 score  : 0.913580150 - 91.35% accuracy 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ames_data_set = pd.read_csv("AmesHousing.csv")
ames_data_set = ames_data_set.dropna(subset=['Lot Area', 'SalePrice','Lot Config'])

# One-Hot Encode the 'Lot Config' column
ames_data_set = pd.get_dummies(ames_data_set, columns=['Lot Config','Land Contour','MS Zoning','Street','Alley','Lot Shape','Utilities','Land Contour','Land Slope','Neighborhood','Condition 1','Condition 2','Bldg Type','House Style','Roof Style','Roof Matl','Exterior 1st','Exterior 2nd','Mas Vnr Type','Exter Qual','Exter Cond','Foundation','Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','Sale Type','Sale Condition','Misc Feature','Misc Val','Mo Sold','Pool QC','Garage Type','Garage Finish','Garage Qual','Garage Cond','Fireplace Qu','Central Air','Electrical','Heating','Heating QC','Kitchen Qual','BsmtFin Type 2'], drop_first=False)  # Use drop_first=True to avoid multicollinearity

# correlation_matrix = ames_data_set.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()

# Select target and predictors
X = ames_data_set.drop(['SalePrice','BsmtFin SF 2','Bsmt Unf SF','Total Bsmt SF','Low Qual Fin SF','TotRms AbvGrd','Functional','Paved Drive','Fence'],axis=1)  # Independent variables
y = ames_data_set['SalePrice']  # Dependent variable

# Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (for models that are sensitive to feature magnitude)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('Random Forest MSE:', mean_squared_error(y_test, y_pred))
print('Random Forest R2 Score:', r2_score(y_test, y_pred))
