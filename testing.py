import seaborn as sns
import pandas as pd

filepath = "./DataAnalysis_and_predictiveModelling_using_amesHousingDataSet/AmesHousing.csv"
data = pd.read_csv(filepath)

sns.heatmap(data=data)