import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

wellbeing = pd.read_csv("D:\\Python\\IITM_DataScience_AI\\Projects\\Stress Calculator\\Wellbeing_and_lifestyle_data_Kaggle.csv")

wellbeing = wellbeing.drop('Timestamp', axis=1)
wellbeing = wellbeing.drop([10005])
age_dict = {'Less than 20' : 1, '21 to 35' : 2, '36 to 50' : 3, '51 or more' : 4}
wellbeing['AGE'] = pd.Series([age_dict[x] for x in wellbeing.AGE], index=wellbeing.index)
gender_dict = {'Female' : 1, 'Male' : 0}
wellbeing['GENDER'] = pd.Series([gender_dict[x] for x in wellbeing.GENDER], index=wellbeing.index)
wellbeing['DAILY_STRESS'] = wellbeing['DAILY_STRESS'].astype(int)

x = wellbeing.drop(['DAILY_STRESS'], axis=1)
y = wellbeing['DAILY_STRESS']

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1, test_size=.2)

my_model = RandomForestRegressor(n_estimators=100)
my_model.fit(x_train, y_train)
y_pred = my_model.predict(x_test)
'''
cols = ["FRUITS_VEGGIES", 
"PLACES_VISITED",
"CORE_CIRCLE",
"SUPPORTING_OTHERS",	
"SOCIAL_NETWORK",
"ACHIEVEMENT",	
"DONATION",
"BMI_RANGE",	
"TODO_COMPLETED",	
"FLOW",	
"DAILY_STEPS",	
"LIVE_VISION",	
"SLEEP_HOURS",	
"LOST_VACATION",	
"DAILY_SHOUTING",	
"SUFFICIENT_INCOME",	
"PERSONAL_AWARDS",	
"TIME_FOR_PASSION",	
"WEEKLY_MEDITATION",	
"AGE",
"GENDER",	
"WORK_LIFE_BALANCE_SCORE"]
'''

pickle.dump(my_model, open('model.pkl','wb'))