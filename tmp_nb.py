# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline , make_pipeline
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split ,   KFold
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# ----

df = pd.read_csv("/kaggle/input/global-ai-and-data-jobs-salary-dataset/global_ai_jobs.csv")

# ----

df

# ----



# ----

cat_col = [ 'country' , 'job_role' , 'ai_specialization'  , 'experience_level' , 'education_required' , 'industry'
          'company_size' , 'work_mode' ]

num_col = ['id', 'experience_years', 'salary_usd', 'bonus_usd', 'interview_rounds',
       'year', 'weekly_hours', 'company_rating', 'job_openings',
       'hiring_difficulty_score', 'layoff_risk', 'ai_adoption_score',
       'company_funding_billion', 'economic_index', 'ai_maturity_years',
       'offer_acceptance_rate', 'tax_rate_percent', 'vacation_days',
       'skill_demand_score', 'automation_risk', 'job_security_score',
       'career_growth_score', 'work_life_balance_score', 'promotion_speed',
       'salary_percentile', 'cost_of_living_index', 'employee_satisfaction']

# ----

scaler = StandardScaler()
df[num_col] = scaler.fit_transform(df[num_col])

# ----

LE = LabelEncoder()
df['country'] = LE.fit_transform(df['country'])
df['job_role'] = LE.fit_transform(df['job_role'])
df['ai_specialization'] = LE.fit_transform(df['ai_specialization'])
df['experience_level'] = LE.fit_transform(df['experience_level'])
df['education_required'] = LE.fit_transform(df['education_required'])
df['industry'] = LE.fit_transform(df['industry'])
df['company_size'] = LE.fit_transform(df['company_size'])
df['work_mode'] = LE.fit_transform(df['work_mode'])

# ----

X = df.drop(columns=['salary_usd' , 'id'])
y = df['salary_usd']

# ----

X_train , x_test , Y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=22)

# ----

kf = KFold(n_splits=5 , shuffle=True , random_state=23)

# ----

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))

test_preds_lgb = np.zeros(len(x_test))
test_preds_xgb = np.zeros(len(x_test))

# ----

lgb = LGBMRegressor(n_estimators=500 , learning_rate=0.5)
xgb = XGBRegressor(n_estimators=500 , learning_rate=0.5 )

# ----

for train_index , test_index in kf.split(X_train):
    X_train_fold = X_train.iloc[train_index]
    X_test_fold = X_train.iloc[test_index]

    Y_train_fold = Y_train.iloc[train_index]
    Y_test_fold = Y_train.iloc[test_index]

    # --- LightGBM ---
    lgb.fit(X_train_fold , Y_train_fold)
    oof_lgb[test_index] = lgb.predict(X_test_fold )
    test_preds_lgb += lgb.predict(x_test) / 5

    # --- XGBoost ---

    xgb.fit(X_train_fold , Y_train_fold)
    oof_xgb[test_index] = xgb.predict(X_test_fold)
    test_preds_xgb += xgb.predict(x_test) / 5
  

    





# ----

meta_X = np.column_stack([oof_lgb, oof_xgb])
meta_test = np.column_stack([test_preds_lgb, test_preds_xgb])

# ----

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X, Y_train)

# ----

final_oof_preds = meta_model.predict(meta_X)
final_test_preds = meta_model.predict(meta_test)

# ----

test_r2 = r2_score(y_test, final_test_preds)
MAE = mean_absolute_error(y_test, final_test_preds)
MSE = mean_squared_error(y_test, final_test_preds)

# ----

print("-" * 30)
print(f"Stacking Test R2 Score:  {test_r2:.4f}")
print(f"mean_absolute_error:  {MAE:.4f}")
print(f"mean_squared_error:  {MSE:.4f}")
print("-" * 30)

# ----



# ----



# ----



# ----

