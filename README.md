
# Predict Steph Curry's Shots 🏀

For this project, I used a dataset with all Steph Curry's NBA field goal attempts from October 2009 through June 2019 (regular season and playoffs). The dataset was collected with the [nba_api](https://github.com/swar/nba_api) Python library.

I created a model to predict whether Curry will make a shot based on his past perfomance.

# Reading and splitting data

To begin with, I imported all required libraries and then read the data in and began to wrangle the data by setting the index to the game date and formatting it as a DateTime object. I then printed the first 15 rows of the dataset to be able to get a better idea of the next best steps.

```
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
```

```
import pandas as pd
url = 'https://drive.google.com/uc?export=download&id=1fL7KPyxgGYfQDsuJoBWHIWwCAf-HTFpX'
df = pd.read_csv(url, parse_dates=['game_date'] ,index_col=['game_date'])

df.head(15)
```

Next, I engineered three new features in order to better visualise the data and give better predictions. These new features included:
['homecourt_adv']: Is the home team (htm) the Golden State Warriors (GSW)?
['sec_remain_in_period]: How many seconds remain in a given period.
['sec_remain_in_game]: How many seconds remain in a given game.
I then printed the first 5 rows of the dataset to ensure that the new features integrated correctly and did not cause any issues to the rest of the information.

```
df['homecourt_adv'] = df['htm'] == 'GSW'
df['sec_remain_in_period'] = df['minutes_remaining'] * 60 + df['seconds_remaining']
df['sec_remain_in_game'] = df['period'] * 60 * 12 + df['sec_remain_in_period']
df.head()
```
Then, I divided the DataFrame df into the feature matrix X and the target vector y, using the target column shot_made_flag.
After dividing the data, I could then split my dataset into training, validation, and testing sets. I chose to use a date cutoff method to split my data.
I printed the last 5 rows of my new X_train dataset to make sure that my target column was not present, and to ensure that the date cutoff was in the correct position.

```
X = df.drop(columns='shot_made_flag')
y = df['shot_made_flag']
```
```
train_end = '2017-06-12'
val_end = '2018-06-08'
test_end = '2019-06-05'

val_start = '2017-10-01'
test_start = '2018-10-01'

X_train = X.loc[:train_end]
y_train = y.loc[:train_end]
X_val = X.loc[val_start:val_end]
y_val = y.loc[val_start:val_end]
X_test = X.loc[test_start:test_end]
y_test = y.loc[test_start:test_end]
```

```
X_train.tail()
```

# Establish Baseline

I established the baseline accuracy score for this classification problem using the training set. I then saved the score to the variable `baseline_acc`.

```
baseline_acc = y_train.value_counts(normalize=True).max()

print('Baseline Accuracy:', baseline_acc)
```

# Build Model

Next, I built a model that included a ordinal encoder for categorical features and a random forest classifier. I then combined these components as well as an imputer to fill any missing values using a pipeline. I then fit the pipeline using the training dataset and checked the feature importances of the various columns to adjust for any potential leakage.

```
model = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(strategy='mean'),
    RandomForestClassifier(n_estimators=5, n_jobs=-1, max_depth=5, random_state=42)
)

model.fit(X_train, y_train)
```

```
importances = model.named_steps['randomforestclassifier'].feature_importances_
features = X_train.columns
pd.Series(importances, index = features).sort_index().tail(50).plot(kind='barh')
```
![Unknown](https://github.com/user-attachments/assets/8a6f6bd3-c4ec-4eba-bb98-9b8742562dd7)

I then checked the training and validation accuracy of my model, and asssigned the scores to `train_acc` and `val_acc` for further analysis.

```
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)

print('Training Accuracy Score:', train_acc)
print('Validation Accuracy Score:', val_acc)
```

# Tune Model
Now that I had a base model set, I began to tune the hyperparameters using [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) and checking my values in order to improve the accuracy of my model.

```
clf=make_pipeline(
        OrdinalEncoder(),
        SimpleImputer(),
        RandomForestClassifier(random_state=42, n_jobs=-1)
    )

param_grid = {
    'simpleimputer__strategy': ['mean', 'median'],
    'randomforestclassifier__max_depth': range(5,40,5),
    'randomforestclassifier__n_estimators': (5,40,5)
}

model_rscv = RandomizedSearchCV(clf, param_grid, n_iter=3, cv=None, n_jobs=-1, verbose=1)

model_rscv.fit(X_train, y_train);

print(model_rscv.best_params_)
print('')
print(model_rscv.best_score_)
print('')
print(model_rscv.best_estimator_)
print('')

print(model_rscv.score(X_val, y_val))
print('')

test_acc = model_rscv.score(X_test, y_test)

print('Testing Accuracy Score:', test_acc)
```
# Communication

To communicate my results, I chose to plot a confusion matrix for my model, and the calculate the precision and recall to check (again) on my accuracy to compare against my baseline.

```
plot_confusion_matrix = ConfusionMatrixDisplay.from_estimator

plot_confusion_matrix(model, X_test, y_test, cmap='Reds');
```
![Unknown-2](https://github.com/user-attachments/assets/f12f7475-34f1-4906-b6ad-7b7785c4f1e0)

```
model_precision = 436 / (436 + 301)
model_recall = 436 / (436 + 361)

print('Model precision', model_precision)
print('Model recall', model_recall)
```
