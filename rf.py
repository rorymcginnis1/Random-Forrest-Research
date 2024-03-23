#Rory McGinnis
#Copyright (c) 2023 Rory McGinnis
#10-28-2023

#import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

#read the data

path = "i1Positive.csv"
data = pd.read_csv(path)

#split the data

X = data.iloc[:, :-1]
y = data['Label']

#splitting the data, creating 5 fold cross validation train and test sets for predictions and labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#running built in Random Forrest OOB training, OOB accuracy the same as cross-val scores

rf = RandomForestClassifier(max_features ="auto",n_estimators=1500, oob_score=True)

#fitting the model with our data

rf.fit(X_train, y_train)

#making predictions on the data

y_pred = rf.predict(X_test)

#getting the oob score

oob_accuracy = rf.oob_score_
print(f'OOB Accuracy: {oob_accuracy:.2f}')

# print report and matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# testing cross validation 5-fold to verrify OOB_accuracy above


cv_scores = cross_val_score(rf, X_train, y_train, cv=5)

#printing the accuracy accross each fold

print("Cross-Validation Accuracy Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.2f}")

#printing the average accuracy

average_accuracy = np.mean(cv_scores)
print(f"Average Accuracy: {average_accuracy:.2f}")



#printing the important features

features = rf.feature_importances_


feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': features})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

top_n = 10
print(f'Top {top_n} Most Important Features:')
print(feature_importance.head(top_n))


#plotting the top features in a graph

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), feature_importance['Importance'][:top_n], align='center')
plt.yticks(range(top_n), feature_importance['Feature'][:top_n])
plt.xlabel('Feature Importance')
plt.title('Top Feature Importances')
plt.gca().invert_yaxis()
plt.show()

#making predictions on random samples

random_samples = X_test.sample(n=2)
samples_features = random_samples

predicted_labels = rf.predict(samples_features)
predicted = random_samples.copy()
predicted['Predicted_Label'] = predicted_labels
print("Predictions for Two Random Samples:")
print(predicted)