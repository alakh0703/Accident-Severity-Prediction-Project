# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:04:00 2023

@author: gitan
"""

### split 
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=17)
#### models  logistic , svc and rf

svc = SVC(random_state=42,probability = True)

pipeline_svc = Pipeline([
    ('svc', svc)
    ])

pipeline_svc.fit(X_train,y_train)
joblib.dump(pipeline_svc,'pipeline_svc.pkl')
scores_svc = cross_val_score(pipeline_svc,
                        X_train,
                        y_train,
                        n_jobs = -1,
                        cv=10,
                        verbose = 1)
print(scores_svc)
print(scores_svc.mean())


pred_svc = pipeline_svc.predict(X_test)
print(pred_svc)
pred_svc2 = pipeline_svc.predict(X_train)
print(pred_svc2)
accuracy_svc_train = metrics.accuracy_score(y_train,pred_svc2)
print(accuracy_svc_train)
accuracy_svc_test = metrics.accuracy_score(y_test,pred_svc)
print(accuracy_svc_test)

con_matrix_svc_train = confusion_matrix(y_train,pred_svc2)
print(con_matrix_svc_train)
con_matrix_svc_test = confusion_matrix(y_test,pred_svc)
print(con_matrix_svc_test)

class_report_svc_train = classification_report(y_train,pred_svc2)
print(class_report_svc_train)
class_report_svc_test = classification_report(y_test,pred_svc)
print(class_report_svc_test)

param_grid = {'svc__kernel': ['linear', 'rbf', 'poly'],
              'svc__C': [0.01, 0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              'svc__degree': [2, 3,4,5]
              }


grid_search_svc = GridSearchCV(estimator = pipeline_svc,
                                 param_grid = param_grid,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)
grid_search_svc.fit(X_train, y_train)
final_svc = grid_search_svc.best_estimator_
print(grid_search_svc.best_estimator_)
print(grid_search_svc.best_params_)
final_svc.fit(X_train, y_train)
joblib.dump(final_svc,'final_svc.pkl')
pred_final_svc = final_svc.predict(X_test)
accuracy_svc = metrics.accuracy_score(y_test,pred_final_svc)
print(accuracy_svc)

con_matrix_svc = confusion_matrix(y_test,pred_final_svc)
print(con_matrix_svc)

class_report_svc = classification_report(y_test,pred_final_svc)
print(class_report_svc)



# Make predictions on the test set
y_test_pred_svc = pipeline_svc.predict(X_test)
y_test_pred_final_svc = final_svc.predict(X_test)
y_test_proba_svc = pipeline_svc.predict_proba(X_test)[:,1]
y_test_proba_final_svc = final_svc.predict_proba(X_test)[:,1]

# Calculate the FPR and TPR for different threshold values
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, y_test_proba_svc)
roc_auc_svc = auc(fpr_svc, tpr_svc)

fpr_final_svc, tpr_final_svc, thresholds_final_svc = roc_curve(y_test, y_test_proba_final_svc)
roc_auc_final_svc = auc(fpr_final_svc, tpr_final_svc)

# Plot the ROC curves
plt.figure(figsize=(8,8))
plt.plot(fpr_svc, tpr_svc, color='darkorange', lw=2, label='SVC (AUC = %0.2f)' % roc_auc_svc)
plt.plot(fpr_final_svc, tpr_final_svc, color='green', lw=2, label='Final SVC (AUC = %0.2f)' % roc_auc_final_svc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()







# logistic
log = LogisticRegression(max_iter=10000)
pipeline_log = Pipeline([
    ('log', log)
    ])
pipeline_log.fit(X_train,y_train)
joblib.dump(pipeline_log,'pipeline_log.pkl')
scores_log = cross_val_score(pipeline_log,
                        X_train,
                        y_train,
                        cv=10,
                        verbose=1)
print(scores_log)
print(scores_log.mean())

pred_log = pipeline_log.predict(X_test)
pred_log2 = pipeline_log.predict(X_train)
accuracy_log_train = metrics.accuracy_score(y_train,pred_log2)
print(accuracy_log_train)
accuracy_log_test = metrics.accuracy_score(y_test,pred_log)
print(accuracy_log_test)


con_matrix_log_train = confusion_matrix(y_train,pred_log2)
print(con_matrix_log_train)
con_matrix_log_test = confusion_matrix(y_test,pred_log)
print(con_matrix_log_test)

class_report_log_train = classification_report(y_train,pred_log2)
print(class_report_log_train)
class_report_log_test = classification_report(y_test,pred_log)
print(class_report_log_test)


param_grid_log = {'log__penalty': ['l1', 'l2', 'elasticnet'],
              'log__C': [0.01, 0.1, 1, 10, 100],
              'log__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}
grid_search_log = GridSearchCV(estimator = pipeline_log,
                                 param_grid = param_grid_log,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)
grid_search_log.fit(X_train, y_train)
final_log = grid_search_log.best_estimator_
print(grid_search_log.best_estimator_)
print(grid_search_log.best_params_)

final_log.fit(X_train, y_train)
joblib.dump(final_log,'final_log.pkl')
pred_final_log = final_log.predict(X_test)
accuracy_log = metrics.accuracy_score(y_test,pred_final_log)
print(accuracy_log)

con_matrix_log = confusion_matrix(y_test,pred_final_log)
print(con_matrix_log)

class_report_log = classification_report(y_test,pred_final_log)
print(class_report_log)


# Make predictions on the test set
y_test_pred_log = pipeline_log.predict(X_test)
y_test_pred_final_log = final_log.predict(X_test)
y_test_proba_log = pipeline_log.predict_proba(X_test)[:,1]
y_test_proba_final_log = final_log.predict_proba(X_test)[:,1]

# Calculate the FPR and TPR for different threshold values
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_test_proba_log)
roc_auc_log = auc(fpr_log, tpr_log)

fpr_final_log, tpr_final_log, thresholds_final_log = roc_curve(y_test, y_test_proba_final_log)
roc_auc_final_log = auc(fpr_final_log, tpr_final_log)

# Plot the ROC curves
plt.figure(figsize=(8,8))
plt.plot(fpr_log, tpr_log, color='darkorange', lw=2, label='log (AUC = %0.2f)' % roc_auc_log)
plt.plot(fpr_final_log, tpr_final_log, color='green', lw=2, label='Final log (AUC = %0.2f)' % roc_auc_final_log)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()











#rf
rf = RandomForestClassifier()
pipeline_rf = Pipeline([
    ('rf', rf)
    ])
pipeline_rf.fit(X_train,y_train)
joblib.dump(pipeline_rf,'pipeline_rf.pkl')
scores_rf = cross_val_score(pipeline_rf,
                        X_train,
                        y_train,
                        cv=10,
                        verbose=1)
print(scores_rf)
print(scores_rf.mean())
pred_rf = pipeline_rf.predict(X_test)
pred_rf2 = pipeline_rf.predict(X_train)
accuracy_rf_train = metrics.accuracy_score(y_train,pred_rf2)
print(accuracy_rf_train)
accuracy_rf_test = metrics.accuracy_score(y_test,pred_rf)
print(accuracy_rf_test)

con_matrix_rf_train = confusion_matrix(y_train,pred_rf2)
print(con_matrix_rf_train)
con_matrix_rf_test = confusion_matrix(y_test,pred_rf)
print(con_matrix_rf_test)

class_report_rf_train = classification_report(y_train,pred_rf2)
print(class_report_rf_train)
class_report_rf_test = classification_report(y_test,pred_rf)
print(class_report_rf_test)


param_grid_rf = {'rf__n_estimators': [100, 150, 200],
                  'rf__criterion': ['gini', 'entropy'],
                  'rf__max_features': ['auto', 'sqrt', 'log2'],
                  'rf__max_depth': [4, 5, 6, 7, 8],
                  'rf__ccp_alpha': [0.0, 0.01, 0.1, 0.5, 1.0],
                  'rf__min_samples_split': [2, 5, 10],
                  'rf__min_samples_leaf': [1, 2, 4],
                  'rf__max_leaf_nodes': [None, 5, 10, 20]}   
grid_search_rf = GridSearchCV(estimator = pipeline_rf,
                                 param_grid = param_grid_rf,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)
grid_search_rf.fit(X_train, y_train)
final_rf = grid_search_rf.best_estimator_
print(grid_search_rf.best_estimator_)
print(grid_search_rf.best_params_)
final_rf.fit(X_train, y_train)
joblib.dump(final_rf,'final_rf.pkl')
pred_final_rf = final_rf.predict(X_test)
accuracy_rf = metrics.accuracy_score(y_test,pred_final_rf)
print(accuracy_rf)

con_matrix_rf = confusion_matrix(y_test,pred_final_rf)
print(con_matrix_rf)

class_report_rf = classification_report(y_test,pred_final_rf)
print(class_report_rf)




# Make predictions on the test set
y_test_pred_rf = pipeline_rf.predict(X_test)
y_test_pred_final_rf = final_rf.predict(X_test)
y_test_proba_rf = pipeline_rf.predict_proba(X_test)[:,1]
y_test_proba_final_rf = final_rf.predict_proba(X_test)[:,1]

# Calculate the FPR and TPR for different threshold values
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_test_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_final_rf, tpr_final_rf, thresholds_final_rf = roc_curve(y_test, y_test_proba_final_rf)
roc_auc_final_rf = auc(fpr_final_rf, tpr_final_rf)

# Plot the ROC curves
plt.figure(figsize=(8,8))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='rf (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_final_rf, tpr_final_rf, color='green', lw=2, label='Final rf (AUC = %0.2f)' % roc_auc_final_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()










# Initialize the model
nb = BernoulliNB()


# Define the pipeline
pipeline_nb = Pipeline([
    # ('col_transformer', transformer),
    ('nb', nb)
])

# Fit the model
pipeline_nb.fit(X_train, y_train)
joblib.dump(pipeline_nb,'pipeline_nb.pkl')


# Cross validate the model
scores_nb = cross_val_score(pipeline_nb, X_train, y_train, cv=10, verbose=1)
print(scores_nb)
print(scores_nb.mean())

# Make predictions on the test set
pred_nb = pipeline_nb.predict(X_test)
print(pred_nb)

# Evaluate the model performance
accuracy_nb_train = metrics.accuracy_score(y_train, pipeline_nb.predict(X_train))
print(accuracy_nb_train)

accuracy_nb_test = metrics.accuracy_score(y_test, pred_nb)
print(accuracy_nb_test)

con_matrix_nb_train = confusion_matrix(y_train, pipeline_nb.predict(X_train))
print(con_matrix_nb_train)

con_matrix_nb_test = confusion_matrix(y_test, pred_nb)
print(con_matrix_nb_test)

class_report_nb_train = classification_report(y_train, pipeline_nb.predict(X_train))
print(class_report_nb_train)

class_report_nb_test = classification_report(y_test, pred_nb)
print(class_report_nb_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'nb__alpha': [0.1, 0.5, 1.0],
    'nb__binarize': [0.0, 0.5, 1.0],
}
# Perform a grid search with cross-validation to find the best hyperparameters
grid_search_nb = GridSearchCV(estimator=pipeline_nb,
                              param_grid=param_grid,
                              scoring='accuracy',
                              refit=True,
                              n_jobs=-1,
                              verbose=3)

grid_search_nb.fit(X_train, y_train)

# Get the best model
final_nb = grid_search_nb.best_estimator_
print(grid_search_nb.best_estimator_)
print(grid_search_nb.best_params_)

# Fit the best model on the training set
final_nb.fit(X_train, y_train)
joblib.dump(final_nb,'final_nb.pkl')
# Make predictions on the test set using the best model
pred_final_nb = final_nb.predict(X_test)

# Evaluate the performance of the best model
accuracy_nb = metrics.accuracy_score(y_test, pred_final_nb)
print(accuracy_nb)

con_matrix_nb = confusion_matrix(y_test, pred_final_nb)
print(con_matrix_nb)

class_report_nb = classification_report(y_test, pred_final_nb)
print(class_report_nb)


# Make predictions on the test set
y_test_pred_nb = pipeline_nb.predict(X_test)
y_test_pred_final_nb = final_nb.predict(X_test)
y_test_proba_nb = pipeline_nb.predict_proba(X_test)[:,1]
y_test_proba_final_nb = final_nb.predict_proba(X_test)[:,1]

# Calculate the FPR and TPR for different threshold values
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_test_proba_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)

fpr_final_nb, tpr_final_nb, thresholds_final_nb = roc_curve(y_test, y_test_proba_final_nb)
roc_auc_final_nb = auc(fpr_final_nb, tpr_final_nb)

# Plot the ROC curves
plt.figure(figsize=(8,8))
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label='nb (AUC = %0.2f)' % roc_auc_nb)
plt.plot(fpr_final_nb, tpr_final_nb, color='green', lw=2, label='Final nb (AUC = %0.2f)' % roc_auc_final_nb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()



# Define the MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=(12, 8), activation='relu', solver='adam', max_iter=10, random_state=42)

# Define the pipeline
pipeline_nn = Pipeline([
    ('nn', model)
])

# Fit the model
pipeline_nn.fit(X_train, y_train)
joblib.dump(pipeline_nn,'pipeline_nn.pkl')


# Cross validate the model
scores_nn = cross_val_score(pipeline_nn, X_train, y_train, cv=10, verbose=1)
print(scores_nn)
print(scores_nn.mean())

# Make predictions on the test set
pred_nn = pipeline_nn.predict(X_test)
print(pred_nn)

# Evaluate the model performance
accuracy_nn_train = accuracy_score(y_train, pipeline_nn.predict(X_train))
print(accuracy_nn_train)

accuracy_nn_test = accuracy_score(y_test, pred_nn)
print(accuracy_nn_test)

con_matrix_nn_train = confusion_matrix(y_train, pipeline_nn.predict(X_train))
print(con_matrix_nn_train)

con_matrix_nn_test = confusion_matrix(y_test, pred_nn)
print(con_matrix_nn_test)

class_report_nn_train = classification_report(y_train, pipeline_nn.predict(X_train))
print(class_report_nn_train)

class_report_nn_test = classification_report(y_test, pred_nn)
print(class_report_nn_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'nn__hidden_layer_sizes': [(12, 8), (10, 10), (8,)],
    'nn__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'nn__max_iter': [10, 50, 100],
}

# Perform a grid search with cross-validation to find the best hyperparameters
grid_search_nn = GridSearchCV(estimator=pipeline_nn,
                              param_grid=param_grid,
                              scoring='accuracy',
                              refit=True,
                              n_jobs=-1,
                              verbose=3)

grid_search_nn.fit(X_train, y_train)

# Get the best model
final_nn = grid_search_nn.best_estimator_
print(grid_search_nn.best_estimator_)
print(grid_search_nn.best_params_)

# Fit the best model on the training set
final_nn.fit(X_train, y_train)
joblib.dump(final_nn,'final_nn.pkl')
# Make predictions on the test set using the best model
pred_final_nn = final_nn.predict(X_test)

# Evaluate the performance of the best model
accuracy_nn = accuracy_score(y_test, pred_final_nn)
print(accuracy_nn)

con_matrix_nn = confusion_matrix(y_test, pred_final_nn)
print(con_matrix_nn)

class_report_nn = classification_report(y_test, pred_final_nn)
print(class_report_nn)



# Make predictions on the test set
y_test_pred_nn = pipeline_nn.predict(X_test)
y_test_pred_final_nn = final_nn.predict(X_test)
y_test_proba_nn = pipeline_nn.predict_proba(X_test)[:,1]
y_test_proba_final_nn = final_nn.predict_proba(X_test)[:,1]

# Calculate the FPR and TPR for different threshold values
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_test_proba_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

fpr_final_nn, tpr_final_nn, thresholds_final_nn = roc_curve(y_test, y_test_proba_final_nn)
roc_auc_final_nn = auc(fpr_final_nn, tpr_final_nn)

# Plot the ROC curves
plt.figure(figsize=(8,8))
plt.plot(fpr_nn, tpr_nn, color='darkorange', lw=2, label='nn (AUC = %0.2f)' % roc_auc_nn)
plt.plot(fpr_final_nn, tpr_final_nn, color='green', lw=2, label='Final nn (AUC = %0.2f)' % roc_auc_final_nn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()










