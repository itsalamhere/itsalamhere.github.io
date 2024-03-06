# 🧑‍💻Data Science Portfolio by Mohamad Alamsyah

Hello, I'm Alam, a biomedical engineering fresh graduate from Indonesia 🇮🇩 Currently I'm learning data science via an online course with several projects across the journey. This website consists of a compilation of notebooks on data analysis and machine learning algorithms. Will try applying it into upcoming personal projects, preferably healthcare-adjacent ones I'm interested in.

## Deep Learning

### Detecting Pneumonia in X-Ray Images 🫁🩻
([GitHub](https://github.com/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/detecting_pneumonia_in_xray_images.ipynb))

This project detect patients with pneumonia through [X-Ray Images](https://dsserver-prod-resources-1.s3.amazonaws.com/cnn/xray_dataset.tar.gz
) by utilizing Deep Learning, specifically CNN and Transfer Learning. 

## Machine Learning

### Predicting Hepatitis C 🧪🩺🌟
([nbviewer](https://github.com/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/classifying_hepatitis_c.ipynb))

This project predicts patients of Hepatitis C based on [dataset](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset) of laboratory blood test using several models machine learning. 9 model experiments result in `GradientBoosting()` being the model with the highest performance, with `Accuracy` of `97.2973%` and `F1-Score` of `90.1961%`.

### Predicting Heart Disease 🔮🫀
([nbviewer](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/predicting_heart_disease.ipynb))

This project predicts a patient of getting a heart disease from several features by using k-Nearest Neighbors (k-NN) or [`KNeighborsClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). The [dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) includes relevant information for each patient, from their personal information up to relevant medical data.

### Classifying Heart Disease 📂🫀
([nbviewer](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/classifying_heart_disease.ipynb))

This project classifies a patient of heart disease through several features by using [`LogisticRegression()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The [dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) is from the UCI Machine Learning Repository which includes several medical characteristics on each patient, including resting blood pressure, fasting blood sugar, up to ST depression induced by exercise and number of major vessels colored by spectroscopy.

### Predicting Employee Productivity 🧑‍🏭💹
([nbviewer](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/predicting_employee_productivity.ipynb))

This project predicts productivity of garment factory employees using a model of [`DecisionTreeClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [`RandomForestClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to get the strongest predictors. The [dataset](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees) is from the UCI Machine Learning Repository which includes several aspects from day, team number, up to standard minute value for a task and incentive.

### Optimizing Model Prediction ⚙️📈
([nbviewer](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/optimizing_model_prediction.ipynb))

This project compares several models on `LinearRegression()` that include: [`SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) of Forward Selection and Backward Selection, [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV), and [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html). The [dataset](https://archive.ics.uci.edu/dataset/162/forest+fires) used consists of feature that results in `area` of damage in a forest. We're using `wind` (Wind speed) and `temp` (Temperature) as columns for reference model, all column the numerical values for regularized models of `RidgeCV` and `LassoCV`, and we'll cherrypick the best 2-6 features on forward and backward selection accordingly.
