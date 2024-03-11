# 🧑‍💻Data Science Portfolio by Mohamad Alamsyah

Hello, I'm Alam, a biomedical engineering fresh graduate from Indonesia 🇮🇩 Currently I'm learning data science via an online course with several projects along the journey. This website consists of a compilation of notebooks I've done on data analysis and machine learning. Let's dive in, shall we?

### Predicting Hepatitis C 🧪🩺🌟
  
🔗[`nbviewer`](https://github.com/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/classifying_hepatitis_c.ipynb) 🏷️`machine-learning` `healthcare-data`

This project predicts patients of Hepatitis C based on a [dataset](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset) of laboratory blood test using nine models of machine learning: [`RandomForest()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), Support Vector Machine [`SVC()`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), up to [`GradientBoosting()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) and [`VotingClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier) 

### Predicting MVPs in NBA Seasons 🏀🏆🌟
🔗[`GitHub`](https://github.com/itsalamhere/predicting-mvp-in-nba-seasons/tree/main), `nbviewer`: [`1-web_scraping`](https://nbviewer.org/github/itsalamhere/predicting-mvp-in-nba-seasons/blob/main/1-web_scraping.ipynb) [`2-data_cleaning`](https://nbviewer.org/github/itsalamhere/predicting-mvp-in-nba-seasons/blob/main/2-data_cleaning.ipynb) [`3-mvp_predictions`](https://nbviewer.org/github/itsalamhere/predicting-mvp-in-nba-seasons/blob/main/3-mvp_predictions.ipynb) 🏷️`web-scraping` `machine-learning` `real-world-data`

This project predicts MVP in NBA seasons in 2020 to 2023 through three steps: collecting the data via web scraping in the [website](www.basketball-reference.com), data cleaning for training model purposes, and MVP predictions using three models of machine learning: [`Ridge()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [`RandomForest()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), and [`GradientBoosting()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html). 

### Detecting Pneumonia in X-Ray Images 🫁🩻
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/detecting_pneumonia_in_xray_images.ipynb) 🏷️`deep-learning` `CNN` `transfer-learning` `biomedical-image-processing` 

This project detect patients with pneumonia through a dataset of patients' [X-Ray images](https://dsserver-prod-resources-1.s3.amazonaws.com/cnn/xray_dataset.tar.gz
) by utilizing Deep Learning, specifically `CNN` and Transfer Learning of ResNet50V2. 

### Classifying Fake Disaster Tweets
🔗[`nbviewer`](https://github.com/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/classifying_disaster_related_tweets.ipynb) 🏷️`deep-learning` `natural-language-processing`

### Time-Series Trade Forecasting
🔗[`nbviewer`](https://github.com/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/time-series_trade_forecasting.ipynb) 🏷️`deep-learning` `sequence-models`

### Predicting Listing Gains of the Indian IPO Market
🔗[`GitHub`](https://github.com/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/predicting_listing_gains_in_indian_ipo_market.ipynb) 🏷️`deep-learning` `neural-network`

This project predicts companies that has listing gains across the Indian IPO market from a dataset of [moneycontrol](https://www.moneycontrol.com/ipo/listed-ipos/?classic=true) consisting of company names with its issues size and price and subscriptions from several entities. It begins from data exploration and visualization, treatment of outliers, and defining the classification model with deep neural network.

### Classifying Heart Disease 📂🫀
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/classifying_heart_disease.ipynb) 🏷️`machine-learning` `logistic-regression`

This project classifies a patient of heart disease through several features by using [`LogisticRegression()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The [dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) is from the UCI Machine Learning Repository which includes several medical characteristics on each patient, including resting blood pressure, fasting blood sugar, up to ST depression induced by exercise and number of major vessels colored by spectroscopy.

### Optimizing Model Prediction ⚙️📈
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/optimizing_model_prediction.ipynb) 🏷️ `machine-learning` `forward-selection` `backward-selection`

This project compares several models on `LinearRegression()` that include: [`SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html) of Forward Selection and Backward Selection, [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV), and [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html). The [dataset](https://archive.ics.uci.edu/dataset/162/forest+fires) used consists of feature that results in `area` of damage in a forest. We're using `wind` (Wind speed) and `temp` (Temperature) as columns for reference model, all column the numerical values for regularized models of `RidgeCV` and `LassoCV`, and we'll cherrypick the best 2-6 features on forward and backward selection accordingly.

### Predicting Employee Productivity 🧑‍🏭💹
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/predicting_employee_productivity.ipynb) 🏷️`machine-learning` `random-forest`

This project predicts productivity of garment factory employees using a model of [`DecisionTreeClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [`RandomForestClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to get the strongest predictors. The [dataset](https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees) is from the UCI Machine Learning Repository which includes several aspects from day, team number, up to standard minute value for a task and incentive. This could give insights to managers regarding aspects that can influence actual productivity compared to the target, from time spent for a garment, up to incentives and over-working time.

### Predicting Heart Disease 🔮🫀
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/predicting_heart_disease.ipynb) 🏷️`machine-learning` `k-nearest-neighbors`

This project predicts a patient of getting a heart disease from several features by using k-Nearest Neighbors (k-NN) or [`KNeighborsClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html). The [dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) includes relevant information for each patient, from their personal information up to relevant medical data.
