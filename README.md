# 🧑‍💻Data Science Portfolio by Mohamad Alamsyah

Hello! 👋 I'm Alam, a biomedical engineering fresh graduate from Indonesia 🇮🇩 Currently I'm learning data science via an online course with several projects along the journey. This website consists of a compilation of projects I've done on data science in various topics from healthcare to marketing. Hope you enjoy!

### Dashboard of Pharma Sales -- Kimia Farma, Indonesia
🔗[`Looker`](https://lookerstudio.google.com/reporting/a70bc6c6-28ef-47ce-a3d6-f1eea9faea19) 🏷️`dashboarding` `pharma` `looker`

This project turns dataset of Kimia Farma, a pharma retail in Indonesia, into a map-based dashboard with metrics conveying ratings, branches, and product. The data is given from the company with its data anonymized to protect the tue nature of the business. The dashboard can be filtered by island, province, and city for further details.

### SportsLogs: Sports Logging Web App Dashboard
🔗[`Github`](https://github.com/itsalamhere/sports-logs-streamlit/tree/master) [`Streamlit`](https://sports-logs-app-iwtkcdcfze2kebejv55swz.streamlit.app/) 🏷️`data-visualization` `dashboarding` `web-development` `sport` `Python` `Streamlit`  

This project transforms 10+ different datasets from biometric sensor (FitBit), sports logging (PMSys), and Google Docs into a list of dashboards visualizing the performance of 16 players. The web app is available to visualize **the Whole Team** (with limited data) and **individual player** (p01 - p16). The visuals varies from calories burned, active metrics in one player's sport activities, sleep stages on a specific date, up to wellness score each week.

### Alam's Productivity Tracker
🔗[`Looker`](https://lookerstudio.google.com/reporting/a13a8300-6ec0-4c8e-a750-ca3b258f59fd/page/bJDYE) 🏷️`dashboarding` `personal-project` `looker`

This project utilizes a personal archive of daily Pomodoro sessions, task and language chunks, read books, and milestones for 450+ days from the website [Studystream](app.studystream.live). The goal is to visualize how much Alam has gained across his journey--at times he doesn't feel good about it. Glad it turned out neat, and I can add the data further as time goes.

### YouTube Metrics Web App Dashboard
🔗[`Github`](https://github.com/itsalamhere/YT-dashboard-streamlit) [`Streamlit`](https://yt-dashboard-app-k5smefz5gaj39bweayxre2.streamlit.app/) 🏷️`data-visualization` `dashboarding` `web-development` `marketing` `business` `YouTube` `Python` `Streamlit`  

This project transforms datasets of aggregated metrics by 200+ videos and country and subscriber status, along with comments and performance over time, into a web app dashboard with Python and Streamlit with two features: **Aggregate Metrics** for the whole channel, and **Individual Video Analysis** for deeper insights per video.

### Research Payments of Non-Covered Recipient Entity (NCRE) in the United States 2023
🔗[`Github`](https://github.com/itsalamhere/research-payments-ncre-2023/blob/main/README.md) | `nbviewer`: [`01-data-cleaning`](https://nbviewer.org/github/itsalamhere/research-payments-ncre-2023/blob/main/01-data-cleaning.ipynb) [`02-data-analysis-bigquery`](https://nbviewer.org/github/itsalamhere/research-payments-ncre-2023/blob/main/02-data-analysis-bigquery.ipynb) | `Tableau`: [`03-research-payments-NCRE-2023`](https://public.tableau.com/app/profile/mohamad.alamsyah/viz/research-payments-NCRE-2023/ResearchPayments2023-NCREs) 🏷️`healthcare` `RnD` `CMS` `real-world-data` `BigQuery` `Python` `Tableau`

This project visualizes research payments of NCREs in form of a dashboard of Tableau connected with database of BigQuery. The dataset are taken from [CMS Open Payments](https://openpaymentsdata.cms.gov/), then cleaned from `ParsingError` in Python, and analyzed through SQLs of BigQuery. The dashboard showcases NCRE records by state, and also research payments by therapeutic area, product, and NCREs. It also comes along with parameters of Smart Search, Product Category, and Top N [5-20].

### Product Segmentation and Customer Classification in an Online Retail Company
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/product-segmentation-and-customer-classification.ipynb) 🏷️`clustering` `k-means` `wordcloud` `pca` `machine-learning` `marketing-data-science`

This project cleans 1M+ data of 2-year-transactions in a [UK online retail company](https://archive.ics.uci.edu/dataset/502/online+retail+ii) into 760K+ invoices with respective cancellations cleaned into a column, clusters the data into 6 categories of products and 11 categories of customers, and classify the test customers with a combination of 3 best models inside a `VotingClassifier` with an accuracy of `91.5696%`.

### A/B Testing: Commitment Check for Online Students in Udacity ✏️📈🔎
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/ab-testing-commitment-check-udacity.ipynb) 🏷️`a/b-testing` `data-analysis` `Python`

Udacity conducts an experiment of commitment check by adding hours to devote per week before enrolling. This is to filter frustrated students who can't make it due to time while retaining current profits. This project creates the experiment design, measures the changes through relevant metrics, and offers recommendation whether or not to launch the experiment. 

### Drug Reviews on [drugs.com](drugs.com): Dashboard and Sentiment Prediction 💊📊⚙️
🔗 [`Github`](https://github.com/itsalamhere/drug-reviews) | [`Tableau (updated with BigQuery)`](https://public.tableau.com/app/profile/mohamad.alamsyah/viz/DrugReviews-bigquery/drug-reviews-bigquery) | `nbviewer`: 
[`01-EDA-and-data-cleaning`](https://nbviewer.org/github/itsalamhere/drug-reviews/blob/main/01-EDA-and-data-cleaning.ipynb) [`02-import-csv-to-postgresql`](https://nbviewer.org/github/itsalamhere/drug-reviews/blob/main/02-import-csv-to-postgresql.ipynb) [`03-dashboard-of-drug-reviews`](https://nbviewer.org/github/itsalamhere/drug-reviews/blob/main/03-dashboard-of-drug-reviews.ipynb) [`04-sentiment-prediction`](https://nbviewer.org/github/itsalamhere/drug-reviews/blob/main/04-sentiment-prediction.ipynb) 🏷️`big-data` `data-cleaning` `Python` `PostgreSQL` `BigQuery` `Tableau` `natural-language-processing`

This project processes a 16-year [dataset](https://data.mendeley.com/datasets/64cc5w5dxy/1) of drug reviews by 390,000+ consumers into 4 sections: data cleaning and exploration, database management in PostgreSQL, data visualization in Tableau, and sentiment prediction through deep learning and transformers. 

### Revenue Prediction via Customer Lifetime Values (CLV)
🔗[`Github`](https://github.com/itsalamhere/revenue-prediction/tree/main) | `nbviewer`: [`01-EDA-and-data-cleaning`](https://nbviewer.org/github/itsalamhere/revenue-prediction/blob/main/01-EDA-and-data-cleaning.ipynb) [`02-data-visualization`](https://nbviewer.org/github/itsalamhere/revenue-prediction/blob/main/02-data-visualization.ipynb) [`03-feature-engineering`](https://nbviewer.org/github/itsalamhere/revenue-prediction/blob/main/03-feature-engineering.ipynb) [`04-revenue-prediction`](https://nbviewer.org/github/itsalamhere/revenue-prediction/blob/main/04-revenue-prediction.ipynb) 🏷️`big-data` `time-series-data` `forecasting` `machine-learning` 

This project predicts revenue via customer lifetime values such as RFM, total quantity, and time-related variables on a dataset of [Online Retail](https://archive.ics.uci.edu/dataset/502/online+retail+ii) in UK. The variables are extracted into 8 periods (each new period is made for every 2 months): 6-month window of data for features and the next 2-months of total revenue per customer as labels.

### COVID-19 Spread in the United States 🗺️
🔗[`Tableau Public`](https://public.tableau.com/app/profile/mohamad.alamsyah/viz/COVID-19SpreadMapintheUnitedStates/Dashboard1) 🏷️`data-analyst` `data-visualization` `dashboard`

This project visualizes the spread of COVID-19 in the US daily in 2020-2023 from a dataset shared by [Opportunity Insights Economic Tracker](https://github.com/OpportunityInsights/EconomicTracker), with data sourced from New York Times (NYT), Centers for Disease Control and Prevention (CDC), and John Hopkins University (JHU). The dashboard consists of:

* top N (adjustable, 1-10) states with highest new cases of COVID-19 on a chosen day, 
* options to look for a specific state by clicking on the state map,
* recent and current statistics of death, vaccination, and COVID-19 cases.

### Analyzing COVID RNA Sequences 🧬🧪
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/analyzing-covid-rna-sequences.ipynb) 🏷️`biopython` `bioinformatics`

This project looks at COVID RNA sequences consists of reference first sample in Wuhan, China, Asia and also the first sample in North America, and two of emerging mutant variants on the period of time: Delta and Omicron. The project shows how to download and parse the target sequences, align them to find similarities towards the reference, and then applying alignments to find mismatches in the base sequences.

### Predicting Hepatitis C 🧪🩺🌟
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/classifying_hepatitis_c.ipynb) 🏷️`machine-learning` `healthcare-data`

This project predicts patients of Hepatitis C based on a [dataset](https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset) of laboratory blood test using nine models of machine learning: [`RandomForest()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), Support Vector Machine [`SVC()`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), up to [`GradientBoosting()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) and [`VotingClassifier()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier) 

### Predicting MVPs in NBA Seasons 🏀🏆🌟
🔗[`GitHub`](https://github.com/itsalamhere/predicting-mvp-in-nba-seasons/tree/main) | `nbviewer`: [`1-web_scraping`](https://nbviewer.org/github/itsalamhere/predicting-mvp-in-nba-seasons/blob/main/1-web_scraping.ipynb) [`2-data_cleaning`](https://nbviewer.org/github/itsalamhere/predicting-mvp-in-nba-seasons/blob/main/2-data_cleaning.ipynb) [`3-mvp_predictions`](https://nbviewer.org/github/itsalamhere/predicting-mvp-in-nba-seasons/blob/main/3-mvp_predictions.ipynb) 🏷️`web-scraping` `machine-learning` `real-world-data`

This project predicts MVP in NBA seasons in 2020 to 2023 through three steps: collecting the data via web scraping in the [website](www.basketball-reference.com), data cleaning for training model purposes, and MVP predictions using three models of machine learning: [`Ridge()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [`RandomForest()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), and [`GradientBoosting()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html). 

### Detecting Pneumonia in X-Ray Images 🫁🩻
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/detecting_pneumonia_in_xray_images.ipynb) 🏷️`deep-learning` `transfer-learning` `biomedical-image-processing` 

This project detect patients with pneumonia through a dataset of patients' [X-Ray images](https://dsserver-prod-resources-1.s3.amazonaws.com/cnn/xray_dataset.tar.gz
) by utilizing Deep Learning, specifically convolutional neural network `CNN` and Transfer Learning of `ResNet50V2`. 

### Classifying Real and Fake Disaster Tweets 🔎📲
🔗[`GitHub`](https://github.com/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/classifying_disaster_related_tweets.ipynb) 🏷️`deep-learning` `natural-language-processing`

This project classifies real and fake disaster tweets through natural language processing. Besides word visualization on both kinds, the models used include shallow and deep neural networks, and also transformer model of [`distilcase-bert-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased). 

### Time-Series Trade Forecasting ⌚📈
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/time-series_trade_forecasting.ipynb) 🏷️`deep-learning` `sequence-models`

This project forecasts future trades of stocks in [Yahoo! Stock Price](https://www.kaggle.com/datasets/arashnic/time-series-forecasting-with-yahoo-stock-price) using deep learning models: Recurrent Neural Network `RNN()`, Long-Short Term Memory `LSTM()`, convolutional layer `Conv1D()`, and also hyerparameter tuning to optimize the model's accuracy in forecasting. 

### Predicting Listing Gains of the Indian IPO Market 💰🪽
🔗[`nbviewer`](https://nbviewer.org/github/itsalamhere/itsalamhere.github.io/blob/main/Notebooks/predicting_listing_gains_in_indian_ipo_market.ipynb) 🏷️`deep-learning` `neural-network`

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
