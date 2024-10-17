# Netflix Movie Analysis & Recommendation System

This is a project on developing a recommendation system based on Netflix's historical data to enhance customer acquisition strategies and improve user retention by providing personalized movie suggestions

## Overview
This repository contains:
- [Jupyter Notebook](#jupyter-notebook): A complete guide through data preprocessing, EDA, model implementation, and evaluation.
- [Data & Preprocessing](#data--preprocessing): Details on the datasets used and how they were processed.
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda): Insights into user and movie behavior.
- [Models](#models): Description of the machine learning models and their performance.
- [Python Libraries](#python-libraries): A list of libraries used in this project.
- [Results](#results): Best-performing models and key findings.
- [Future Work](#future-work): Potential improvements to enhance the recommendation system.
- [Contributors](#contributors) and [Acknowledgments](#acknowledgments).

## Jupyter Notebook

The Jupyter Notebook walks through:
- **Data Preprocessing**: Cleaning, transforming, and merging datasets.
- **EDA**: Visual and statistical analysis of user and movie behavior.
- **Modeling**: Implementing machine learning algorithms (SVD, KNN, ALS, etc.).
- **Evaluation**: Comparing models based on RMSE.
- **Conclusions**: Summarizing the best model and insights.

## Data & Preprocessing

We used two main datasets sourced from **Kaggle**:

| Dataset               | Description                                                                           |
|-----------------------|---------------------------------------------------------------------------------------|
| `combined_data_1`  | User ratings for movies, including user IDs, movie IDs, ratings (1-5), and rating dates.|
| `movie_titles`     | Movie metadata, including movie IDs, titles, and release years.                       |

### Preprocessing Steps:
- **Data Merging**: Combined ratings and movie metadata to create a unified dataset.
- **Cleaning**: Addressed missing data, including movie release years (using web scraping for missing values).
- **Downsampling**: Reduced dataset size for computational efficiency by removing unnecessary rows.
- **Date Formatting**: Converted date fields to the DateTime format.
- **Splitting**: The data was chronologically split into training (80%), validation (10%), and testing (10%) sets to maintain temporal integrity.

## Exploratory Data Analysis (EDA)

### User Insights:
- **Top 10 Active Users**: Showed significant variability in their ratings. Some consistently rated highly, while others showed diverse rating behaviors, highlighting different viewing habits.
- **Rating Distribution**: The majority of ratings were between 3 and 4, indicating generally positive feedback, though a few users provided extreme ratings.

### Movie Insights:
- **Movie Trends (1999-2005)**: Viewer preferences evolved over time, shifting from love-themed movies in 1999 to vampire-themed movies by 2002, as seen in word cloud visualizations.
- **Top Reviewed vs. Top Rated Movies**: The most reviewed movies weren’t always the highest-rated, indicating that popularity and quality don’t always align.
- **Rating Correlation**: There was little correlation between the number of ratings a movie received and its average rating, suggesting that niche movies often had more favorable ratings.

## Models

We tested four models to build the recommendation system. Below is a summary of each model and its performance:

| Model        | Algorithm                              | RMSE    |
|--------------|----------------------------------------|---------|
| **Model I**  | SVD, KNN                               | 0.9876  |
| **Model II** | SVD++, TF-IDF                          | 1.3041  |
| **Model III**| SVD, NMF                               | 0.9866  |
| **Model IV** | ALS, KNN                               | 1.0902  |

### Model Descriptions:
1. **Model I**: Combines matrix factorization (SVD) with item-based similarity (KNN). It performed well with an RMSE of 0.9876 but was slightly outperformed by Model III.
   
2. **Model II**: A hybrid of collaborative filtering (SVD++) and content-based filtering (TF-IDF). This model struggled with complexity and had the highest RMSE of 1.3041.

3. **Model III**: The best-performing model. It integrates SVD and NMF to capture latent factors in user preferences, achieving an RMSE of 0.9866.

4. **Model IV**: Uses ALS for matrix factorization, combined with KNN for similarity-based recommendations. It performed reasonably well with an RMSE of 1.0902.


## Python Libraries

### Recommender System Libraries

- [Surprise](http://surpriselib.com) - Scikit-learn type API for matrix factorization and other traditional recommendation algorithms.
- [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) - Distributed data processing and the Alternating Least Squares (ALS) model for collaborative filtering.

### Matrix Factorization Based Libraries

- [Surprise](http://surpriselib.com) - Provides SVD, NMF, and KNN algorithms for recommendation systems.
- [PySpark ALS](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html) - Implements matrix factorization using ALS for large-scale recommendations.

### Content-Based Libraries

- [scikit-learn](https://scikit-learn.org) - TF-IDF and cosine similarity for content-based filtering and text vectorization.

### Web Scraping Libraries

- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - Web scraping tool used for retrieving movie metadata (e.g., release year).
- [requests](https://docs.python-requests.org) - HTTP library for sending requests during the web scraping process.

### Data Visualization Libraries

- [matplotlib](https://matplotlib.org) - Visualizes exploratory data analysis (EDA) results and model performance.
- [seaborn](https://seaborn.pydata.org) - Enhances visualizations with statistical data plots.
- [wordcloud](https://github.com/amueller/word_cloud) - Generates word clouds to explore movie title trends.

### Similarity Search Libraries

- [scikit-learn](https://scikit-learn.org) - Implements similarity measures like cosine similarity for content-based filtering.


## Results

**Model III (SVD/ NMF)** performed best with an RMSE of **0.9866**, indicating it effectively captures user preferences for personalized recommendations.

## Future Work

1. **Hyperparameter Tuning**: Optimize SVD, ALS, and KNN models.
2. **Feature Engineering**: Add user demographics and movie genres.
3. **Cross-Validation**: Use time-based cross-validation for better performance evaluation.
4. **Scaling**: Incorporate a larger dataset to capture more diverse user behaviors.

## Contributors

- **Jui-Jia Chen**
- **Chi-Ning Liu**
- **Shuo Chen**
- **Pamela Hsieh**

## Acknowledgments

Supervised by **Professor Yi Zhang** as part of Columbia University's Business Analytics course.

## Reference

**Data Source**:  
Netflix, Chris Crawford. Netflix Prize data, [Netflix Prize Data on Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data)
