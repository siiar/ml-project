# DecentML: CMPT 726 Machine Learning Final Project

> [Link to project page](https://www2.cs.sfu.ca/~mori/courses/cmpt726/project.html) | [Online Companion](https://sinablk.github.io/cmpt726/)

Real-world datasets are usually comprised of numerous categorical features which have high-cardinality for one-hot encoding or other common categorical encoding methods to be effective.  In the case of high-cardinality, such techniques create large  sparse  matrices  that  lead  to  curse  of  dimensionality.   Furthermore,  these datasets also contain class imbalances that adversely affect a classification model’s prediction power because the model will be biased towards the majority class and,thus, will not be able to generalize well on new unseen data.  In this project, we compare different permutations of categorical encoding methods, namely Target,LeaveOneOut, and Catboost, with two of the most popular over-sampling methods, namely SMOTE and ADASYN, and evaluate each combination by training a random forest classifier. We choose the movie dataset as our case study because it has both high-cardinal features as well as imbalanced classes.

## Team

- Saboora M. Roshan
- Sina Balkhi
- Max Ou 
- Urvi Chauhan

## Data

- From [TMDB API](https://www.themoviedb.org/documentation/api). Budget and revenue features were scraped from [The Numbers](https://www.the-numbers.com/movie/budgets/all).
- Drop the `production_countries` feature because movies made in US typically make more money than those made in other countries. This can cause problems in our prediction evaluations as our model would simply learn to classify movies made in US vs. other countries.
- **Target var:** `df['target'] = np.where(df.Production_Budget < df.Worldwide_Gross, 1, 0)`. So we define a "success" movie as one where its Worldwide Gross Revenue exceeded its Production Budget, labeling it as $`1`$, and $`0`$ otherwise (flop).

## Models

- Random forest
- Decision trees