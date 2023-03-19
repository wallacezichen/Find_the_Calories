# Find_the_Calories!!

## Can we predict Calories without primary nutritions(fat, carb, and protein)?

## Framing the Problem

Wally already knows that "Calories" are heavily based on primary nutritions such as fat, protein, and carbohydrates. So, when he chooses which food to eat, he just looks at how much fat, protein, and carbohydrates that the food contains. Wally thinks that nutrition other than fat, protein, and carbohydrates are not important. However, Hyunsoo does not agree with Wally's idea. He claims that other nutritions are as important as primary nutritions. Wally claims that fat, protein, and carbohydrates can be used as predicting calories, but other nutritions cannot be used. Hyunsoo claims that calories still can be predicted with without primary nutritions(fat, carb, and protein).

In addition, Hyunsoo claims that higher calories would get higher rating. He thinks that calories are the great indication of taste. Wally is still not sure about that claim. So, they decided to include ratings in their prediction model as well.

With the dataset given, they observed that there are `rating`, `s_fat`, `sugar`, and `sodium` information that they can use. Also, each food has multiple ratings. So, to see if calories can be predicted on ratings.

#### Quesiton: Can we predict Calories without primary nutritions(fat, carb, and protein)?

#### Type: Regression

###### Response Variable: `calories`

`calories` data is selected as a response variable, because Wally and Hyunsoo want to know if it can be predicted based on nutrition information other than primary nutritions(fat, carb, and protein) and ratings.

###### Evaluation Metric: **Root Mean Squared Error**

Root Mean Squared Error is used over other suitable metrics, because the model is a regression model. Using metrics that measure the difference between the predicted and actual values of the target variable would be more appropriate. Accuracy metric would be useful when we use classfication models, however we are predicting continuous numeric values.

##### Brief Information of the dataset used.
Our exploratory data analysis on this dataset can be found here:https://lionjhs98.github.io/exclamation-mark-shows-satisfaction/

Number of **Rows** in the Merged dataset of `Recipes` and `Interactions` is
**234429**

Columns Explanation for Data Prediction.

`calories` : The column shows the kilocalories for each recipe. 

`s_fat` : The column shows the amount of s_fat in grams for each recipe.

`sugar` : The column shows the amount of sugar in grams for each recipe.

`sodium` : The column shows the amount of sodium in grams for each recipe.

`rating` : The column which contains people's opinion of the recipe in numeric values from 1 to 5. We analyze the people's judgement on the recipes in numeric value which would make the comparison between recipes easier. Sometimes it is hard to judge by just looking at the recipes and description.

Above five columns are the data that we use to test the question we have.
We want to see if `calories` can be predicted by `sodium`, `sugar`, and `s_fat`columns. Also, we need corresponding `rating` in average for each recipes for a prediction.

---

## Cleaning and EDA

### Data Cleaning

The datasets that Zichen and Hyunsoo are using unfortunately have issues. People say the more the betterm but in this case, they are seeing too many information and cannot decide which data to use to predict Calories. In order to make a prediction. Also, some data have problems such as having an outlier, wrong data , etc. They had to look closely on each column and check if there is any problem before they use.

Below are the steps that they have taken to get their data set ready for analyzation.

#### Merging `interactions` and `recipes`

Zichen and Hyunsoo realized that `Interactions` dataset contains information for the individual recipes in `recipes` dataset. They wanted to look at all the information at the same time, so they decided to merge two data. Both data had common data in columnd `id` and `recipe_id` which indicated id for the individual recipes. So they decided to merge the datasets on common `id` column. They did not want to lose all the reviews that `interactions` dataset contains in the process of merging, so they generated more rows based on the number of rows in `interactions` dataset.

    merged_df = pd.merge(recipes,interactions,left_on = 'id',
                     right_on = 'recipe_id',how = 'left')


#### Because there is "\n" in the `review` and `description`, the dataframe in the markdown is ruined, so we need to get rid of them.

Hyunsoo finds that so merged_df only have review for one whole row, which is very odd. Hyunsoo look at the dataframe closely and finds there is "\n" and they decide to get rid of them with str.replace method.

    merged_df['review'] = merged_df['review'].str.replace("\n","")
    merged_df['description'] = merged_df['description'].str.replace("\n","")

Here is how `merged_df` look like:

| name                               |     id | review                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | description                                                                                                                                                                                                                                                                                                                                                                      |
| :--------------------------------- | -----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 brownies in the world best ever  | 333281 | These were pretty good, but took forever to bake. I would send it ended up being almost an hour! Even then, the brownies stuck to the foil, and were on the overly moist side and not easy to cut. They did taste quite rich, though! Made for My 3 Chefs.                                                                                                                                                                                                                      | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                             |
| 1 in canada chocolate chip cookies | 453467 | Originally I was gonna cut the recipe in half (just the 2 of us here), but then we had a park-wide yard sale, & I made the whole batch & used them as enticements for potential buyers ~ what the hey, a free cookie as delicious as these are, definitely works its magic! Will be making these again, for sure! Thanks for posting the recipe!                                                                                                                                | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                           |
| 412 broccoli casserole             | 306168 | This was one of the best broccoli casseroles that I have ever made. I made my own chicken soup for this recipe. I was a bit worried about the tsp of soy sauce but it gave the casserole the best flavor. YUM! The photos you took (shapeweaver) inspired me to make this recipe and it actually does look just like them when it comes out of the oven. Thanks so much for sharing your recipe shapeweaver. It was wonderful! Going into my family's favorite Zaar cookbook :) | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 |
| 412 broccoli casserole             | 306168 | I made this for my son's first birthday party this weekend. Our guests INHALED it! Everyone kept saying how delicious it was. I was I could have gotten to try it.                                                                                                                                                                                                                                                                                                              | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 |
| 412 broccoli casserole             | 306168 | Loved this. Be sure to completely thaw the broccoli. I didn&#039;t and it didn&#039;t get done in time specified. Just cooked it a little longer though and it was perfect. Thanks Chef.                                                                                                                                                                                                                                                                                        | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 |

if you compare the dataframe created in the merging step and getting read of line changer step, you can observe that the columns with just reviews are gone.

#### Change 0 in rating columns to NaN

There is some "unfaithful" data in the `rating` column that present as 0, so Zichen decide to change all of 0 in the rating to NaN. Having 0 in `rating` is "unfaithful", because when the users put ratings, zero is not an option. If the users have put the rating correctly, the rating should be in between 1 to 5. So, having 0 instead does not make sence. So in this step, we replaced the 0 to null value.

    merged_df.loc[merged_df['rating'] == 0, 'rating'] = np.nan

How `merged_df` look like after cleaning:

| name                               |     id | rating |
| :--------------------------------- | -----: | -----: |
| 1 brownies in the world best ever  | 333281 |      4 |
| 1 in canada chocolate chip cookies | 453467 |      5 |
| 412 broccoli casserole             | 306168 |      5 |
| 412 broccoli casserole             | 306168 |      5 |
| 412 broccoli casserole             | 306168 |      5 |

#### Average rating for each recipe

Zichen thinks that average rating is a very useful information to find. So he use the groupby method to find each recipes' average rating and add that data to the `res` dataframe.

    mean_df = pd.DataFrame(merged_df.groupby("id")['rating'].mean())
    res = pd.merge(merged_df, mean_df,left_on = 'id',right_index = True,how = 'inner',)
    res = res.rename(columns = {"rating_y" : "Average Rating", "rating_x" : "rating"})

#### Get rid of outliers from n_steps and minutes columns

When Zichen want to check what recipe takes the longest time to make, he surprisingly find out that it takes 1051200 minutes to make. Zichen thinks that is definitely an outlier for the later test, so he decides to remove some outlier like these in the minutes and n_steps. He set the threshold to only include the recipes which takes less than or equal to 30 steps to make and takes minutes which is less than or equal to 800 minutes.

    res = res[res['minutes'] < 800]
    res = res[res['n_steps'] <= 30]

This is how res look like after cleaning:

| name                               |     id | minutes | n_steps |
| :--------------------------------- | -----: | ------: | ------: |
| 1 brownies in the world best ever  | 333281 |      40 |      10 |
| 1 in canada chocolate chip cookies | 453467 |      45 |      12 |
| 412 broccoli casserole             | 306168 |      40 |       6 |
| 412 broccoli casserole             | 306168 |      40 |       6 |
| 412 broccoli casserole             | 306168 |      40 |       6 |

#### Clean the nutrition column into seperate columns, such as "[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), and carbohydrates (PDV)]"

Hyunsoo finds that the `nutrition` which represent calories, total fat, sugar, sodium, protein, saturated fat, and carbohydrates, was express as object type in the `res` so he decide to split nutrition into 7 different new columns and include these columns in the `res`.

    cur = res['nutrition'].str[1:-1].str.split(',')
    res['calories'] = cur.str[0].astype(float)
    res['fat'] = cur.str[1].astype(float)
    res['sugar'] = cur.str[2].astype(float)
    res['sodium'] = cur.str[3].astype(float)
    res['protein'] = cur.str[4].astype(float)
    res['s_fat'] = cur.str[5].astype(float)
    res['carb'] = cur.str[6].astype(float)
    res['rating'] = merged_df['rating'].astype(float)
    res = res.drop(columns = 'nutrition')

Here is how `res` look like after cleaning:

| name                               |     id | calories | fat | sugar | sodium | protein | s_fat | carb | rating |
| :--------------------------------- | -----: | -------: | --: | ----: | -----: | ------: | ----: | ---: | -----: |
| 1 brownies in the world best ever  | 333281 |    138.4 |  10 |    50 |      3 |       3 |    19 |    6 |      4 |
| 1 in canada chocolate chip cookies | 453467 |    595.1 |  46 |   211 |     22 |      13 |    51 |   26 |      5 |
| 412 broccoli casserole             | 306168 |    194.8 |  20 |     6 |     32 |      22 |    36 |    3 |      5 |
| 412 broccoli casserole             | 306168 |    194.8 |  20 |     6 |     32 |      22 |    36 |    3 |      5 |
| 412 broccoli casserole             | 306168 |    194.8 |  20 |     6 |     32 |      22 |    36 |

#### Change empty list in tags column to NaN

Hyunsoo is a careful student, he finds out that there are also unfaithful data appears in the tags, which represent as empty list, so he decide to replace them as NAN.

    res['tags'].replace("['']",np.nan,inplace = True)

Here is how dataset looks like after cleaning:

| name                               |     id | tags                                                                                                                                                                                                                        |
| :--------------------------------- | -----: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 brownies in the world best ever  | 333281 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] |
| 1 in canada chocolate chip cookies | 453467 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               |
| 412 broccoli casserole             | 306168 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |
| 412 broccoli casserole             | 306168 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |
| 412 broccoli casserole             | 306168 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |
###### Codes and Comments in Cleaning and EDA are copied from Zichen and Hyunsoo's project3

##### Now we are done with cleaning the data!! Let's predict the Calories!!

---

## Baseline Model

#### Define Root Mean Squared Error function

    def RMSE(actual, pred):
        return np.sqrt(np.mean((actual - pred) ** 2))

We mainly used RMSE as an evaulation metric through out testing the models.
This RMSE function is used in both Baseline and Final models.

Below are the scatter plots which show the relationship in between response variable `calories` and `sugar`, `s_fat`, `sodium`, and `rating`.

#### Scatter plot for `sugar` and `calories`
<iframe src="asset/fig1.html" width=600 height=400 frameBorder=0></iframe>

#### Scatter plot for `s_fat` and `calories`
<iframe src="asset/fig2.html" width=600 height=400 frameBorder=0></iframe>

#### Scatter plot for `sodium` and `calories`
<iframe src="asset/fig3.html" width=600 height=400 frameBorder=0></iframe>

#### Scatter plot for `rating` and `calories`
<iframe src="asset/fig4.html" width=600 height=400 frameBorder=0></iframe>

##### From the above four graphs, we did not see any trend between four features and calories, so we could not use log or square root to transform the data. Also, they are continuing numerics values, so we decided not to change these numeric columns in our baseline model. 

#### Model Description

Based on our analysis with the data based on the graphs above, we decided to used column datas as what it is already, so we created a preprocessor which contain the original column data without any modification. Used features are `sugar`, `s_fat`, `sodium`, and `rating`.

    preproc = ColumnTransformer(
    transformers=[
        ('FT1',FunctionTransformer(lambda x: x),list(baseline_X.columns)),
    ],remainder='passthrough')

All **four features** are **quantitative** values, so there was no necessary encoding required.

For the pipeline for our model, we have sets continuing numeric values. So, we used **Linear Regression** model along with preprocessor we created above. 

#### Performance Report

Our baseline model's accuracy score is evaluated by Root Mean Square Error.

##### Baseline model Checking Overfitting & Underfitting

We wanted to check if there is an overfitting and underfitting issues with the baseline model.

So, we splited our dataset into training and testing.

    baseline_X_train, baseline_X_test, baseline_y_train, baseline_y_test = train_test_split(baseline_X, baseline_y, test_size=0.2, random_state=1)

The results are followings:
    **train set RMSE score : 244.85**
    **test set RMSE score : 261.55**

The RMSE scores are high, but there is no extreme difference in between the train set's RMSE and the test set's RMSE. We concluded it does not seem like our model is overfitting to the train data. Our model is able to generalize well.

##### Baseline Model Analysis

**RMSE Score: 245.98**

RMSE score close to zero indicates that the model is accurate in predicting values. However, our baseline model shows **245.98** as a RMSE score which is **not good** RMSE score. It is critically higher than 0 which is an ideal score. So, we concluded that we need to improve the model and input data we have.

---

## Final Model

### Exproring the Model

#### Successful Exploration

##### 1. QuantileTransformer

- We applied `QuantileTransformer()` on the data `sodium` and `sugar`.
The reason why we applied the quantile transformer is that, for some recipes such as "christmas ham" they use about 4,000 grams of sodium. For both, data sets `sodium` and `sugar` there are outliers which recipes contain amount over 1000 grams. Those could be the outliers which can harm our regression. But, those are still reasonable data, we should not delete them. So, in order to lower the impact of outliers in our dataset we tansformed `sodium` and `sugar`. 

##### 2. PolynomialFeatures

- When we look at the graph **Scatter plot for `s_fat` and `calories`**, there are alot of non-linear datas along with linear datas. So, we thought that using PloynomialFeatures transfomer for `s_fat` would better fit the model. In order to find the best hyperparameters for PolynomialFeatures transformer, we tried multiple degrees(1 ~ 5). Having degree of **3** improved the model the most.  

##### 3. K-Fold Cross Validation

- We thought that we do not have enough dataset could be a problem, so we decided to add on new features. Multiplying the quantative values could provide improvements in predcition, because some combinations of nutrition could be related to calories. For example, high calories can be predicted when there are large amount of sodium and saturated fat. 10 new features such as 'sugar * rating', 'sugar * sodium', etc were created. 

- However, adding all the combinatorial features could cause issues. Some information could be redundant or irrelevant. So we created combinations of two features and add them on the original dataset we used in baseline model.
For example, first model add the columns added `sugar * rating` and `sugar * sodium` and the second model would contain `sugar * rating`, `sugar * rating` as so on.

- To decide the which combination of features would actually improve our model, we decided to use K-Fold Cross Validation. The reason why we chose K-Fold Cross Validation is that we have observed many overfitting issue when we were exploring various methods to improve the model(refer to "Unsuccessful Exploration" section below). 

- We used 5 folds for cross validation model and got average **RMSE** scores for each data combinations. The model with new features of combinations of `sugar * rating` and `sodium * s_fat` showed the lowest RMSE score with the model. 

#### Final Model Description

The final model is built and improved based on the **Successful Exploration**. 

In the baseline model, we thought transforming the quantative datas are not necessary which resulted in bad RMSE score. In order to improve the model, we used **QuantileTransformer** on `sodium` and `sugar` data to generalize outlier datas. (Details in Successful Exploration: 1. QuantileTransformer)

The `s_fat` data had non-linear relationship with `calories`, so we used **PolynomialFeatures** to make data more fit. We tested multiple degrees to find the best hyperparameter and **Degree 3** returned the lowest RMSE score for the model. 
(Details in Successful Exploration: 2. PolynomialFeatures)

Above two tranformers are included in our preprocessor.
Then, we used **LinearRegression** with the preprocessor created above.

In order to prevent overfitting issues and find the best features to add into model, we used K-Fold Cross Validation. (Details in Successful Exploration: 3. K-Fold Cross Validation)


#### Unsuccessful Exploration

##### Gridsearch & DecisionTreeRegressor

Grid Search Cross Validation was used to improve the model at first. By default GridSearchCV has classifier score as a standard of choosing the best parameter. However we are using regressor instead of classfier, so we customized the score measurement to **RMSE** to figure out the best new features and hyperparameter. Also, we used DecisionTreeRegressor instead of LinearRegression for the GridSearch.

Just like how we added the combination of new features in **3. K-Fold Cross Validation**, we tried the same process with GridSearchCV. We created the combinations of two features each and created the new model for each combination.

The GridSearch Cross Validation has returned that adding a combination of features of **'sugar * s_fat' & 'sodium * rating'** to the model would give the lowest RMSE score. 

However, the model created by the GridSearchCV and DecisionTreeRegressor had a critical issue. In order to check whether the model is overfitting or not, we splited the data to training data and testing data.

Followings are the results we got for each data set:

    Training Data's RMSE : 28.14440356736906
    Testing Data's RMSE  : 130.94614978314289

As we see, there are unignorable difference in RMSE score in between Training Data and Testing Data.

So, we concluded that using Gridsearch & DecisionTreeRegressor to tune our model might overfit to training data. Even though the RMSE score has been noticeably improved, we cannot generalize by using this model due to overfitting.

### Final Model Performance Report

##### Checking Overfitting & Underfitting

Before, we get the RMSE score for the final model(refer to the final model from Successful Exploration), it is necessary to check whether the model has overfitting or underfitting issue.

So, we splited the data just like how we did in the Baseline model to check the issue.

Followings are the results: 

    Training Data's RMSE : 226.730711098181
    Testing Data's RMSE  : 228.29595970522894

The training data's RMSE score and testing data's RMSE score has not critical difference. They are similar, so it does not seem like the model is overfitting to the training data. The model is able to generalize well.

##### Score Improvement 

Final Model Performance RMSE Score

                            227.02


Our baseline model RMSE score was **245.98** and our final model RMSE score is **227.02**. There is a  **18.96** unit improvement in the model. 

We could imprive the final model by using QuantileTransformer, PolynomialFeatures with hyperparameter 3, and K-Fold Cross Validation.

Now we want to check if our model is fair.

---
# Fairness Analysis 🧐

 Wally wonders about the question that Does our model perform worse for recipe groups that contain higher sugar(>= threshold grams) than it does for recipe groups that contain lower sugar(< threshold grams)? So they decide to investgate it. 

## Investigate Question: Does our model perform worse for recipe groups that contain higher sugar(>= threshold grams) than it does for recipe groups that contain lower sugar(< threshold grams)?
---

The first step is to set the threshold of seperating the orginal dataframe into two groups. By looking at the amount of sugar in recipe histogram, Hyunsoo suggest to use 36 grams of sugar as the threshold to seperate the dataframe into two part because according to AHA, a grown adult is not recommanded to intake 36 grams of sugar or more in one day. 

The second step is to Binarize the each row to their corresponding group. We achieve this step by the code below.

    binarizer = Binarizer(threshold=36)
    sugar_df = pd.concat([useful_df,final_pipeline_df[['sugar * rating', 'sodium * s_fat']]],axis = 1)
    sugar_df["sugar_bin"] = binarizer.transform(sugar_df[['sugar']].values)

Then we are ready to perform permutation test to investigate this question. 

## Permutation Test
#### Null Hypothesis:  
Our model is fair. Its precision for the recipe groups that contain high sugar(>= 36 grams) and recipe groups that contain high sugar(< 36 grams) are roughly the **same**, and any differences are due to random chance.
#### Alternative Hypothesis: 
Our model is unfair. Its precision of recipe groups that contain high sugar(>= 36 grams) and the precision of recipe groups that contain high sugar(< 36 grams) is **different**

    Group X : The recipe groups that contain high sugar(>= 36 grams)
    Group Y: The recipe groups that contain low sugar(< 36 grams)
    Evaluation Metric: Absolute Difference of RMSE between Group X and Group Y

#### Test Statistics(Absolute Difference of RMSE)

    118.74243857032002 


#### P-Value
    
    0.0

#### Significance level
    
    0.05

### Conclusion
    P-value(0.0) < 0.05
With the information above, because the pvalue(0) < significance level(0.05), **we rejected** the null hypothesis, indicate that the distribution of recipe groups that contain high sugar(>= 36 grams) and recipe groups that contain high sugar(< 36 grams) are not **same**!

Can we answer the question with the test results?

**Unfortunately Not.**

Even though the test rejected the null hypothesis, we cannot state that the **alternative hypothesis**, Our model is unfair. Its precision of recipe groups that contain high sugar(>= 36 grams) and the precision of recipe groups that contain high sugar(< 36 grams) is **different**, is absolutely correct.