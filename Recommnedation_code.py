from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# Creating the SparkSession
spark = SparkSession.builder.appName('MovieRecommendation').getOrCreate()

# Loading the dataset
ratings = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\\Task\\ml-latest-small\\ratings.csv').cache()
movies = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\\Task\\ml-latest-small\\movies.csv').cache()

# Spliting the ratings dataset into training, validation, and test datasets
(training, validation, test) = ratings.randomSplit([0.6, 0.2, 0.2], seed=42)

# Defining the ALS model
als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', nonnegative=True, coldStartStrategy='drop')

# Defining the parameter grid for tuning
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 50]) \
    .addGrid(als.regParam, [0.1, 0.01]) \
    .build()

# Defining the evaluation metric
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

# Defining the cross validator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# Training the ALS model using the training dataset
best_model = cv.fit(training)

# Using the validation dataset to select the best model
predictions = best_model.transform(validation)
rmse = evaluator.evaluate(predictions)
print('Root-mean-square error =', rmse)

# Using the test dataset to evaluate the performance of the best model
predictions = best_model.transform(test)
rmse = evaluator.evaluate(predictions)
print('Root-mean-square error on test set =', rmse)


# Defining the user ID for which we want to recommend movies
user_id = 1

# Getting the movies the user has rated
user_ratings = ratings.filter(col('userId') == user_id)

print('Movies liked by user:')
user_ratings.filter(col('rating') >= 4).join(movies, 'movieId').select('title').show(truncate=False)
print('Movies disliked by user:')
user_ratings.filter(col('rating') < 4).join(movies, 'movieId').select('title').show(truncate=False)

# Getting the top 10 recommended movies for the user
user_recs = best_model.bestModel.recommendForAllUsers(10).filter(col('userId') == user_id)


# Extracting the recommended movie IDs and exploding it into the separate rows
user_recs = user_recs.selectExpr("explode(recommendations) as rec")
user_recs = user_recs.select('rec.movieId')

# Joining with the movies DataFrame to get the movie titles
user_recs = user_recs.join(movies, 'movieId')

print('Top 10 recommended movies for the user:')
user_recs.select('title').show(truncate=False)
