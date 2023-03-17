from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from movierecommendation.config import cache

@require_GET
def recommend_movies(request, user_id='1'):
    # Create a SparkSession
   # spark = SparkSession.builder.appName('MovieRecommendation').getOrCreate()
    spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.executor.instances", "1") \
    .getOrCreate()

    # Loading the dataset
    ratings = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\\Task\\ml-latest-small\\ratings.csv').cache()
    movies = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\\Task\\ml-latest-small\\movies.csv').cache()

    # Load the saved model
    saved_model = ALSModel.load('C:\\Users\\Sachin\\Downloads\\ALS_model\\content\\ALS_model')

    # Get the movies the user has rated
    user_ratings = ratings.filter(col('userId') == user_id)

    # Get the top 10 recommended movies for the user
    user_recs = saved_model.bestModel.recommendForAllUsers(10).filter(col('userId') == user_id)

    # Extract the recommended movie IDs and explode them into separate rows
    user_recs = user_recs.selectExpr("explode(recommendations) as rec")
    user_recs = user_recs.select('rec.movieId')

    # Join with the movies DataFrame to get the movie titles
    user_recs = user_recs.join(movies, 'movieId')

    # Serialize the result as JSON and return it
    result = {
        'user_id': user_id,
        'movies_liked': list(user_ratings.filter(col('rating') >= 4).join(movies, 'movieId').select('title').toPandas()['title']),
        'movies_disliked': list(user_ratings.filter(col('rating') < 4).join(movies, 'movieId').select('title').toPandas()['title']),
        'recommended_movies': list(user_recs.select('title').toPandas()['title'])
    }
    return JsonResponse(result)


@require_GET
def recommend_movies_with_training(request, user_id='1'):
    global cache
    # Create a SparkSession
    spark = SparkSession.builder.appName('MovieRecommendation').getOrCreate()

    # Loading the dataset
    ratings = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\\Task\\ml-latest-small\\ratings.csv').cache()
    movies = spark.read.format('csv').options(header='true', inferSchema='true').load('D:\\Task\\ml-latest-small\\movies.csv').cache()

    if not cache:
        # Load the saved model
        # saved_model = ALSModel.load('C:\\Users\\Sachin\\Downloads\\ALS_model\\content\\ALS_model')
        (training, validation, test) = ratings.randomSplit([0.6, 0.2, 0.2], seed=42)
        als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', nonnegative=True, coldStartStrategy='drop')
        param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [50]) \
        .addGrid(als.regParam, [0.1]) \
        .build()
        
        evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
        cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
        saved_model = cv.fit(training)
        cache["model"] = saved_model
    else:
        saved_model = cache["model"]

    # Get the movies the user has rated
    user_ratings = ratings.filter(col('userId') == user_id)

    # Get the top 10 recommended movies for the user
    user_recs = saved_model.bestModel.recommendForAllUsers(10).filter(col('userId') == user_id)

    # Extract the recommended movie IDs and explode them into separate rows
    user_recs = user_recs.selectExpr("explode(recommendations) as rec")
    user_recs = user_recs.select('rec.movieId')

    # Join with the movies DataFrame to get the movie titles
    user_recs = user_recs.join(movies, 'movieId')

    # Serialize the result as JSON and return it
    result = {
        'user_id': user_id,
        'movies_liked': list(user_ratings.filter(col('rating') >= 4).join(movies, 'movieId').select('title').toPandas()['title']),
        'movies_disliked': list(user_ratings.filter(col('rating') < 4).join(movies, 'movieId').select('title').toPandas()['title']),
        'recommended_movies': list(user_recs.select('title').toPandas()['title'])
    }
    return JsonResponse(result)
