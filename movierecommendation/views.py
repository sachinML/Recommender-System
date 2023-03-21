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


DATA_FOLDER = "ml-latest-small"


@require_GET
def recommend_movies(request, user_id='1'):
    # Create a SparkSession
    spark = SparkSession.builder.appName('MovieRecommendation').getOrCreate()

    # Loading the dataset
    ratings = spark.read.format('csv').options(header='true', inferSchema='true').load(f'{DATA_FOLDER}//ratings.csv').cache()
    movies = spark.read.format('csv').options(header='true', inferSchema='true').load(f'{DATA_FOLDER}//movies.csv').cache()

    if not cache:
        # Load the saved model
        saved_model = ALSModel.load("savedALS_model")
        cache["model"] = saved_model
    else:
        saved_model = cache["model"]

    # Get the movies the user has rated
    user_ratings = ratings.filter(col('userId') == user_id)

    # Get the top 10 recommended movies for the user
    user_recs = saved_model.recommendForAllUsers(10).filter(col('userId') == user_id)

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
