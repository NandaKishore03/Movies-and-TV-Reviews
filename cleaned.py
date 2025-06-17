from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, when, from_unixtime

input_path = "gs://nk009/Movies_and_TV.csv/movies.csv"
output_path = "gs://nk009/output/cleaned_movies_reviews"

spark = SparkSession.builder.appName("MoviesTVWrangling").getOrCreate()

# Load data
df = spark.read.option("header", "true").csv(input_path)

# Drop duplicate reviews based on user_id and asin
df = df.dropDuplicates(["user_id", "asin"])

# Drop rows with missing essential fields
df = df.dropna(subset=["user_id", "asin", "text", "rating"])

# Clean text: trim whitespace, lowercase title
df = df.withColumn("title", trim(lower(col("title"))))

# Convert rating to integer
df = df.withColumn("rating", col("rating").cast("int"))

# Create a verified_purchase flag (1 for 'Y', 0 otherwise)
df = df.withColumn("verified_flag", when(col("verified_purchase") == "Y", 1).otherwise(0))

# Convert timestamp (assuming it's in Unix time) to date
df = df.withColumn("review_date", from_unixtime(col("timestamp")).cast("date"))

# Filter out invalid ratings
df = df.filter((col("rating") >= 1) & (col("rating") <= 5))

# Select relevant columns for analysis
columns_to_keep = [
    "asin", "parent_asin", "user_id", "title", "text",
    "rating", "helpful_vote", "verified_flag", "review_date"
]
df_clean = df.select(*columns_to_keep)

# Show a sample of cleaned data
df_clean.show(5)

# Save cleaned data as CSV (overwrite existing files)
df_clean.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

spark.stop()
