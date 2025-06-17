from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder.appName("JSONL_to_CSV").getOrCreate()

    # Input and output GCS paths
    input_path = "gs://nk009/Movies_and_TV.jsonl"
    output_path = "gs://nk009/path/to/Movies_andTV_csv/movies.csv"

    # Read JSONL file
    df = spark.read.json(input_path)

    # Optional: print schema for debugging
    df.printSchema()

    # Write to a single CSV file with header
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

    spark.stop()

if __name__ == "__main__":
    main()
