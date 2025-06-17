from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess

# Start Spark session with custom temp directory (avoids /tmp issues)
spark = SparkSession.builder \
    .appName("StableReviewPipeline_RF") \
    .config("spark.hadoop.fs.defaultFS", "gs://nk009") \
    .config("spark.local.dir", "/mnt/disks/scratch/") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load CSV data from GCS
data = spark.read.option("header", True).csv("gs://nk009/output/cleaned_movies_reviews/cleanedcsv/cleaned dataset.csv")

# Create sentiment label column
data = data.withColumn("label", when(col("rating") >= 3, "positive").otherwise("negative"))
data = data.filter((col("text").isNotNull()) & (col("label").isNotNull()))

# Sample only 20% of data
sampled_data = data.sample(withReplacement=False, fraction=0.2, seed=42)
train, test = sampled_data.randomSplit([0.8, 0.2], seed=42)

# Define ML pipeline stages
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=50)

# Assemble pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, labelIndexer, rf])

# Train model
model = pipeline.fit(train)

# Run predictions
predictions = model.transform(test)

# Evaluate metrics
accuracy = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
precision = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
recall = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
f1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1").evaluate(predictions)
roc_auc = BinaryClassificationEvaluator(labelCol="indexedLabel", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)

print("===== MODEL PERFORMANCE - RANDOM FOREST (20% Sample) =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("===========================================================")

# Save predictions to GCS
predictions.select("text", "label", "prediction") \
    .write.mode("overwrite") \
    .option("header", True) \
    .csv("gs://nk009/output/predictions_sampled/")

# ================================
# Confusion Matrix to GCS
# ================================
conf_df = predictions.select("indexedLabel", "prediction").toPandas()
cm = pd.crosstab(conf_df["indexedLabel"], conf_df["prediction"], rownames=["Actual"], colnames=["Predicted"])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest Classifier (20% Sample)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Save locally and upload to GCS
local_img_path = "/tmp/confusion_matrix_rf.png"
plt.savefig(local_img_path)
subprocess.run(["gsutil", "cp", local_img_path, "gs://nk009/output/plots/confusion_matrix_rf.png"])

# Stop Spark session
spark.stop()
