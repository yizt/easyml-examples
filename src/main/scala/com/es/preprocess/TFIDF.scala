package com.es.preprocess
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
  * Created by mick.yi on 2017/12/18.
  */
object TFIDF {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("SparkSessionZipsExample")
      .enableHiveSupport()
      .getOrCreate()
    //spark.sqlContext.read.parquet()
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")
    sentenceData.write.parquet("")
    spark.read.parquet()



    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val cols=Array("a","b","c")
    val rel=cols.drop(0)
    wordsData.select("ss",cols : _ *)

    import org.apache.spark.ml.feature.StopWordsRemover

    val remover = new StopWordsRemover().
      setInputCol("raw").
      setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)


    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
  }


}
