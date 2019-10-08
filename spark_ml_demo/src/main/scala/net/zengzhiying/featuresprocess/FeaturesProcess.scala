package net.zengzhiying.featuresprocess

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.StringIndexer


object FeaturesProcess {
  def main(args: Array[String]) : Unit = {
    val spark = SparkSession
      .builder
      .master("local[1]")  // 实际提交到spark集群的时候注释掉这里 使用参数提交
      .appName("FeaturesProcess")
      .getOrCreate()
//    wordToVector(spark)
//      countVector(spark)
//      stringIndexerTest(spark)
      
    val a = "3.0"
    val b = a.toDouble
    println(b)
    spark.close()
  }
  
  
  /**
   * 将字符后面添加对应索引的标签
   */
  def stringIndexerTest(spark: SparkSession) {
    val df = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")
    
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
    
    val indexed = indexer.fit(df).transform(df)
    indexed.show()
  }
  
  
  /**
   * 数据转换为计数量
   */
  def countVector(spark: SparkSession) {
    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")
    
    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)
    
    // alternatively, define CountVectorizerModel with a-priori vocabulary
    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")
    
    cvModel.transform(df).show(false)
  }
  
  /**
   * 文本关键词转向量
   */
  def wordToVector(spark : SparkSession) {
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)
    val result = model.transform(documentDF)
    result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }
  }
  
  
}