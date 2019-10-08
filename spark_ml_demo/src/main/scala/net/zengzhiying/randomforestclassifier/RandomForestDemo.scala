package net.zengzhiying.randomforestclassifier

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.linalg.Vectors

/**
 * 随机森林分类器
 */
object RandomForestDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("RandomForestDemo")
      .getOrCreate()
      
    // 加载样本集
    val dataSet = spark.read.format("libsvm").load("data/data3.txt").toDF("label", "features")
    
    // 设置label
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(dataSet)
      
    // 转换features为向量
    val featuresIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(dataSet)
      
    // 分割样本集与测试集
    val Array(trainingData, testData) = dataSet.randomSplit(Array(0.7, 0.3))
    
    // 初始化随机森林分类器训练模型
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
    
    // 转换模型预测索引标签为原始易读的标签
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
      
    // 初始化管道 组装输入，输出模型等组件
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featuresIndexer, rf, labelConverter))
      
    // 开始训练
    val model = pipeline.fit(trainingData)
    
    // 测试集计算
    val predictions = model.transform(testData)
    
    // 打印结果集
    predictions.select("predictedLabel", "label", "features").show(5)
    
    // 计算error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")  // 设置输出错误率的列
      
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
    
    // 打印模型详情
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification forest model:\n" + rfModel.toDebugString)
    
    
    // 预测结果
    val predictionData = Array(
        Vectors.sparse(3, Seq((0, 123.5), (1, 20.5), (2, 4.0))),
        Vectors.dense(126, 30, 0),
        Vectors.dense(124.5, 23, 0)
    )
    
    val predictionDataSet = spark.createDataFrame(predictionData.map(Tuple1.apply)).toDF("features")
    
    val predictResult = model.transform(predictionDataSet)
    
    predictResult.show()
    var index = 1
    println("预测结果： ")
    predictResult.foreach(f => {
//      println(f(5))
      val preLabel: Float = f(5).toString().toFloat
      print(s"第${index}个结果：")
      if(preLabel > 0) {
        println("男.")
      } else {
        println("女.")
      }
      index += 1
    })
    
    spark.close()
  }
}