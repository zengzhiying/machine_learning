package net.zengzhiying.decisiontreeclassification

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector

/**
 * 决策树分类器
 */
object DecisionTreeClassificationDemo {
  
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("DecisionTreeClassificationDemo")
      .getOrCreate()
      
    val dataSet = spark.read.format("libsvm").load("data/data2.txt")
      .toDF("label","features")
    
    dataSet.foreach(d => {
      println(d.get(0))
      println(d(1))
    })
    
    // 处理label编号索引
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(dataSet)
      
    labelIndexer.transform(dataSet).show(3)
    
    // 处理feature索引
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(dataSet)
      
    featureIndexer.transform(dataSet).show(3)
    
    // 分割数据集 70% 训练，30% 测试
    val Array(trainingData, testData) = dataSet.randomSplit(Array(0.7, 0.3))

    // 定义训练决策树模型
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    
    // 转换索引标签为原始标签
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
      
    // 连接管道
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    
    // 开始训练模型
    val model = pipeline.fit(trainingData)
    
    // 预测模型并输出
    val predictions = model.transform(testData)
    predictions.select("predictedLabel", "label", "features").show(3)
    
    // 保存模型
    // model.save("data/decisiontree.model")
    // 加载模型
    // val loadModel = DecisionTreeClassificationModel.load("data/decisiontree.model")
    // 关于模型和管道的保存和加载：http://www.infoq.com/cn/articles/spark-apache-2-preview
    
    // 选择标签并计算测试的错误率
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(accuracy)
    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
    
    println("预测结果.")
    val testVector = Array(
        Vectors.sparse(3, Seq((0, 198.2), (1, 187.0), (2, 186.0))),
        Vectors.dense(93.0, 92.1, 108.3)
//        Vectors.dense(601, 602, 600) // 分类3测试
    )
//    println(testVector(0)(0))
    val testFatures = spark.createDataFrame(testVector.map(Tuple1.apply(_)))
      .toDF("features")
    new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(testFatures)
    val result2 = model.transform(testFatures).select("predictedLabel")
    result2.foreach(r => {
      println(r.get(0))
    })
    spark.stop()
  }
  
}