package net.zengzhiying.linearsupportvectormachine

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.linalg.Vectors

object SupportVectorMachineDemo {
  def main(args: Array[String]): Unit = {
    // 初始化spark session任务
    val spark = SparkSession
      .builder
      .master("local")
      .appName("SupportVectorMachineDemo")
      .getOrCreate()
      
    // 加载训练集
    val training = spark.read.format("libsvm").load("data/data2.txt")

    // 实例化线性svc并设置其参数
    val lsvc = new LinearSVC()
      .setMaxIter(100)
      .setRegParam(0.1)
      
    // 训练svm模型
    val lsvcModel = lsvc.fit(training)
    
    // 填入测试数据 并封装为dataframe
    val testVector11 = Array(
            Vectors.sparse(3, Seq((0, 189.2), (1, 178.0), (2, 186.0))),
            Vectors.dense(93.0, 92.1, 108.3))        
    val testFatures11 = spark.createDataFrame(testVector11.map(Tuple1.apply))
      .toDF("features")
      
    // 打印线性svc系数和截距
    println(lsvcModel.coefficients)  // 系数
    println(lsvcModel.intercept)  // 截距
    // 打印测试结果
    lsvcModel.transform(testFatures11).show()
    
    spark.close()
  }
}