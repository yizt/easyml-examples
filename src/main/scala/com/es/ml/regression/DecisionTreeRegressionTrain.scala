package com.es.ml.regression

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser


/**
  * Created by zhangw on 2017/12/18.
  * DecisionTreeRegression 训练
  */
object DecisionTreeRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "DecisionTreeRegression_Train",
                    num_iterations: Int = 100  //训练迭代次数
                   )
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("DecisionTreeRegression_Train") {
      head("DecisionTreeRegression_Train: 决策树回归训练.")
      opt[String]("train_data")
        .required()
        .text("训练数据路径")
        .action((x, c) => c.copy(train_data = x))
      opt[String]("model_out")
        .required()
        .text("模型保存路径")
        .action((x, c) => c.copy(model_out = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
      opt[Int]("num_iterations")
        .required()
        .text("迭代次数")
        .action((x, c) => c.copy(num_iterations = x))
    }

    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  def run(p:Params): Unit = {
    val conf = new SparkConf().setAppName(p.appname)
    val sc = new SparkContext(conf)
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32
    val training = MLUtils.loadLibSVMFile(sc,p.train_data) //加载数据
    val model = DecisionTree.trainRegressor(training, categoricalFeaturesInfo, impurity,
        maxDepth, maxBins)
    model.save(sc,p.model_out) //保存模型
    sc.stop()
  }
}
