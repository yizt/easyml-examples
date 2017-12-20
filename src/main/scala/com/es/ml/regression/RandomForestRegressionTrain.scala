package com.es.ml.regression

import com.es.ml.classifier.RandomForestClassifierTrain.Params
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/18.
  * RandomForestRegression 训练
  */
object RandomForestRegressionTrain {
  /** 命令行参数 */
  case class Params(train_data: String = "", //训练数据路径
                    model_out: String = "",  //模型保存路径
                    appname: String = "RandomForestRegression_Train",
                    num_classes: Int = 2 , //类别数
                    num_trees : Int= 3, // Use more in practice.
                    feature_subset_strategy : String= "auto",// Let the algorithm choose.
                    impurity : String= "variance",//
                    max_depth: Int = 4,//
                    max_bins : Int= 32//
                   )
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("RandomForestRegression_Train") {
      head("RandomForestRegression_Train: 随机森林回归训练.")
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
      opt[Int]("num_classes")
        .required()
        .text("类别数")
        .action((x, c) => c.copy(num_classes = x))
      opt[Int]("num_trees")
        .required()
        .text("树的个数")
        .action((x, c) => c.copy(num_trees = x))
      opt[String]("feature_subset_strategy")
        .required()
        .text("特征策略")
        .action((x, c) => c.copy(feature_subset_strategy = x))
      opt[String]("impurity")
        .required()
        .text("impurity")
        .action((x, c) => c.copy(impurity = x))
      opt[Int]("max_depth")
        .required()
        .text("最大深度")
        .action((x, c) => c.copy(max_depth = x))
      opt[Int]("max_bins")
        .required()
        .text("max_bins")
        .action((x, c) => c.copy(max_bins = x))
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
    val categoricalFeaturesInfo = Map[Int, Int]()



    val training = MLUtils.loadLibSVMFile(sc,p.train_data) //加载数据

    val model = RandomForest.trainRegressor(training, categoricalFeaturesInfo,
      p.num_trees, p.feature_subset_strategy, p.impurity, p.max_depth, p.max_bins)

    model.save(sc,p.model_out) //保存模型
    sc.stop()
  }
}
