package com.es.evaluate

import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/12/25.
  * 二分类评估模型
  */
object BinaryClassification {
  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,text格式
                     appName: String = "BinaryClassificationEvaluate"
                   )

  def main(args: Array[String]) {
    if (args.length < 1) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("BinaryClassificationEvaluate") {
      head("BinaryClassificationEvaluate:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))

      opt[String]("appName")
        .required()
        .text("appName")
        .action((x, c) => c.copy(appName = x))
    }
    parser.parse(args, default_params).map { params =>
      run(params)
    } getOrElse {
      System.exit(1)
    }

  }

  def run(p: Params): Unit = {
    val conf = new SparkConf().setAppName(p.appName)
    val sc = new SparkContext(conf)

    val input = sc.textFile(p.input).map(elem=>{
      val arrs=elem.split(" ")
      (arrs(0).toDouble,arrs(1).toDouble)
    })
    import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

    val metrics = new BinaryClassificationMetrics(input)
    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    // F-measure
    val f1Score = metrics.fMeasureByThreshold
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    f1Score.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    // AUPRC
    val auPRC = metrics.areaUnderPR
    println("Area under precision-recall curve = " + auPRC)

    // Compute thresholds used in ROC and PR curves
    val thresholds = precision.map(_._1)

    // ROC Curve
    val roc = metrics.roc

    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)

    sc.stop()
  }

}
