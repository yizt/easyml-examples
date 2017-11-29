package com.es.ml

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by mick.yi on 2017/11/29.
  */
object KMeansTrain {
  /** 命令行参数 */
  case class Params(data: String = "", //测试数据路径
                    cluster_out: String = "", //聚类结果保存路径
                    nunclusters:Int = 3, //中心点个数(即聚为几类)
                    appname: String = "KMeans"
                   )
  def main(args: Array[String]) {
    if (args.length < 6) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("KMeansTrain") {
      head("KMeans: 聚类.")
      opt[String]("data")
        .required()
        .text("测试数据路径")
        .action((x, c) => c.copy(data = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
      opt[String]("cluster_out")
        .required()
        .text("聚类结果输出")
        .action((x, c) => c.copy(cluster_out = x))
      opt[Int]("nunclusters")
        .required()
        .text("中心点个数")
        .action((x, c) => c.copy(nunclusters = x))
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
    val data = sc.textFile(p.data)
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val numIterations = 20
    val clusters = KMeans.train(parsedData, p.nunclusters, numIterations)
    //打印所有中心点
    clusters.clusterCenters.foreach(center=>println("中心点:"+center))
    //保存聚类结果
    parsedData.map(elem=>(clusters.predict(elem),elem)).saveAsTextFile(p.cluster_out)
    sc.stop()
  }

}
