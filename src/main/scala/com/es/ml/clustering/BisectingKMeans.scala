package com.es.ml.clustering
import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
  * Created by zhangw on 2017/12/19.
  */
object BisectingKMeans {
  /** 命令行参数 */
  case class Params(data: String = "", //测试数据路径
                    cluster_out: String = "", //聚类结果保存路径
                    nunclusters:Int = 6, //中心点个数(即聚为几类)
                    appname: String = "BisectingKMeans"
                   )
  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("BisectingKMeans") {
      head("BisectingKMeans: 二分-聚类.")
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

    def parse(line: String): Vector = Vectors.dense(line.split(" ").map(_.toDouble))
    val data = sc.textFile(p.data).map(parse).cache()

    val bkm = new BisectingKMeans().setK(p.nunclusters)
    val model = bkm.run(data)
    //打印所有中心点
    model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
      println(s"Cluster Center ${idx}: ${center}")
    }
    model.save(sc,p.cluster_out)
    //保存聚类结果
    //parsedData.map(elem=>(model.predict(elem),elem)).saveAsTextFile(p.cluster_out)
    sc.stop()
  }
}
