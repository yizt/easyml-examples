package com.es.ml.clustering
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
/**
  * Created by zhangw on 2017/12/19.
  * LDA 训练 Latent Dirichlet allocation潜在狄利克雷分布
  */
object LDATrain {
  /** 命令行参数 */
  case class Params(data: String = "", //测试数据路径
                    cluster_out: String = "", //聚类结果保存路径
                    nunclusters:Int = 3, //中心点个数(即聚为几类)
                    appname: String = "LDA"
                   )
  def main(args: Array[String]) {
    if (args.length <4) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("LDA") {
      head("LDA: 聚类.")
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

    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))
    // Index documents with unique IDs
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()
    val model = new LDA().setK(p.nunclusters).run(corpus)

    println("Learned topics (as distributions over vocab of " + model.vocabSize + " words):")
    val topics = model.topicsMatrix
    for (topic <- Range(0, 3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, model.vocabSize)) { print(" " + topics(word, topic)); }
      println()
    }
    //保存聚类结果
   // parsedData.map(elem=>(model.topicsMatrix(elem),elem)).saveAsTextFile(p.cluster_out)

    sc.stop()
  }
}
