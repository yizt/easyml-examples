import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.SparkContext._
/**
  * Created by Administrator on 2017/8/22.
  */

object WordCount {

  /** command line parameters */
  case class Params(input_pt: String = "",
                    output_pt: String = "",
                    appname: String = "")

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("WordCount") {
      head("WordCount: Count words in documents.")
      opt[String]("input_pt")
        .required()
        .text("Input document file path")
        .action((x, c) => c.copy(input_pt = x))
      opt[String]("output_pt")
        .required()
        .text("Output document file path")
        .action((x, c) => c.copy(output_pt = x))
      opt[String]("appname")
        .required()
        .text("appname")
        .action((x, c) => c.copy(appname = x))
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
    val line = sc.textFile(p.input_pt)
    //output to file
    line.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_ + _).saveAsTextFile(p.output_pt)
    //output to  screen
    line.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_ + _).collect().foreach(println)
    sc.stop()
  }

}