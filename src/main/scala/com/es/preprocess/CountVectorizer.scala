package src.main.scala.com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}

/**
  * Created by zhangw on 2017/12/26.
  * CountVectorizer 统计字信息
  */
object CountVectorizer {

  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    output: String = "", //输出数据,parquet格式
                    inputCol: String = "", //输入列名
                    outputCol: String = "", //输出列名
                    vcabSize:Int=3,//Max size of the vocabulary
                    minDF:Int=2,
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "CountVectorizer"
                   )

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("CountVectorizer") {
      head("CountVectorizer:.")
      opt[String]("input")
        .required()
        .text("输入数据")
        .action((x, c) => c.copy(input = x))
      opt[String]("output")
        .required()
        .text("输出数据")
        .action((x, c) => c.copy(output = x))
      opt[String]("inputCol")
        .required()
        .text("输入列名")
        .action((x, c) => c.copy(inputCol = x))
      opt[String]("outputCol")
        .required()
        .text("输出列名")
        .action((x, c) => c.copy(outputCol = x))
      opt[Int]("vcabSize")
        .required()
        .text("词汇最大长度")
        .action((x, c) => c.copy(vcabSize = x))
      opt[Int]("minDF")
        .required()
        .text("minDF")
        .action((x, c) => c.copy(minDF = x))
      opt[String]("resultCols")
        .required()
        .text("输出结果保留的列")
        .action((x, c) => c.copy(resultCols = x))
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
    val sqlContext = new SQLContext(sc)
    import sqlContext.sparkSession.implicits._
    val inputDF = sqlContext.read.parquet(p.input)

    // fit a CountVectorizerModel from the corpus
    val outputDF = new CountVectorizer()
      .setInputCol(p.inputCol)
      .setOutputCol(p.outputCol)
      .setVocabSize(p.vcabSize)//Max size of the vocabulary
      .setMinDF(p.minDF)//Specifies the minimum number of different documents a term must appear in to be included in the vocabulary.
      .fit(inputDF)
      .transform(inputDF)

    //保存结果
    val resultDF = DataFrameUtil.select(outputDF, p.resultCols) //只保存选择的列
    resultDF.write.parquet(p.output)

    sc.stop()
  }

}
