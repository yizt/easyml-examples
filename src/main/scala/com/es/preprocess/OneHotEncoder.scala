package src.main.scala.com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.sql.SQLContext
/**
  * Created by zhangw on 2017/12/26.
  * OneHotEncoder 类别特征映射为二进制向量
  */
object OneHotEncoder {
  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    output: String = "", //输出数据,parquet格式
                    inputCol: String = "categoryIndex", //输入列名
                    outputCol: String = "categoryVec", //输出列名
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "OneHotEncoder"
                   )

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("OneHotEncoder") {
      head("OneHotEncoder:.")
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

    val outputDF = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")
      .transform(inputDF)

    //保存结果
    val resultDF = DataFrameUtil.select(outputDF, p.resultCols) //只保存选择的列
    resultDF.write.parquet(p.output)


    sc.stop()
  }

}
