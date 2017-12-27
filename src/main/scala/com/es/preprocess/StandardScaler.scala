package src.main.scala.com.es.preprocess

import com.es.util.DataFrameUtil
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

/**
  * Created by Administrator on 2017/12/26.
  * StandardScaler 零均值规范化
  *
  */
object StandardScaler {
  /** 命令行参数 */
  case class Params(input: String = "", //输入数据,parquet格式
                    output: String = "", //输出数据,parquet格式
                    inputCol: String = "features", //输入列名
                    outputCol: String = "scaledFeatures", //输出列名
                    withStd:Boolean=true,//标准值
                    withMean:Boolean=false,//均值
                    resultCols: String = "", //输出结果保留的列,默认全部输出
                    appName: String = "StandardScaler"
                   )

  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: <file>")
      System.exit(1)
    }

    val default_params = Params()
    val parser = new OptionParser[Params]("StandardScaler") {
      head("StandardScaler:.")
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
      opt[Boolean]("withStd")
        .required()
        .text("标准值")
        .action((x, c) => c.copy(withStd = x))
      opt[Boolean]("withMean")
        .required()
        .text("均值")
        .action((x, c) => c.copy(withMean = x))
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
    val spark = SparkSession.builder.appName(p.appName).getOrCreate()
    val sc = spark.sparkContext
    val data = spark.read.format("libsvm").load(p.input)

    val outputDF = new StandardScaler()
      .setInputCol(p.inputCol)
      .setOutputCol(p.outputCol)
      .setWithStd(p.withStd)
      .setWithMean(p.withMean)
      .fit(data)
      .transform(data)

    //保存结果
    val resultDF = DataFrameUtil.select(outputDF, p.resultCols) //只保存选择的列
    resultDF.write.parquet(p.output)

    sc.stop()
  }


}
