
import breeze.linalg.InjectNumericOps
import breeze.linalg.Matrix.castOps
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.desc
import spire.std.map

import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
object Test {
  val url = "jdbc:mysql://120.55.126.186:3306/spark1?useSSL=false&useUnicode=true&characterEncoding=UTF-8&useAffectedRows=true"
  val user = "root"		//数据库用户名
  val pwd = "123456"			//数据库密码
  val properties = new java.util.Properties()
  properties.setProperty("user", user)
  properties.setProperty("password", pwd)
  properties.setProperty("driver", "com.mysql.jdbc.Driver")



  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Test")
    val sc = new SparkContext(conf)

    val filestr = "file:///H:\\IntelliJ IDEA 2021.3.1\\projects\\Spark\\"
    val movies = sc.textFile(filestr+"movies.dat")
    val ratings = sc.textFile(filestr+"ratings.dat")

    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder.master("local[4]").getOrCreate
    import spark.implicits._


    val data_rating = ratings.map{ row=>{
      val Array(id,mid,rate,date)=row.split("::")
      (id,mid,rate,date.toLong)
    }}.toDF("id","mid","rate","date")
    val data_movies = movies.map{ row=>{
      val Array(id,name,cate)=row.split("::")
      (id,name,cate)
    }}.toDF("mid","name","cate")


////    data2.show()
//    // 获取评分区间电影数
////    data_rating.show()
//    // 计算电影评分
//    val movies_rating = data_rating.groupBy("mid").agg(("rate","avg"))
//    val num_movies =new ArrayBuffer[(String, Int)]()
//    for (i <-1 to 5){
//      num_movies.append((i.toString,movies_rating.filter("avg(rate)>="+i.toString+" and avg(rate)<"+(i+1).toString).count().toInt))
//    }
//    val num_movies_df = num_movies.toDF("rate","num")
//    num_movies_df.show()
////    num_movies_df.write.jdbc(url,"movies_num",properties)
//
//    //获取每种类型的电影数
////    data_movies.show()
//    val movies_cate = movies.map(x=>x.split("::")(2)).collect().toList.flatMap(_.split("\\|")).map(x=>(x,1)).toDF("cate","num")
//    val num_movies_cate = movies_cate.groupBy("cate").agg(("num","sum")).toDF("cate","num")
//    num_movies_cate.show()
////    num_movies_cate.write.jdbc(url,"movies_cate",properties)
//    // 获取每种类型电影的打分
//    val data_movies_cate_rate = movies_rating.join(data_movies,"mid")
//    data_movies_cate_rate.show()
//    var movies_cate_rate = data_movies_cate_rate.select("avg(rate)", "cate").collect().toList
//    var data_cate_rate = new ArrayBuffer[(String, Double)]()
//    for(i <- 0 until movies_cate_rate.size){
//      val cate = movies_cate_rate(i)(1).toString.split("\\|")
//      for(j <- 0 until cate.length)
//      data_cate_rate.append((cate(j),movies_cate_rate(i)(0).toString.toDouble))
//    }
//    var data_cate_rate1=data_cate_rate.toDF("cate","rate").groupBy("cate").agg(("rate","avg"))
//    data_cate_rate1.show()
////    data_cate_rate1.write.jdbc(url,"movies_cate_rate",properties)
//    //计算评分前100
//    val movies_rate_100 = data_rating.groupBy("mid").count.filter("count(mid) >= 10").join(movies_rating,"mid").sort(desc("avg(rate)")).limit(100)
//    movies_rate_100.show()
////    movies_rate_100.write.jdbc(url,"movies_rate_rank",properties)
//    //计算每种类型电影评分top10
//    val movies_cate_10 = data_rating.groupBy("mid").count.filter("count(mid) >= 10").join(movies_rating,"mid").join(data_movies,"mid")
//    movies_cate_rate = movies_cate_10.select("avg(rate)","cate").collect().toList
//    data_cate_rate = new ArrayBuffer[(String, Double)]()
//    for(i <- 0 until movies_cate_rate.size){
//      val cate = movies_cate_rate(i)(1).toString.split("\\|")
//      for(j <- 0 until cate.length)
//        data_cate_rate.append((cate(j),movies_cate_rate(i)(0).toString.toDouble))
//    }
//    val data_cate_rate2 = data_cate_rate.toDF("cate","rate").groupBy("cate").agg(("rate","avg")).limit(10)
//    data_cate_rate2.show()
////    data_cate_rate2.write.jdbc(url,"movies_cate_rank",properties)

    val person_data = sc.textFile(filestr+"personalRatings.dat").map(x=>x.split("::") match {case Array(id,mid,rate,date)=>Rating(id.toString.toInt,mid.toString.toInt,rate.toString.toInt)})
    val model = fun(data_rating.select("id","mid","rate").rdd.map(x=>Rating(x(0).toString.toInt,x(1).toString.toInt,x(2).toString.toInt)),person_data)

    //
    val res = model.recommendProducts(1,10).map{
      case Rating(user,product,rating)=> (product,rating)
    }.toList.toDF("mid","rate")
    res.show()
//    res.write.jdbc(url,"recommend_movies",properties)



  }

  def fun(data: RDD[Rating],person_data:RDD[Rating]): MatrixFactorizationModel = {


    val model = ALS.train(data, 9, 12, 0.03)
    val pre = model.predict(person_data.map(x=>(x.user,x.product))).map(x=>((x.user,x.product),x.rating))

    val res = person_data.map(x=>((x.user,x.product),x.rating)).join(pre)
//    res.foreach(println)
    val MSE = res.map{case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err*err
    }.mean()
    println(MSE)
    return model
  }



}
