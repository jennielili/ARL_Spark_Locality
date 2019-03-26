package test
import java.nio.ByteBuffer
import org.apache.spark.{ SparkContext, SparkConf, Partitioner }
import org.apache.spark.rdd.RDD
import alluxio.AlluxioURI
import alluxio.client.file.options.CreateFileOptions
import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import org.apache.hadoop.fs._
import alluxio.client.file.FileSystem
import alluxio.client.WriteType
import alluxio.client.ReadType
import alluxio.client.file.options.OpenFileOptions
import alluxio.Configuration
import alluxio.Constants
import alluxio.PropertyKey
import scala.util.hashing.MurmurHash3
import org.apache.spark.broadcast.Broadcast
class SDPPartitioner_pharo_alluxio(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts
  override def getPartition(key: Any): Int = {
    key.toString.split(',')(2).toInt
  }
}
class SDPPartitioner_facets_alluxio(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts
  override def getPartition(key: Any): Int = {
    key.toString.split(',')(3).toInt
  }
}
object Pipeline_partitioning_alluxio {
  type Data = Array[Byte]

  // Scale data by 1/4 by default to convert from Byte size to Int size
  // This also means we are going to hit the maximum array size limit only
  // at about 8GB.
  val scale_data: Double = 1.0/10.0;
  val scale_compute: Double = 0.0;
  var alluxioHost = "hadoop1"
  var alluxioPort = "19998"
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SDP Pipeline")
    val sc = new SparkContext(conf)

    class MapPartitioner[T](partitions: HashMap[T, Int]) extends Partitioner {
      override def getPartition(key: Any): Int = partitions(key.asInstanceOf[T])
      override def numPartitions = partitions.size
    }

    // Parameters:
    //  Tsnap       = 1008.7 s
    //  Nfacet      = 9x9
    //  Nf_max      = 65536 (using 1638)
    //  Tobs        = 3600.0 s (using 1200.0 s)
    //  Nmajortotal = 9 (using 1)
    // === Extract_LSM ===
    val extract_lsm: RDD[((Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int), Int]()
      var partition = 0
      val initset = ListBuffer[((Int, Int), Unit)]()
      val beam = 0
      val major_loop = 0
      partitions += Tuple2(Tuple2(beam, major_loop), partition)
      partition += 1
      initset += Tuple2(Tuple2(beam, major_loop), ())
      val partitioner = new MapPartitioner(partitions)
      sc.parallelize(initset).partitionBy(partitioner).mapPartitions(extract_lsm_kernel, true)
    }
    printf("   the scale_data    "+scale_data)
    var broadcast_lsm = sc.broadcast(extract_lsm.collect())
    // === Local Sky Model ===
    val local_sky_model: RDD[(Unit, Data)] = {
      val partitions = HashMap[Unit, Int]()
      var partition = 0
      val initset = ListBuffer[(Unit, Unit)]()
      partitions += Tuple2((), partition)
      partition += 1
      initset += Tuple2((), ())
      val partitioner = new MapPartitioner(partitions)
      sc.parallelize(initset).partitionBy(partitioner).mapPartitions(local_sky_model_kernel, true)
    }
    // === Telescope Management ===
    val telescope_management: RDD[(Unit, Data)] = {
      val partitions = HashMap[Unit, Int]()
      var partition = 0
      val initset = ListBuffer[(Unit, Unit)]()
      partitions += Tuple2((), partition)
      partition += 1
      initset += Tuple2((), ())
      val partitioner = new MapPartitioner(partitions)
      sc.parallelize(initset).partitionBy(partitioner).mapPartitions(telescope_management_kernel, true)
    }

    // === Visibility Buffer ===
    /*   val visibility_buffer : RDD[((Int,Int,Int,Int,Int), Data)] = {
            val partitions = HashMap[(Int,Int,Int,Int,Int), Int]()
            var partition = 0
            val initset = ListBuffer[((Int,Int,Int,Int,Int), Unit)]()
            val beam = 0
            for (frequency <- 0 until 20) {
                val time = 0
                val baseline = 0
                val polarisation = 0
                partitions += Tuple2(Tuple5(beam, frequency, time, baseline, polarisation), partition)
                partition += 1
                initset += Tuple2(Tuple5(beam, frequency, time, baseline, polarisation), ())
            }
            val partitioner = new MapPartitioner(partitions)
            sc.parallelize(initset).partitionBy(partitioner).mapPartitions(visibility_buffer_kernel, true)
        }*/
    val visibility_buffer: RDD[(Int, Int, Int, Int, Int)] = {
      val initset = ListBuffer[(Int, Int, Int, Int, Int)]()
      val beam = 0
      for (frequency <- 0 until 20) {
        val time = 0
        val baseline = 0
        val polarisation = 0
        initset += Tuple5(beam, frequency, time, baseline, polarisation)
      }
      sc.parallelize(initset,20).map(visibility_buffer_kernel)
    }
    printf(" the size of new rdd visibility_buffer" + visibility_buffer.count())
    // === Reprojection Predict + IFFT ===
    /*    val reppre_ifft : RDD[((Int,Int,Int,Int,Int,Int), Data)] = {
            val partitions = HashMap[(Int,Int,Int,Int,Int,Int), Int]()
            var partition = 0
            val dep_extract_lsm : HashMap[(Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]] = HashMap[(Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]]()
            val beam = 0
            val major_loop = 0
            for (frequency <- 0 until 5) {
                val time = 0
                for (facet <- 0 until 81) {
                    for (polarisation <- 0 until 4) {
                        partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, facet, polarisation), partition)
                        partition += 1
                        dep_extract_lsm.getOrElseUpdate(Tuple2(beam, major_loop), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
                    }
                }
            }
            val input_extract_lsm : RDD[((Int,Int,Int,Int,Int,Int), Data)] =
              extract_lsm.flatMap(ix_data => dep_extract_lsm(ix_data._1).map((_, ix_data._2)))
            val partitioner = new MapPartitioner(partitions)
            input_extract_lsm.partitionBy(partitioner).mapPartitions(reppre_ifft_kernel, true)
        }*/

    val reppre_ifft: RDD[(Int, Int, Int, Int, Int, Int)] = {
      var initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val dep_extract_lsm = HashMap[(Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      for (frequency <- 0 until 5) {
        val time = 0
        for (facet <- 0 until 81) {
          for (polarisation <- 0 until 4) {
            //dep_extract_lsm.getOrElseUpdate(Tuple2(beam, major_loop), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
            initset += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
          }
        }
      }
      sc.parallelize(initset,80).map(ix => reppre_ifft_kernel(ix, broadcast_lsm))
    }
    reppre_ifft.cache()
    printf(" the size of new rdd reppre_ifft" + reppre_ifft.count())
    // === Telescope Data ===
    val telescope_data: RDD[((Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_telescope_management: HashMap[Unit, ListBuffer[(Int, Int, Int, Int)]] = HashMap[Unit, ListBuffer[(Int, Int, Int, Int)]]()
      val beam = 0
      val frequency = 0
      val time = 0
      val baseline = 0
      partitions += Tuple2(Tuple4(beam, frequency, time, baseline), partition)
      partition += 1
      dep_telescope_management.getOrElseUpdate((), ListBuffer()) += Tuple4(beam, frequency, time, baseline)
      val input_telescope_management: RDD[((Int, Int, Int, Int), Data)] =
        telescope_management.flatMap(ix_data => dep_telescope_management(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_telescope_management.partitionBy(partitioner).mapPartitions(telescope_data_kernel, true)
    }
    var broads_input_telescope_data = sc.broadcast(telescope_data.collect())
    // === Degridding Kernel Update + Degrid ===
    /*   val degkerupd_deg : RDD[((Int,Int,Int,Int,Int,Int), Data)] = {
            val partitions = HashMap[(Int,Int,Int,Int,Int,Int), Int]()
            var partition = 0
            val dep_reppre_ifft : HashMap[(Int,Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]] = HashMap[(Int,Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]]()
            val dep_telescope_data : HashMap[(Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]] = HashMap[(Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]]()
            val beam = 0
            val major_loop = 0
            for (frequency <- 0 until 20) {
                val time = 0
                for (facet <- 0 until 81) {
                    for (polarisation <- 0 until 4) {
                        partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, facet, polarisation), partition)
                        partition += 1
                        dep_telescope_data.getOrElseUpdate(Tuple4(beam, (frequency)/20, time, 0), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
                        dep_reppre_ifft.getOrElseUpdate(Tuple6(beam, major_loop, (frequency)/4, time, facet, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
                    }
                }
            }
            val input_reppre_ifft : RDD[((Int,Int,Int,Int,Int,Int), Data)] =
              reppre_ifft.flatMap(ix_data => dep_reppre_ifft(ix_data._1).map((_, ix_data._2)))
            val input_telescope_data : RDD[((Int,Int,Int,Int,Int,Int), Data)] =
              telescope_data.flatMap(ix_data => dep_telescope_data(ix_data._1).map((_, ix_data._2)))
            val partitioner = new MapPartitioner(partitions)
            input_reppre_ifft.partitionBy(partitioner).zipPartitions(input_telescope_data.partitionBy(partitioner), true)(degkerupd_deg_kernel)
        }*/
    var degkerupd_deg: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      reppre_ifft.flatMap(ix => degkerupd_deg_kernel(ix, broads_input_telescope_data))
    }
    printf(" the size of new rdd degkerupd_deg" + degkerupd_deg.count())
    degkerupd_deg.cache()
    // === Phase Rotation Predict + DFT + Sum visibilities ===
    /* val pharotpre_dft_sumvis : RDD[((Int,Int,Int,Int,Int,Int), Data)] = {
            val partitions = HashMap[(Int,Int,Int,Int,Int,Int), Int]()
            var partition = 0
            val dep_degkerupd_deg : HashMap[(Int,Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]] = HashMap[(Int,Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]]()
            val dep_extract_lsm : HashMap[(Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]] = HashMap[(Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]]()
            val beam = 0
            val major_loop = 0
            for (frequency <- 0 until 20) {
                val time = 0
                val baseline = 0
                val polarisation = 0
                partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, baseline, polarisation), partition)
                partition += 1
                dep_extract_lsm.getOrElseUpdate(Tuple2(beam, major_loop), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
                for (i_facet <- 0 until 81) {
                    for (i_polarisation <- 0 until 4) {
                        dep_degkerupd_deg.getOrElseUpdate(Tuple6(beam, major_loop, frequency, time, i_facet, (polarisation)*4+i_polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
                    }
                }
            }
            val input_degkerupd_deg : RDD[((Int,Int,Int,Int,Int,Int), Data)] =
              degkerupd_deg.flatMap(ix_data => dep_degkerupd_deg(ix_data._1).map((_, ix_data._2)))
            val input_extract_lsm : RDD[((Int,Int,Int,Int,Int,Int), Data)] =
              extract_lsm.flatMap(ix_data => dep_extract_lsm(ix_data._1).map((_, ix_data._2)))
            val partitioner = new MapPartitioner(partitions)
            input_degkerupd_deg.partitionBy(partitioner).zipPartitions(input_extract_lsm.partitionBy(partitioner), true)(pharotpre_dft_sumvis_kernel)
        }
    */
    val pharotpre_dft_sumvis: RDD[(Int, Int, Int, Int, Int, Int)] = {
      val dep_extract_lsm = HashMap[(Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val dep_degkerupd_deg = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val initset = ListBuffer[(Int, Int, Int, Int, Int)]()
      val beam = 0
      for (frequency <- 0 until 20) {
        val time = 0
        val baseline = 0
        val polarisation = 0
        initset += Tuple5(beam, frequency, time, baseline, polarisation)
      }
      degkerupd_deg.partitionBy(new SDPPartitioner_pharo_alluxio(20)).mapPartitions(ix => pharotpre_dft_sumvis_kernel(ix, broadcast_lsm))

    }
    pharotpre_dft_sumvis.cache()
    printf(" the size of new rdd pharotpre_dft_sumvis" + pharotpre_dft_sumvis.count())
    
     val visibility_data: HashMap[Int, Data] = {
      var result = HashMap[Int, Data]()
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      for (frequency <- 0 until 20) {
        var temp = frequency
        val path = new AlluxioURI("/visibility_buffer/" + frequency)
        using(fs.openFile(path, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in =>
          var buf = new Array[Byte](in.remaining().toInt)
          var tempdata = in.read(buf)
          result += Tuple2(temp, buf)
          //   in.close
        }
      } //for
      result
    }
    val pharo_data: HashMap[Int, Data] = {
      var result = HashMap[Int, Data]()
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      for (frequency <- 0 until 20) {
        var temp = frequency
        val path = new AlluxioURI("/visibility_buffer/" + frequency)
        using(fs.openFile(path, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in =>
          var buf = new Array[Byte](in.remaining().toInt)
          var tempdata = in.read(buf)
          result += Tuple2(temp, buf)
          //   in.close
        }
      } //for
      result
    }
    var broad_pharo=sc.broadcast(pharo_data)
    var broad_visibility=sc.broadcast(visibility_data)
    // === Timeslots ===
    /*    val timeslots : RDD[((Int,Int,Int,Int,Int,Int), Data)] = {
            val partitions = HashMap[(Int,Int,Int,Int,Int,Int), Int]()
            var partition = 0
            val dep_visibility_buffer : HashMap[(Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]] = HashMap[(Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]]()
            val dep_pharotpre_dft_sumvis : HashMap[(Int,Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]] = HashMap[(Int,Int,Int,Int,Int,Int), ListBuffer[(Int,Int,Int,Int,Int,Int)]]()
            val beam = 0
            val major_loop = 0
            val frequency = 0
            for (time <- 0 until 120) {
                val baseline = 0
                val polarisation = 0
                partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, baseline, polarisation), partition)
                partition += 1
                for (i_frequency <- 0 until 20) {
                    dep_pharotpre_dft_sumvis.getOrElseUpdate(Tuple6(beam, major_loop, (frequency)*20+i_frequency, (time)/120, baseline, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
                }
                for (i_frequency <- 0 until 20) {
                    dep_visibility_buffer.getOrElseUpdate(Tuple5(beam, (frequency)*20+i_frequency, (time)/120, baseline, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
                }
            }
            val input_visibility_buffer : RDD[((Int,Int,Int,Int,Int,Int), Data)] =
              visibility_buffer.flatMap(ix_data => dep_visibility_buffer(ix_data._1).map((_, ix_data._2)))
            val input_pharotpre_dft_sumvis : RDD[((Int,Int,Int,Int,Int,Int), Data)] =
              pharotpre_dft_sumvis.flatMap(ix_data => dep_pharotpre_dft_sumvis(ix_data._1).map((_, ix_data._2)))
            val partitioner = new MapPartitioner(partitions)
            input_visibility_buffer.partitionBy(partitioner).zipPartitions(input_pharotpre_dft_sumvis.partitionBy(partitioner), true)(timeslots_kernel)
        }
   
   // printf(" the size of new rdd pharotpre_dft_sumvis" + pharotpre_dft_sumvis.count())
*/
   // var broads_input0=sc.broadcast(pharotpre_dft_sumvis.collect())
  //  var broads_input1=sc.broadcast(visibility_buffer.collect())
    //read one time and broadcast them
    val timeslots: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val beam = 0
    //  for (time <- 0 until 120) {
      for (time <- 0 until 120) {
        val frequency = 0
        val baseline = 0
        val polarisation = 0
        val major_loop = 0
        initset += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
      }
      sc.parallelize(initset).map(ix => timeslots_kernel(ix,broad_pharo,broad_visibility))

    }
    println(" time slots   "+timeslots.count())
    // === Solve ===
   /* val solve: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_timeslots: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      val frequency = 0
      for (time <- 0 until 120) {
        val baseline = 0
        val polarisation = 0
        partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, baseline, polarisation), partition)
        partition += 1
        dep_timeslots.getOrElseUpdate(Tuple6(beam, major_loop, frequency, time, baseline, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
      }
      val input_timeslots: RDD[((Int, Int, Int, Int, Int, Int), Data)] =
        timeslots.flatMap(ix_data => dep_timeslots(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_timeslots.partitionBy(partitioner).mapPartitions(solve_kernel, true)
    }*/
    
   
     val solve: RDD[(Int, Int, Int, Int, Int,Int)] = {
      val dep_timeslots = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int,Int)]]()
      val beam = 0
      val major_loop = 0
      val baseline = 0
      val frequency = 0
      for (time <- 0 until 120) {
        val polarisation = 0
        dep_timeslots.getOrElseUpdate(Tuple6(beam, major_loop, frequency, time, baseline, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time,baseline, polarisation)
      }
       timeslots.map(solve_kernel)
    }
     
    printf(" the size of solve" + solve.count())
    val solve_data: HashMap[Int, Data] = {
      var result = HashMap[Int, Data]()
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      for (time <- 0 until 120) {
        val pathTime = new AlluxioURI("/solve/" + time)
         using(fs.openFile(pathTime, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in2 =>
          var buf = new Array[Byte](in2.remaining().toInt)
          var tempdata = in2.read(buf)
          result += Tuple2(tempdata, buf)
          in2.close
        }
      }
      result
    }
    
    // === Correct + Subtract Visibility + Flag ===
   /* val cor_subvis_flag: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_pharotpre_dft_sumvis: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val dep_visibility_buffer: HashMap[(Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val dep_solve: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      for (frequency <- 0 until 20) {
        val time = 0
        val baseline = 0
        val polarisation = 0
        partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, baseline, polarisation), partition)
        partition += 1
        for (i_time <- 0 until 120) {
          dep_solve.getOrElseUpdate(Tuple6(beam, major_loop, (frequency) / 20, (time) * 120 + i_time, baseline, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
        }
        dep_visibility_buffer.getOrElseUpdate(Tuple5(beam, frequency, time, baseline, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
        dep_pharotpre_dft_sumvis.getOrElseUpdate(Tuple6(beam, major_loop, frequency, time, baseline, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
      }
      val input_pharotpre_dft_sumvis: RDD[((Int, Int, Int, Int, Int, Int), Data)] =
        pharotpre_dft_sumvis.flatMap(ix_data => dep_pharotpre_dft_sumvis(ix_data._1).map((_, ix_data._2)))
      val input_visibility_buffer: RDD[((Int, Int, Int, Int, Int, Int), Data)] =
        visibility_buffer.flatMap(ix_data => dep_visibility_buffer(ix_data._1).map((_, ix_data._2)))
      val input_solve: RDD[((Int, Int, Int, Int, Int, Int), Data)] =
        solve.flatMap(ix_data => dep_solve(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_pharotpre_dft_sumvis.partitionBy(partitioner).zipPartitions(input_visibility_buffer.partitionBy(partitioner), input_solve.partitionBy(partitioner), true)(cor_subvis_flag_kernel)
    }
    */
   // solve.cache()
    var broads_solve=sc.broadcast(solve_data)
    
    
    val cor_subvis_flag: RDD[(Int, Int, Int, Int, Int, Int)] = {
      val initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val beam = 0
      for (frequency <- 0 until 20) {
        var time = 0
        val baseline = 0
        val polarisation = 0
        val major_loop = 0
        initset += Tuple6(beam, major_loop, frequency, time, baseline, polarisation)
      }
      sc.parallelize(initset).map(ix => cor_subvis_flag_kernel(ix,broad_pharo,broad_visibility,broads_solve))

    }
    
    printf(" cor_subvis_flag     "+cor_subvis_flag.count());
    
    /*val cor_subvis_flag_data: HashMap[Int, Data] = {
      var result = HashMap[Int, Data]()
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      for (frequency <- 0 until 20) {
        var temp = frequency
        val path = new AlluxioURI("/cor_subvis_flag/" + frequency)
        using(fs.openFile(path, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in =>
          var buf = new Array[Byte](in.remaining().toInt)
          var tempdata = in.read(buf)
          result += Tuple2(temp, buf)
          //   in.close
        }
      } //for
      result
    }
    var borad_cor=sc.broadcast(cor_subvis_flag_data)
    */
   // cor_subvis_flag.cache()
    // === Gridding Kernel Update + Phase Rotation + Grid + FFT + Reprojection ===
  /*  val grikerupd_pharot_grid_fft_rep: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_telescope_data: HashMap[(Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val dep_cor_subvis_flag: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      val frequency = 0
      val time = 0
      for (facet <- 0 until 81) {
        for (polarisation <- 0 until 4) {
          partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, facet, polarisation), partition)
          partition += 1
          for (i_frequency <- 0 until 20) {
            dep_cor_subvis_flag.getOrElseUpdate(Tuple6(beam, major_loop, (frequency) * 20 + i_frequency, time, 0, (polarisation) / 4), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
          }
          dep_telescope_data.getOrElseUpdate(Tuple4(beam, frequency, time, 0), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
        }
      }
      val input_telescope_data: RDD[((Int, Int, Int, Int, Int, Int), Data)] =
        telescope_data.flatMap(ix_data => dep_telescope_data(ix_data._1).map((_, ix_data._2)))
      val input_cor_subvis_flag: RDD[((Int, Int, Int, Int, Int, Int), Data)] =
        cor_subvis_flag.flatMap(ix_data => dep_cor_subvis_flag(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_telescope_data.partitionBy(partitioner).zipPartitions(input_cor_subvis_flag.partitionBy(partitioner), true)(grikerupd_pharot_grid_fft_rep_kernel)
    }*/
    
   //  var broads_input = sc.broadcast(cor_subvis_flag.collect())
    // === Gridding Kernel Update + Phase Rotation + Grid + FFT + Reprojection ===
    val grikerupd_pharot_grid_fft_rep: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {

      val initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val beam = 0
      var frequency = 0
      for (facet <- 0 until 81) {
        for (polarisation <- 0 until 4) {
          val time = 0
          val major_loop = 0
          initset += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
        }
      }
      sc.parallelize(initset).map(ix => grikerupd_pharot_grid_fft_rep_kernel(ix,broads_input_telescope_data))
    }
   grikerupd_pharot_grid_fft_rep.cache()
   
  
    // === Sum Facets ===
    /*val sum_facets: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_grikerupd_pharot_grid_fft_rep: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      val frequency = 0
      val time = 0
      for (facet <- 0 until 81) {
        for (polarisation <- 0 until 4) {
          partitions += Tuple2(Tuple6(beam, major_loop, frequency, time, facet, polarisation), partition)
          partition += 1
          dep_grikerupd_pharot_grid_fft_rep.getOrElseUpdate(Tuple6(beam, major_loop, frequency, time, facet, polarisation), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
        }
      }
      val input_grikerupd_pharot_grid_fft_rep: RDD[((Int, Int, Int, Int, Int, Int), Data)] =
        grikerupd_pharot_grid_fft_rep.flatMap(ix_data => dep_grikerupd_pharot_grid_fft_rep(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_grikerupd_pharot_grid_fft_rep.partitionBy(partitioner).mapPartitions(sum_facets_kernel, true)
    }*/
    println(" grikerupd_pharot_grid_fft_rep   "+grikerupd_pharot_grid_fft_rep.count());
      // === Sum Facets ===
    val sum_facets: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val beam = 0
      var frequency = 0
      for (facet <- 0 until 81) {
        for (polarisation <- 0 until 4) {
          val time = 0
          val major_loop = 0
          initset += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
        }
      }
      grikerupd_pharot_grid_fft_rep.map(sum_facets_kernel)
    }
   sum_facets.cache()
   println(" Sum Facets "+sum_facets.count());
    // === Identify Component ===
    /*val identify_component: RDD[((Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_sum_facets: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      val frequency = 0
      for (facet <- 0 until 81) {
        partitions += Tuple2(Tuple4(beam, major_loop, frequency, facet), partition)
        partition += 1
        for (i_polarisation <- 0 until 4) {
          dep_sum_facets.getOrElseUpdate(Tuple6(beam, major_loop, frequency, 0, facet, i_polarisation), ListBuffer()) += Tuple4(beam, major_loop, frequency, facet)
        }
      }
      val input_sum_facets: RDD[((Int, Int, Int, Int), Data)] =
        sum_facets.flatMap(ix_data => dep_sum_facets(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_sum_facets.partitionBy(partitioner).mapPartitions(identify_component_kernel, true)
    }
   identify_component.cache()
   */
    val identify_component: RDD[((Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_sum_facets: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      val frequency = 0
      for (facet <- 0 until 81) {
        partitions += Tuple2(Tuple4(beam, major_loop, frequency, facet), partition)
        partition += 1
        for (i_polarisation <- 0 until 4) {
          dep_sum_facets.getOrElseUpdate(Tuple6(beam, major_loop, frequency, 0, facet, i_polarisation), ListBuffer()) += Tuple4(beam, major_loop, frequency, facet)
        }
      }
   //   val input_sum_facets: RDD[((Int, Int, Int, Int), Data)] =
     //   sum_facets.flatMap(ix_data => dep_sum_facets(ix_data._1).map((_, ix_data._2)))
     // val partitioner = new MapPartitioner(partitions)
    //  input_sum_facets.partitionBy(partitioner).mapPartitions(identify_component_kernel, true)
         sum_facets.partitionBy(new SDPPartitioner_facets_alluxio(81)).mapPartitions(identify_component_kernel)
    }
     println(" identify_components "+identify_component.count());
   var identify=sc.broadcast(identify_component.collect())
    // === Source Find ===
    val source_find: RDD[((Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int), Int]()
      var partition = 0
      val dep_identify_component: HashMap[(Int, Int, Int, Int), ListBuffer[(Int, Int)]] = HashMap[(Int, Int, Int, Int), ListBuffer[(Int, Int)]]()
      val beam = 0
      val major_loop = 0
      partitions += Tuple2(Tuple2(beam, major_loop), partition)
      partition += 1
      for (i_facet <- 0 until 81) {
        dep_identify_component.getOrElseUpdate(Tuple4(beam, major_loop, 0, i_facet), ListBuffer()) += Tuple2(beam, major_loop)
      }
      val input_identify_component: RDD[((Int, Int), Data)] =
        identify_component.flatMap(ix_data => dep_identify_component(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_identify_component.partitionBy(partitioner).mapPartitions(source_find_kernel, true)
    }
     println(" Source Find  "+source_find.count());
    // === Subtract Image Component ===
    val subimacom: RDD[((Int, Int, Int, Int), Data)] = {
       sum_facets.partitionBy(new SDPPartitioner_facets_alluxio(81)).mapPartitions(ix =>(subimacom_kernel(ix,identify)))
    }
    println(" Subtract Image Component   "+subimacom.count())
    // === Update LSM ===
    val update_lsm: RDD[((Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int), Int]()
      var partition = 0
      val dep_local_sky_model: HashMap[Unit, ListBuffer[(Int, Int)]] = HashMap[Unit, ListBuffer[(Int, Int)]]()
      val dep_source_find: HashMap[(Int, Int), ListBuffer[(Int, Int)]] = HashMap[(Int, Int), ListBuffer[(Int, Int)]]()
      val beam = 0
      val major_loop = 0
      partitions += Tuple2(Tuple2(beam, major_loop), partition)
      partition += 1
      dep_source_find.getOrElseUpdate(Tuple2(beam, major_loop), ListBuffer()) += Tuple2(beam, major_loop)
      dep_local_sky_model.getOrElseUpdate((), ListBuffer()) += Tuple2(beam, major_loop)
      val input_local_sky_model: RDD[((Int, Int), Data)] =
        local_sky_model.flatMap(ix_data => dep_local_sky_model(ix_data._1).map((_, ix_data._2)))
      val input_source_find: RDD[((Int, Int), Data)] =
        source_find.flatMap(ix_data => dep_source_find(ix_data._1).map((_, ix_data._2)))
      val partitioner = new MapPartitioner(partitions)
      input_local_sky_model.partitionBy(partitioner).zipPartitions(input_source_find.partitionBy(partitioner), true)(update_lsm_kernel)
    }
    // === Terminate ===
    println("Finishing...")
    println(f"Subtract Image Component: ${subimacom.count()}%d")
    println(f"Update LSM: ${update_lsm.count()}%d")
  
    
    sc.stop()
  }
  def extract_lsm_kernel: (Iterator[((Int, Int), Unit)]) => Iterator[((Int, Int), Data)] = {
    case (ixs) =>
      var hash: Int = 0
      var input_size: Long = 0
      val ix: (Int, Int) = ixs.next._1
      val label: String = "Extract_LSM (0.0 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 0L).toInt))
   //   result(0) = hash
//      ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

  def local_sky_model_kernel: (Iterator[(Unit, Unit)]) => Iterator[(Unit, Data)] = {
    case (ixs) =>
      var hash: Int = 0
      var input_size: Long = 0
      val ix: Unit = ixs.next._1
      val label: String = "Local Sky Model (0.0 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 0L).toInt))
    //  result(0) = hash
    //  ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }
  def using[A <: { def close(): Unit }, B](resource: A)(f: A => B): B =
    try { f(resource) } finally { resource.close() }
  def telescope_management_kernel: (Iterator[(Unit, Unit)]) => Iterator[(Unit, Data)] = {
    case (ixs) =>
      var hash: Int = 0
      var input_size: Long = 0
      val ix: Unit = ixs.next._1
      val label: String = "Telescope Management (0.0 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 0L).toInt))
//      result(0) = hash
   //   ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

   def visibility_buffer_kernel: ((Int, Int, Int, Int, Int)) => ((Int, Int, Int, Int, Int)) = {
    case ix =>
      var label: String = "Visibility Buffer (46168.5 MB, 0.00 Tflop) " + ix.toString
      var hash: Int = MurmurHash3.stringHash(label)
      var input_size: Long = 0
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      val path = new AlluxioURI("/visibility_buffer/" + ix._2)
      val result = new Array[Byte](math.max(4, (scale_data * 1823123744L).toInt))
      using(fs.createFile(path, CreateFileOptions.defaults().setWriteType(WriteType.MUST_CACHE))) { out =>
        out.write(result)
      }
      (ix)
  }
 /* def visibility_buffer_kernel: ((Int, Int, Int, Int, Int)) => (((Int, Int, Int, Int, Int), Data)) = {
    case (ixs) =>
      var hash: Int = 0
      var input_size: Long = 0
      val ix: (Int, Int, Int, Int, Int) = ixs
      val label: String = "Visibility Buffer (546937.1 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 1823123744L).toInt))
     // result(0) = hash
      ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      (ix, result)
  }*/

  def reppre_ifft_kernel: ((Int, Int, Int, Int, Int, Int), Broadcast[Array[((Int, Int), Data)]]) => (Int, Int, Int, Int, Int, Int) = {
    case (reppre, data_extract_lsm) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      ix = reppre
      for ((dix, data) <- data_extract_lsm.value) {
        hash ^= data(0)
        input_size += data.length

      }
      val label: String = "Reprojection Predict + IFFT (14645.6 MB, 2.56 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 48818555L).toInt))
   //   result(0) = hash
   //   Thread.sleep((scale_compute * 412).toInt)
      (ix, result)
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get()
      val path = new AlluxioURI("/reppre_ifft/" + ix._1 + "_" + ix._2 + "_" + ix._3 + "_" + ix._4 + "_" + ix._5 + "_" + ix._6)
      if(fs.exists(path))
        fs.delete(path)
      using(fs.createFile(path, CreateFileOptions.defaults().setWriteType(WriteType.MUST_CACHE))) { out =>
        out.write(result)
      }
      (ix)
  }

  def telescope_data_kernel: (Iterator[((Int, Int, Int, Int), Data)]) => Iterator[((Int, Int, Int, Int), Data)] = {
    case (data_telescope_management) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int) = (0, 0, 0, 0)
      for ((dix, data) <- data_telescope_management) {
        hash ^= data(0)
        input_size += data.length
        ix = dix
      }
      val label: String = "Telescope Data (0.0 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 0L).toInt))
    //  result(0) = hash.toByte
  //    ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

  def degkerupd_deg_kernel: ((Int, Int, Int, Int, Int, Int), Broadcast[Array[((Int, Int, Int, Int), Data)]]) => TraversableOnce[(((Int, Int, Int, Int, Int, Int), Data))] = {
    case (data_reppre_ifft, data_telescope_data) =>
      var hash: Int = 0
       var input_size: Long = 0
        var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
   
      ix = data_reppre_ifft
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      var temp = ix._1 + "_" + ix._2 + "_" + ix._3 + "_" + ix._4 + "_" + ix._5 + "_" + ix._6
      val path = new AlluxioURI("/reppre_ifft/" + temp)
      using(fs.openFile(path, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in =>
        var buf = new Array[Byte](in.remaining().toInt)
        var tempdata = in.read(buf)
//        hash ^= tempdata(0)
//        input_size += tempdata.length
        in.close
      }
     
     
      for ((dix2, data2) <- data_telescope_data.value) {
        hash ^= data2(0)
        input_size += data2.length
        // ix = dix
      }
      
      
      val label: String = "Degridding Kernel Update + Degrid (674.8 MB, 0.59 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
    //  val result = new Array[Int](math.max(1, (scale_data * 2249494L).toInt))

    //  Thread.sleep((scale_compute * 95).toInt)
      var mylist = new Array[((Int, Int, Int, Int, Int, Int), Data)](4)
      val result1 = new Array[Byte](math.max(4, (scale_data * 2249494L).toInt))
      result1(0) = hash.toByte
       ByteBuffer.wrap(result1).putInt(0, hash)
      val result2 = new Array[Byte](math.max(4, (scale_data * 2249494L).toInt))
      result2(0) = hash.toByte
      ByteBuffer.wrap(result2).putInt(0, hash)
      val result3 = new Array[Byte](math.max(4, (scale_data * 2249494L).toInt))
      result3(0) = hash.toByte
      ByteBuffer.wrap(result3).putInt(0, hash)
      val result4 = new Array[Byte](math.max(4, (scale_data * 2249494L).toInt))
      result4(0) = hash.toByte
      ByteBuffer.wrap(result4).putInt(0, hash)

      var temp1 = ix._3 * 4
      mylist(0) = ((ix._1, ix._2, temp1, ix._4, ix._5, ix._6), result1)
      var temp2 = ix._3 * 4 + 1
      mylist(1) = ((ix._1, ix._2, temp2, ix._4, ix._5, ix._6), result2)
      var temp3 = ix._3 * 4 + 2
      mylist(2) = ((ix._1, ix._2, temp3, ix._4, ix._5, ix._6), result3)
      var temp4 = ix._3 * 4 + 3
      mylist(3) = ((ix._1, ix._2, temp4, ix._4, ix._5, ix._6), result4)
      mylist

  }

  def pharotpre_dft_sumvis_kernel: (Iterator[((Int, Int, Int, Int, Int, Int),Data)], Broadcast[Array[((Int, Int), Data)]]) => Iterator[(Int, Int, Int, Int, Int, Int)] = {
    case (data_degkerupd_deg, data_extract_lsm) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      val label: String = "Phase Rotation Predict + DFT + Sum visibilities (546937.1 MB, 512.53 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
    //  val result = new Array[Byte](math.max(1, (scale_data * 1823123744L).toInt))
     // result(0) = hash
     
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      if(data_degkerupd_deg.hasNext)
      {
        var temp = data_degkerupd_deg.next()
        var result2 = new Array[(Int, Int, Int, Int, Int,Int)](1)
        result2(0) = (temp._1._1, temp._1._2, temp._1._3, temp._1._4,temp._1._5,temp._1._6)

        val result = new Array[Byte](math.max(4, (scale_data * 1823123744L).toInt))
        ByteBuffer.wrap(result).putInt(0, hash)
         //write the pair to alluxio
        val path3 = new AlluxioURI("/pharotpre_dft_sumvis/" + temp._1._3)
        using(fs.createFile(path3, CreateFileOptions.defaults().setWriteType(WriteType.MUST_CACHE))) { out =>
          out.write(result)
        } //using
        Iterator((ix))
      }
      else {
        var result2 = new Array[(Int, Int, Int, Int, Int,Int)](1)
        result2(0) = ((0, 0, 0, 0, 0,0))
        result2.iterator

      }
    //  Thread.sleep((scale_compute * 82666).toInt)
      //Iterator((ix, result))
  }

  /*def timeslots_kernel: ((Int, Int, Int, Int, Int, Int), Broadcast[Array[((Int, Int, Int, Int, Int,Int), Data)]], Broadcast[Array[((Int, Int, Int, Int, Int), Data)]]) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (idx,data_visibility_buffer, data_pharotpre_dft_sumvis) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      ix=idx
      for ((dix, data) <- data_visibility_buffer.value) {
        hash ^= data(0)
        input_size += data.length
     //   ix = dix
      }
      for ((dix, data) <- data_pharotpre_dft_sumvis.value) {
        hash ^= data(0)
        input_size += data.length
      //  ix = dix
      }
      val label: String = "Timeslots (1518.3 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 5060952L).toInt))
    //  result(0) = hash
      ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      (ix, result)
  }*/
  def timeslots_kernel: ((Int, Int, Int, Int, Int, Int),Broadcast[HashMap[Int, Data]],Broadcast[HashMap[Int, Data]]) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (ix,broad1,broad2) =>
      val label: String = "Timeslots (1518.3 MB, 0.00 Tflop) " + ix.toString
      //from alluxio get the info 
   /*   var hash: Int = 0
       var input_size : Long = 0
      for ((dix, data) <- broad1.value) {
       hash ^= MurmurHash3.bytesHash(data.slice(0,4))
        input_size += data.length
      //  ix = dix
      }*/
   //  println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
    /*  for (frequency <- 0 until 20) {
        var temp = frequency
        val path = new AlluxioURI("/pharotpre_dft_sumvis/" + temp)
        val path2 = new AlluxioURI("/visibility_buffer/" + temp)

        using(fs.openFile(path, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in =>
          var buf = new Array[Byte](in.remaining().toInt)
          var tempdata = in.read(buf)
          in.close
        }
        using(fs.openFile(path2, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in2 =>
          var buf = new Array[Byte](in2.remaining().toInt)
          var tempdata = in2.read(buf)
          in2.close
        }
      } //for
      * 
      */
      val result = new Array[Byte](math.max(4, (scale_data * 5060952L).toInt))
      (ix, result)
  }
def solve_kernel: (((Int, Int, Int, Int, Int,Int),Data)) => (Int, Int, Int, Int, Int,Int) = {
    case (ix, data) =>
      var label : String = "Solve (8262.8 MB, 2939.73 Tflop) "+ix.toString
      var hash: Int = MurmurHash3.stringHash(label)
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      val result = new Array[Byte](math.max(4, (scale_data * 27542596L).toInt))
      val path3 = new AlluxioURI("/solve/" + ix._4)
      using(fs.createFile(path3, CreateFileOptions.defaults().setWriteType(WriteType.MUST_CACHE))) { out =>
        out.write(result)
      } //using
      (ix)
  }
 /* def solve_kernel: ((Int, Int, Int, Int, Int, Int)) => (Int, Int, Int, Int, Int, Int) = {
   /* case (data_timeslots) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      val (dix,data)=data_timeslots
      hash ^= data(0)
      input_size += data.length
      ix = dix
     
      val label: String = "Solve (8262.8 MB, 16.63 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 27542596L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 2682).toInt)
      (ix, result)
      */
      case (ix) =>
      var label : String = "Solve (8262.8 MB, 2939.73 Tflop) "+ix.toString
      var hash: Int = MurmurHash3.stringHash(label)
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      val result = new Array[Byte](math.max(4, (scale_data * 27542596L).toInt))
      val path3 = new AlluxioURI("/solve/" + ix._4)
      using(fs.createFile(path3, CreateFileOptions.defaults().setWriteType(WriteType.MUST_CACHE))) { out =>
        out.write(result)
      } //using
      (ix)
  }
*/
  def cor_subvis_flag_kernel: ((Int, Int, Int, Int, Int, Int ),Broadcast[HashMap[Int, Data]],Broadcast[HashMap[Int, Data]],Broadcast[HashMap[Int, Data]])=> ((Int, Int, Int, Int, Int, Int)) = {
    case (ix,broad1,broad2,broad3) =>
     // var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      var hash: Int = 0
      var input_size: Long = 0
     // var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      val label: String = "Correct + Subtract Visibility + Flag (153534.1 MB, 4.08 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
    /*  for (frequency <- 0 until 20) {
        var temp = frequency
        val path = new AlluxioURI("/pharotpre_dft_sumvis/" + temp)
        val path2 = new AlluxioURI("/visibility_buffer/" + temp)

        using(fs.openFile(path, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in =>
          var buf = new Array[Byte](in.remaining().toInt)
          var tempdata = in.read(buf)
          in.close
        }
        using(fs.openFile(path2, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in2 =>
          var buf = new Array[Byte](in2.remaining().toInt)
          var tempdata = in2.read(buf)
          in2.close
        }
      } //for
      * 
      */
     /*  for (time <- 0 until 120) {
        val pathTime = new AlluxioURI("/solve/" + time)
         using(fs.openFile(pathTime, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in2 =>
          var buf = new Array[Byte](in2.remaining().toInt)
          var tempdata = in2.read(buf)
          in2.close
        }
      }*/
     val result = new Array[Byte](math.max(4, (scale_data * 511780275L/81).toInt))
          //write to alluxio  /cor_subvis_flag/
     for(facet<- 0 until 81)
     {
       for(pol<- 0 until 4)
       {
	     val path3 = new AlluxioURI("/cor_subvis_flag/" + ix._3+"_"+facet+"_"+pol)
	     using(fs.createFile(path3, CreateFileOptions.defaults().setWriteType(WriteType.MUST_CACHE))) { out =>
	         out.write(result)
	     } //using
       }
     }
      (ix)
  }
  /*def grikerupd_pharot_grid_fft_rep_kernel: (Iterator[((Int, Int, Int, Int, Int, Int),Data)],Broadcast[Array[((Int, Int, Int, Int), Data)]]) => (Iterator[((Int, Int, Int, Int, Int, Int), Data)]) = {
    case (data_cor_subvis_flag,broad) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      for ((dix, data) <- data_cor_subvis_flag) {
     //   hash ^= data(0)
        input_size += data.length
        ix = dix
      }
      val label: String = "Gridding Kernel Update + Phase Rotation + Grid + FFT + Reprojection (14644.9 MB, 20.06 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
       log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 48816273L).toInt))
    //  result(0) = hash
    //  Thread.sleep((scale_compute * 3234).toInt)
      Iterator((ix, result))
   }*/
  def grikerupd_pharot_grid_fft_rep_kernel: ((Int, Int, Int, Int, Int, Int),Broadcast[Array[((Int, Int, Int, Int), Data)]]) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (idx, data_telescope_data) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      ix=idx
   /*   for ((dix, data) <- data_telescope_data.value) {
        hash ^= data(0)
        input_size += data.length
     //   ix = dix
      }
      for ((dix, data) <- data_cor_subvis_flag.value) {
        hash ^= data(0)
        input_size += data.length
     //   ix = dix
      }*/
      //read data from alluxio 
      Configuration.set(PropertyKey.MASTER_HOSTNAME, alluxioHost);
      Configuration.set(PropertyKey.MASTER_RPC_PORT, alluxioPort);
      System.setProperty("HADOOP_USER_NAME", "hadoop")
      val fs = FileSystem.Factory.get();
      for(fre<- 0 until 20)
      {
	      val path = new AlluxioURI("/cor_subvis_flag/" + fre+"_"+ix._5+"_"+ix._6)
	      using(fs.openFile(path, OpenFileOptions.defaults().setReadType(ReadType.CACHE))) { in =>
	          var buf = new Array[Byte](in.remaining().toInt)
	          var tempdata = in.read(buf)
	          in.close
	       }
      }
      val label: String = "Gridding Kernel Update + Phase Rotation + Grid + FFT + Reprojection (14644.9 MB, 20.06 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 48816273L).toInt))
     // result(0) = hash
//      ByteBuffer.wrap(result).putInt(0, hash)
    //  Thread.sleep((scale_compute * 3234).toInt)
      (ix, result)
  }

  def sum_facets_kernel: (((Int, Int, Int, Int, Int, Int), Data)) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (data_grikerupd_pharot_grid_fft_rep) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
     val (dix, data) = data_grikerupd_pharot_grid_fft_rep
        hash ^= data(0)
        input_size += data.length
        ix = dix
      
      val label: String = "Sum Facets (14644.9 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 48816273L).toInt))
      
      // result(0) = hash
//      ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      (ix, result)
  }

  def identify_component_kernel: (Iterator[((Int, Int, Int, Int,Int,Int), Data)]) => Iterator[((Int, Int, Int, Int), Data)] = {
    case (data_sum_facets) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int) = (0, 0, 0, 0)
      for ((dix, data) <- data_sum_facets) {
        hash ^= data(0)
        input_size += data.length
        ix = (dix._1,dix._2,dix._3,dix._5)
      }
      val label: String = "Identify Component (0.2 MB, 1830.61 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 533L).toInt))
    //  result(0) = hash
 //     ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 295259).toInt)
      Iterator((ix, result))
  }

  def source_find_kernel: (Iterator[((Int, Int), Data)]) => Iterator[((Int, Int), Data)] = {
    case (data_identify_component) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int) = (0, 0)
      for ((dix, data) <- data_identify_component) {
        hash ^= data(0)
        input_size += data.length
        ix = dix
      }
      val label: String = "Source Find (5.8 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 19200L).toInt))
     // result(0) = hash
    //  ByteBuffer.wrap(result).putInt(0, hash)
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

  def subimacom_kernel: (Iterator[((Int, Int, Int, Int,Int,Int), Data)],Broadcast[Array[((Int, Int, Int, Int), Data)]]) => Iterator[((Int, Int, Int, Int), Data)] = {
    case ( data_sum_facets,data_identify_component) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int) = (0, 0, 0, 0)
      for ((dix, data) <- data_identify_component.value) {
        hash ^= data(0)
        input_size += data.length
        
      }
      for ((dix, data) <- data_sum_facets) {
        hash ^= data(0)
        input_size += data.length
        ix = (dix._1,dix._2,dix._3,dix._5)
     //   ix = dix
      }
      val label: String = "Subtract Image Component (73224.4 MB, 67.14 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 244081369L).toInt))
     //result(0) = hash
  //    ByteBuffer.wrap(result).putInt(0, hash)
    //  Thread.sleep((scale_compute * 10829).toInt)
      Iterator((ix, result))
  }

  def update_lsm_kernel: (Iterator[((Int, Int), Data)], Iterator[((Int, Int), Data)]) => Iterator[((Int, Int), Data)] = {
    case (data_local_sky_model, data_source_find) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int) = (0, 0)
      for ((dix, data) <- data_local_sky_model) {
        hash ^= data(0)
        input_size += data.length
        ix = dix
      }
      for ((dix, data) <- data_source_find) {
        hash ^= data(0)
        input_size += data.length
        ix = dix
      }
      val label: String = "Update LSM (0.0 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Byte](math.max(1, (scale_data * 0L).toInt))
     // result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

}
//  * 10825 tasks
//  * 268.68 GB produced (reduced from 80.61 TB, factor 300)
//  * 185.21 Pflop represented
// This is roughly(!):
//  * 497.89 min node time (6.20 Tflop/s effective)
//  * 529.18 s island time (0.35 Pflop/s effective)
//  * 13.23 s cluster time (14.00 Pflop/s effective)
//  * 0.367% SKA SDP
//  * 0.00122% SKA SDP internal data rate

