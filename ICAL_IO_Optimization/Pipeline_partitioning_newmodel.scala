package test
import org.apache.log4j.Logger
import org.apache.spark.{ SparkContext, SparkConf, Partitioner }
import org.apache.spark.rdd.RDD

import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import org.apache.spark.serializer.KryoRegistrator;
import scala.util.hashing.MurmurHash3
import org.apache.spark.broadcast.Broadcast
class SDPPartitioner_pharo_newmodel(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts
  override def getPartition(key: Any): Int = {
    key.toString.split(',')(3).toInt
  }
}
class SDPPartitioner_facets_newmodel(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts
  override def getPartition(key: Any): Int = {
    key.toString.split(',')(4).toInt
  }
}
class SDPPartitioner_facets_time(numParts: Int) extends Partitioner {
  override def numPartitions: Int = numParts
  override def getPartition(key: Any): Int = {
    key.toString.split(',')(3).toInt
  }
}
object Pipeline_partitioning_newmodel {
  type Data = Array[Int]
  val log = Logger.getLogger(getClass.getName)
  // Scale data by 1/4 by default to convert from Byte size to Int size
  // This also means we are going to hit the maximum array size limit only
  // at about 8GB.
  var scale_data: Double = 6.0 / 4.0;
  val scale_compute: Double = 0.0;
  var timeslot=72;
  var uvparallel=72;

  def main(args: Array[String]) {

    // System.setProperty("spark.serializer", "org.apache.spark.serializer.KryoRegistrator")
    //System.setProperty("spark.kryo.registrator", "MyRegistrator")
    val conf = new SparkConf().setAppName("SDP Pipeline")
    //  scale_data=args(0).toDouble
    val sc = new SparkContext(conf)
    conf.registerKryoClasses(Array(classOf[Array[Int]]))

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
      printf("   the ************* scale_data    " + scale_data)
      val partitions = HashMap[(Int, Int), Int]()
      var partition = 0
      val initset = ListBuffer[((Int, Int), Unit)]()
      val beam = 0
      val major_loop = 0
      partitions += Tuple2(Tuple2(beam, major_loop), partition)
      partition += 1
      initset += Tuple2(Tuple2(beam, major_loop), ())
      val partitioner = new MapPartitioner(partitions)

      sc.parallelize(initset, 1).partitionBy(partitioner).mapPartitions(extract_lsm_kernel, true)
    }
    printf("   the scale_data    " + scale_data)
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
      sc.parallelize(initset, 1).partitionBy(partitioner).mapPartitions(local_sky_model_kernel, true)
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
      sc.parallelize(initset, 1).partitionBy(partitioner).mapPartitions(telescope_management_kernel, true)
    }

    var broads_localskymodel = sc.broadcast(local_sky_model.collect())
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
    val degkerupd_deg: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      var initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val dep_extract_lsm = HashMap[(Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      var frequency = 0
      //  for (frequency <- 0 until 5) {
      for (time <- 0 until timeslot) {
        for (facet <- 0 until 81) {
          for (polarisation <- 0 until 4) {
            //dep_extract_lsm.getOrElseUpdate(Tuple2(beam, major_loop), ListBuffer()) += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
            initset += Tuple6(beam, major_loop, frequency, time, facet, polarisation)
          }
        }
        //  }
      }
      //   sc.parallelize(initset).map(ix => reppre_ifft_kernel(ix, broadcast_lsm))
      sc.parallelize(initset, timeslot).flatMap(ix => reppre_ifft_Degrid_kernel(ix, broads_input_telescope_data, broadcast_lsm))
    }
    degkerupd_deg.cache()
    val pharotpre_dft_sumvis: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val dep_extract_lsm = HashMap[(Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val dep_degkerupd_deg = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val initset = ListBuffer[(Int, Int, Int, Int, Int)]()
      val beam = 0
      for (frequency <- 0 until 1) {
        val time = 0
        val baseline = 0
        val polarisation = 0
        initset += Tuple5(beam, frequency, time, baseline, polarisation)
      }
      degkerupd_deg.partitionBy(new SDPPartitioner_pharo_newmodel(timeslot)).mapPartitions(ix => pharotpre_dft_sumvis_kernel(ix, broadcast_lsm))
      //    degkerupd_deg.map(ix=> ((0,0,ix._1._3,0,0,0),ix._2)).groupByKey().map(ix => pharotpre_dft_sumvis_kernel(ix, broadcast_lsm))
    }

  
    printf(" the size of new rdd pharotpre_dft_sumvis" + pharotpre_dft_sumvis.count())
    pharotpre_dft_sumvis.cache()
    val timeslots: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      pharotpre_dft_sumvis.map(timeslots_kernel)

    }

    println("timeslots    " + timeslots.count())
    timeslots.cache()
    val solve: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val dep_timeslots = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      val baseline = 0
      val frequency = 0
      timeslots.map(solve_kernel)

    }
    println("sovle    " + solve.count())
    solve.cache()
    var broads_input2 = sc.broadcast(solve.collect())
    // data reduction by 20 of cor_subvis_flag 
    val cor_subvis_flag: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      pharotpre_dft_sumvis.map(ix => cor_subvis_flag_kernel(ix, broads_input2))
    }
    cor_subvis_flag.cache()
    println("    cor_subvis_flag    " + cor_subvis_flag.count())
    val cor_subvis_flag_reduction: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      cor_subvis_flag.map(cor_subvis_flag_kernel_reduction)

    }
    var correct_input_reduction = sc.broadcast(cor_subvis_flag_reduction.collect())
    val grikerupd_pharot_grid_fft_rep: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val beam = 0
      val major_loop = 0
      val frequency = 0
      val time = 0
      val polarisation = 0
      for (facet <- 0 until 81) {
        initset += Tuple6(beam, major_loop, frequency, time, facet, polarisation)

      }
      sc.parallelize(initset, uvparallel).map(ix => grikerupd_pharot_grid_fft_rep_kernel(ix, broads_input_telescope_data, correct_input_reduction))

    }
    println("      grikerupd_pharot_grid    " + grikerupd_pharot_grid_fft_rep.count())
    grikerupd_pharot_grid_fft_rep.cache()
    val sum_facets: RDD[((Int, Int, Int, Int, Int, Int), Data)] = {
      val initset = ListBuffer[(Int, Int, Int, Int, Int, Int)]()
      val beam = 0
      var frequency = 0
      grikerupd_pharot_grid_fft_rep.map(sum_facets_kernel)
    }
    sum_facets.cache()
    println("      sum_facets    " + sum_facets.count())
    // === Identify Component ===
    val identify_component: RDD[((Int, Int, Int, Int), Data)] = {
      val partitions = HashMap[(Int, Int, Int, Int), Int]()
      var partition = 0
      val dep_sum_facets: HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int)]] = HashMap[(Int, Int, Int, Int, Int, Int), ListBuffer[(Int, Int, Int, Int)]]()
      val beam = 0
      val major_loop = 0
      val frequency = 0
      sum_facets.map(identify_component_kernel)
    }
    var broads_componet = sc.broadcast(identify_component.collect())
    
    println("      identify_component    " + identify_component.count())
    val source_find: RDD[((Int, Int), Data)] = {
      
      val initset = ListBuffer[(Int,Int)]()
      initset+=Tuple2(0,0)
      sc.parallelize(initset,1).map(ix => source_find_kernel(ix, broads_componet))
      
    }
    source_find.cache()
    println("      source_find    " + source_find.count())
   
    val subimacom: RDD[((Int, Int, Int, Int), Data)] = {
      sum_facets.map(ix => subimacom_kernel(ix, broads_componet))

    }
   println("      subimacom    " + subimacom.count())
    // === Update LSM ===
   /* val update_lsm: RDD[((Int, Int), Data)] = {
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
    }*/
    val update_lsm: RDD[((Int, Int), Data)] = {
      source_find.map(ix => update_lsm_kernel(ix, broads_localskymodel))

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
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 0L).toInt))
      result(0) = hash
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
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)" + "  scale_data " + scale_data)
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 0L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

  def telescope_management_kernel: (Iterator[(Unit, Unit)]) => Iterator[(Unit, Data)] = {
    case (ixs) =>
      var hash: Int = 0
      var input_size: Long = 0
      val ix: Unit = ixs.next._1
      val label: String = "Telescope Management (0.0 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 0L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

  def visibility_buffer_kernel: ((Int, Int, Int, Int, Int)) => (((Int, Int, Int, Int, Int), Data)) = {
    case (ixs) =>
      var hash: Int = 0
      var input_size: Long = 0
      val ix: (Int, Int, Int, Int, Int) = ixs
      val label: String = "Visibility Buffer (546937.1 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 1823123744L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      (ix, result)
  }

  def reppre_ifft_kernel: ((Int, Int, Int, Int, Int, Int), Broadcast[Array[((Int, Int), Data)]]) => ((Int, Int, Int, Int, Int, Int), Data) = {
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
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 48818555L / timeslot).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 412).toInt)
      (ix, result)
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
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 0L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      Iterator((ix, result))
  }

  def reppre_ifft_Degrid_kernel: ((Int, Int, Int, Int, Int, Int), Broadcast[Array[((Int, Int, Int, Int), Data)]], Broadcast[Array[((Int, Int), Data)]]) => TraversableOnce[((Int, Int, Int, Int, Int, Int), Data)] = {
    case (reppre, data_telescope_data, data_extract_lsm) =>
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
      val result = new Array[Byte](math.max(1, (scale_data * 48818555L *5/ timeslot).toInt))
      //   result(0) = hash
      Thread.sleep((scale_compute * 412).toInt)
      Thread.sleep((scale_compute * 95).toInt)
      var mylist = new Array[((Int, Int, Int, Int, Int, Int), Data)](4)
      val result1 = new Array[Int](math.max(4, (scale_data * 2249494L*20/timeslot).toInt))
      result1(0) = hash
      val result2 = new Array[Int](math.max(4, (scale_data * 2249494L*20/timeslot).toInt))
      result2(0) = hash
      val result3 = new Array[Int](math.max(4, (scale_data * 2249494L*20/timeslot).toInt))
      result3(0) = hash
      val result4 = new Array[Int](math.max(4, (scale_data * 2249494L*20/timeslot).toInt))
      result4(0) = hash

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
  def degkerupd_deg_kernel: (((Int, Int, Int, Int, Int, Int), Data), Broadcast[Array[((Int, Int, Int, Int), Data)]]) => TraversableOnce[(((Int, Int, Int, Int, Int, Int), Data))] = {
    case (data_reppre_ifft, data_telescope_data) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      val (dix, data) = data_reppre_ifft
      hash ^= data(0)
      input_size += data.length
      ix = dix
      for ((dix2, data2) <- data_telescope_data.value) {
        hash ^= data2(0)
        input_size += data.length
        // ix = dix
      }
      val label: String = "Degridding Kernel Update + Degrid (674.8 MB, 0.59 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      //  val result = new Array[Int](math.max(1, (scale_data * 2249494L).toInt))

      Thread.sleep((scale_compute * 95).toInt)
      var mylist = new Array[((Int, Int, Int, Int, Int, Int), Data)](4)
      val result1 = new Array[Int](math.max(4, (scale_data * 2249494L).toInt))
      result1(0) = hash
      val result2 = new Array[Int](math.max(4, (scale_data * 2249494L).toInt))
      result2(0) = hash
      val result3 = new Array[Int](math.max(4, (scale_data * 2249494L).toInt))
      result3(0) = hash
      val result4 = new Array[Int](math.max(4, (scale_data * 2249494L).toInt))
      result4(0) = hash

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

  def pharotpre_dft_sumvis_kernel: (Iterator[((Int, Int, Int, Int, Int, Int), Data)], Broadcast[Array[((Int, Int), Data)]]) => Iterator[((Int, Int, Int, Int, Int, Int), Data)] = {
    case (data_degkerupd_deg, data_extract_lsm) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      for ((dix, data) <- data_degkerupd_deg) {
        hash ^= data(0)
        input_size += data.length
        ix = (0, 0, dix._3, 0, 0, 0)
        //    println("   ix    in pharotpre_dft_sumvis_kernel  ********    "+ix)
      }
      for ((dix, data) <- data_extract_lsm.value) {
        hash ^= data(0)
        input_size += data.length
        //         ix = dix
      }
      val label: String = "Phase Rotation Predict + DFT + Sum visibilities (546937.1 MB, 512.53 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      //    log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      // Here generate the visibility data, from alluxio or by socket, 20 frequencies together, 12 time slots
      val result = new Array[Int](math.max(1, (scale_data * 2 * 1823123744L*20/ timeslot).toInt))
      result(0) = hash
      //  Thread.sleep((scale_compute * 82666).toInt)
      Iterator((ix, result))
  }
  /* def pharotpre_dft_sumvis_kernel: (((Int, Int, Int, Int, Int, Int), Iterable[Data]), Broadcast[Array[((Int, Int), Data)]]) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (data_degkerupd_deg, data_extract_lsm) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
   /*   for ((dix, data) <- data_degkerupd_deg) {
        hash ^= data(0)
        input_size += data.length
        ix = dix
      }
      for ((dix, data) <- data_extract_lsm.value) {
        hash ^= data(0)
        input_size += data.length
        // ix = dix
      }*/
      ix=data_degkerupd_deg._1
      val label: String = "Phase Rotation Predict + DFT + Sum visibilities (546937.1 MB, 512.53 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
       log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 1823123744L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 82666).toInt)
     (ix, result)
  }*/

  def timeslots_kernel: (((Int, Int, Int, Int, Int, Int), Data)) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (data_timeslots) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      val (dix, data) = data_timeslots
      ix = dix

      val label: String = "Timeslots (1518.3 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      //  log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 5060952L * 10).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      (ix, result)
  }

  def solve_kernel: (((Int, Int, Int, Int, Int, Int), Data)) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (data_timeslots) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      val (dix, data) = data_timeslots
      hash ^= data(0)
      input_size += data.length
      ix = dix

      val label: String = "Solve (8262.8 MB, 16.63 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      //   log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 27542596L * timeslot/120).toInt))
      result(0) = hash
      // Thread.sleep((scale_compute * 2682).toInt)
      (ix, result)
  }

  def cor_subvis_flag_kernel: (((Int, Int, Int, Int, Int, Int), Data), Broadcast[Array[((Int, Int, Int, Int, Int, Int), Data)]]) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (pha_vis, data_solve) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      val (ixx, data) = pha_vis
      hash ^= data(0)
      input_size += data.length
      for ((dix, data) <- data_solve.value) {
        hash ^= data(0)
        input_size += data.length
        //  ix = dix
      }
      ix = ixx
      val label: String = "Correct + Subtract Visibility + Flag (153534.1 MB, 4.08 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      //    log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 511780275L * 20 / timeslot).toInt))
      result(0) = hash
      // Thread.sleep((scale_compute * 658).toInt)
      (ix, result)
  }
  def cor_subvis_flag_kernel_reduction: (((Int, Int, Int, Int, Int, Int), Data)) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (ix, data) =>
      val label: String = "Correct reduction (153534.1 MB, 4.08 Tflop) " + ix.toString

      val result = new Array[Int](math.max(1, (scale_data * 511780275L / timeslot).toInt))
      // Thread.sleep((scale_compute * 658).toInt)
      (ix, result)
  }

  def grikerupd_pharot_grid_fft_rep_kernel: ((Int, Int, Int, Int, Int, Int), Broadcast[Array[((Int, Int, Int, Int), Data)]], Broadcast[Array[((Int, Int, Int, Int, Int, Int), Data)]]) => ((Int, Int, Int, Int, Int, Int), Data) = {
    case (idx, data_telescope_data, data_cor_subvis_flag) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0)
      ix = idx
      val label: String = "Gridding Kernel Update + Phase Rotation + Grid + FFT + Reprojection (14644.9 MB, 20.06 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 48816273L*4).toInt))
      result(0) = hash
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
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 48816273L*4).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      (ix, result)
  }
 
  def identify_component_kernel: (((Int, Int, Int, Int, Int, Int), Data)) => ((Int, Int, Int, Int), Data) = {
    case (facet) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int) = (0, 0, 0, 0)
      val (dix, data) = facet
      hash ^= data(0)
      input_size += data.length
      ix = (dix._1, dix._2, dix._3, dix._5)
      val label: String = "Identify Component (0.2 MB, 1830.61 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 533L).toInt))
      result(0) = hash
      //  Thread.sleep((scale_compute * 295259).toInt)
      (ix, result)
  }

  def source_find_kernel: (((Int, Int)),Broadcast[Array[((Int, Int, Int, Int), Data)]]) => ((Int, Int), Data) = {
    case (id,data_identify_component) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int) = (0, 0)
      
      val label: String = "Source Find (5.8 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      val result = new Array[Int](math.max(1, (scale_data * 19200L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      (ix,result)
  }

  def subimacom_kernel: (((Int, Int, Int, Int, Int, Int), Data), Broadcast[Array[((Int, Int, Int, Int), Data)]]) => ((Int, Int, Int, Int), Data) = {
    case (data_sum_facets, data_identify_component) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int, Int, Int) = (0, 0, 0, 0)
      for ((dix, data) <- data_identify_component.value) {
        hash ^= data(0)
        input_size += data.length
        ix = dix
      }
      val (dix, data) = data_sum_facets
      hash ^= data(0)
      input_size += data.length
      //   ix = dix

      val label: String = "Subtract Image Component (73224.4 MB, 67.14 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000).toString() + " KB input)")
      val result = new Array[Int](math.max(1, (scale_data * 244081369L).toInt))
      result(0) = hash
      //  Thread.sleep((scale_compute * 10829).toInt)
      (ix, result)
  }

  def update_lsm_kernel: (((Int, Int), Data), Broadcast[Array[(Unit, Data)]])  => ((Int, Int), Data) = {
    case (data_source_find,data_local_sky_model) =>
      var hash: Int = 0
      var input_size: Long = 0
      var ix: (Int, Int) = (0, 0)
      val label: String = "Update LSM (0.0 MB, 0.00 Tflop) " + ix.toString
      hash ^= MurmurHash3.stringHash(label)
      println(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000000).toString() + " MB input)")
      log.info(label + " (hash " + Integer.toHexString(hash) + " from " + (input_size / 1000).toString() + " KB input)")
      val result = new Array[Int](math.max(1, (scale_data * 0L).toInt))
      result(0) = hash
      Thread.sleep((scale_compute * 0).toInt)
      (ix, result)
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

