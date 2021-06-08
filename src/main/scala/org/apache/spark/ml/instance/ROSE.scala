/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.instance

import org.apache.spark.ml.param.Params
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.param.shared.HasLabelCol
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.sql.functions._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.LabelUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.linalg.{Vector, Vectors, DenseMatrix}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors, Vector => OldVector, DenseMatrix => OldDenseMatrix, SparseMatrix => OldSparseMatrix}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import scala.util.Random
import scala.math.pow
import org.apache.spark.sql.Row
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.ml.param.BooleanParam

/**
 * ROSE Transformer.
 * It produces a synthetic, possibly balanced, sample of data simulated 
 * according to a smoothed-bootstrap approach.
 *
 * @author Mario Juez-Gil
 * @param uid  All types inheriting from `Identifiable` require a `uid`.
 *             This includes Transformers, Estimators, and Models.
 */
class ROSE(override val uid: String) extends Transformer 
  with ROSEParams with HasLabelCol with HasFeaturesCol 
  with HasSeed with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("rosetr"))

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setMinoProb(value: Double): this.type = set(minoProb, value)

  /** @group setParam */
  def setShrinkFactor(value: Double): this.type = set(shrinkFactor, value)

  /** @group setParam */
  def setKeepOrigSamples(value: Boolean): this.type = set(keepOrigSamples, value)

  /** @group setParam */
  def setMajoRUS(value: Boolean): this.type = set(majoRUS, value)

  def setKeepMajoSize(value: Boolean): this.type = set(keepMajoSize, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    print($(seed))
    print($(minoProb))
    val rng = new Random($(seed))
    
    val classCounts = LabelUtils.getLabelsInfo(dataset, $(labelCol))

    require(classCounts.size == 2, 
      "Random Balance requires a binary classification data set")

    val (minorityLabel, minoritySize) = classCounts(0)
    val (majorityLabel, majoritySize) = classCounts(1)

    val numFeatures = countFeatures(dataset)

    val size = if($(keepMajoSize)){
      majoritySize * 2
    }else{
      minoritySize + majoritySize
    }
    val newMinorityFrac = (size * $(minoProb)) / minoritySize
    val newMajorityFrac = (size * (1 - $(minoProb))) / majoritySize

    val minority = dataset.filter(col($(labelCol)) === minorityLabel).cache
    val majority = dataset.filter(col($(labelCol)) === majorityLabel).cache

    val newMino = newSample(minority, minoritySize, numFeatures, minorityLabel, newMinorityFrac, rng)
    val newMajo = if($(majoRUS)){
      if($(keepMajoSize)){
        majority.toDF
      }else{
        majority.sample(false, newMajorityFrac, rng.nextLong).toDF
      }
    }else{
      newSample(majority, majoritySize, numFeatures, majorityLabel, newMajorityFrac, rng)
    }
    
    //val newMinoritySample = minority.sample(true, newMinorityFrac, rng.nextLong).cache
    //val newMajoritySample = majority.sample(true, newMajorityFrac, rng.nextLong).cache
    //val minorityROSE = doROSE(minority, minoritySize, newMinoritySample, numFeatures, minorityLabel, rng)
    //val majorityROSE = doROSE(majority, majoritySize, newMajoritySample, numFeatures, majorityLabel, rng)
    //minority.unpersist
    //majority.unpersist
    //newMinoritySample.unpersist
    //newMajoritySample.unpersist
    
    //minorityROSE.union(majorityROSE)
    newMino.union(newMajo)
  }

  private def newSample(data: Dataset[_], dataSize: Long, 
      numFeatures: Int, label: Double, frac: Double, rng: Random): DataFrame = {
    val newFrac = if($(keepOrigSamples)){
      frac - 1
    }else{
      frac
    }
    if(newFrac > 0){
      val newSample = data.sample(true, newFrac, rng.nextLong).cache
      val roseSample = doROSE(data, dataSize, newSample, numFeatures, label, rng)
      newSample.unpersist
      if($(keepOrigSamples)){
        data.toDF.union(roseSample)
      }else{
        roseSample
      }
    }else{
      data.sample(false, newFrac + 1.0, rng.nextLong).toDF
    }
  }

  private def doROSE(originalData: Dataset[_], sizeOrig: Long, 
      sample: Dataset[_], numFeatures: Int, label: Double, rng: Random): DataFrame = {
    //https://medium.com/balabit-unsupervised/scalable-sparse-matrix-multiplication-in-apache-spark-c79e9ffc0703
    val session = originalData.sparkSession
    val sc = session.sparkContext
    val kernel = pow((4.0/((numFeatures+2)*sizeOrig)), (1.0/(numFeatures+4)))
    // val variance = new RowMatrix(originalData
    //   .select(col($(featuresCol)))
    //   .rdd.map{case Row(v: Vector) => OldVectors.fromML(v)})
    //   .computeColumnSummaryStatistics()
    //   .variance.asBreeze
    val variance = originalData.select(Summarizer.variance(col($(featuresCol)))).first.getAs[Vector](0).asBreeze
    val std = variance.map{e => math.sqrt(e)}
    val diag = OldVectors.fromBreeze($(shrinkFactor) * kernel * std)
    val diagMatBcast = sc.broadcast(OldDenseMatrix.diag(diag))
    val sampleSize = sample.count.toInt
    val rnormMat = new IndexedRowMatrix(sc.parallelize(Array.range(0, sampleSize))
      .map{ id => 
        val rnormArr = Array.fill(numFeatures)(rng.nextGaussian)
        new IndexedRow(id, OldVectors.dense(rnormArr))
      })
    val randSampleMat = rnormMat
      .multiply(diagMatBcast.value)
      .rows.map{ ir =>
        (ir.index, ir.vector.asML)
      }
    diagMatBcast.destroy(false)
    //val randSampleMat = diagMat
    //  .multiply(new IndexedRowMatrix(sc.parallelize(Array.range(0, sampleSize))
    //    .map{ id => 
    //      val rnormArr = Array.fill(numFeatures)(rng.nextGaussian)
    //      new IndexedRow(id, OldVectors.dense(rnormArr))
    //    }).toBlockMatrix(50,50))
    //val randMatRDD = new RowMatrix(randSample)
    //  .multiply(diagMatBcast.value).rows
    //  .zipWithIndex.map{ case(v, i) => i -> v }
    //diagMatBcast.destroy(false)
    val sampleMat = sample
      .select(col($(featuresCol)))
      .rdd.zipWithIndex.map{ case (Row(v: Vector), id) => 
        (id, v)
      }
    val synthRDD = randSampleMat.join(sampleMat).map{ e =>
     val feats = Vectors.fromBreeze(e._2._1.asBreeze + e._2._2.asBreeze)
     Row(label, feats)
    }
    // val synthRDD = randSampleMat.add(sampleMat)
    //   .toIndexedRowMatrix.rows.map{ case IndexedRow(_, v) =>
    //     Row(label, v.asML)
    //   }
    session.createDataFrame(synthRDD, sample.schema)
    //https://stackoverflow.com/questions/35497736/difference-between-rowmatrix-and-matrix-in-apache-spark
    //https://stackoverflow.com/questions/33558755/matrix-multiplication-in-apache-spark
  }

  private def countFeatures(dataset: Dataset[_]): Int = {
    try {
      dataset
        .schema($(featuresCol))
        .metadata
        .getMetadata("ml_attr")
        .getLong("num_attrs")
        .asInstanceOf[Int]
    }catch{
      case x: NoSuchElementException => {
        dataset
          .schema($(featuresCol))
          .metadata
          .getLong("numFeatures")
          .asInstanceOf[Int]
      }
    }
  }

  /**
   * The schema of the output Dataset is the same as the input one.
   * 
   * @param schema Input schema.
   */
  override def transformSchema(schema: StructType): StructType = schema

  /**
   * Creates a copy of this instance.
   * 
   * @param extra  Param values which will overwrite Params in the copy.
   */
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}

trait ROSEParams extends Params {
  /** @group param */
  val minoProb: DoubleParam = new DoubleParam(this, "minoProb", 
    "Probability of a sample of the minority class.", 
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  val shrinkFactor: DoubleParam = new DoubleParam(this, "shrinkFactor", 
    "Shrink factor.", 
    ParamValidators.gt(0.0))

  val keepOrigSamples: BooleanParam = new BooleanParam(this, "keepOrigSamples",
    "If keeping original samples or creating a full synthetic dataset.")

  val majoRUS: BooleanParam = new BooleanParam(this, "majoRUS",
    "If do simple Random Undersampling to majority class.")

  val keepMajoSize: BooleanParam = new BooleanParam(this, "keepMajoSize",
    "If keeping majority size.")

  setDefault(minoProb -> 0.5, shrinkFactor -> 1, keepOrigSamples -> false, majoRUS -> false, keepMajoSize -> false)

  /** @group getParam */
  def getMinoProb: Double = $(minoProb)

  /** @group getParam */
  def getShrinkFactor: Double = $(shrinkFactor)

  /** @group getParam */
  def getKeepOrigSamples: Boolean = $(keepOrigSamples)

  /** @group getParam */
  def getMajoRUS: Boolean = $(majoRUS)

  /** @group getParam */
  def getKeepMajoSize: Boolean = $(keepMajoSize)
}