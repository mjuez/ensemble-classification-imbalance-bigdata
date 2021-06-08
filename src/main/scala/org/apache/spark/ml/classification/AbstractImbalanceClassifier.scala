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

package org.apache.spark.ml.classification

import org.apache.spark.annotation.Since
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.RandomForestClassifierParams
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.Dataset
import scala.util.Random
import org.apache.spark.ml.util.LabelUtils
import org.apache.spark.sql.Row
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.ParamValidators
import org.apache.spark.sql.functions._
import java.{util => ju}

@Since("2.4.5")
abstract class AbstractImbalanceClassifier @Since("2.4.5")
  extends ProbabilisticClassifier[Vector, AbstractImbalanceClassifier, RandomForestClassificationModel]
  with AbstractImbalanceClassifierParams with DefaultParamsWritable {

  /** @group setParam */
  def setNumResamples(value: Int): this.type = set(numResamples, value)

  /** @group setParam */
  def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  def setSubsamplingRate(value: Double): this.type = 
    set(subsamplingRate, value)

  /** @group setParam */
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  def setMinInstancesPerNode(value: Int): this.type = 
    set(minInstancesPerNode, value)

  /** @group setParam */
  def setMinInfoGain(value: Double): this.type = 
    set(minInfoGain, value)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = 
    set(checkpointInterval, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setExpertParam */
  def setMaxMemoryInMB(value: Int): this.type = 
    set(maxMemoryInMB, value)

  /** @group setExpertParam */
  def setCacheNodeIds(value: Boolean): this.type = 
    set(cacheNodeIds, value)

  @Since("2.4.5")
  override def train(dataset: Dataset[_]): RandomForestClassificationModel = {
    val rand = new Random($(seed))
  
    val classCounts = LabelUtils.getLabelsInfo(dataset, $(labelCol))
    require(classCounts.size == 2, "A binary classification data set is required.")

    val numFeatures = dataset.select(col($(featuresCol))).first match {
      case Row(features: Vector) => features.size
    }

    val trees = (1 to $(numResamples)).flatMap( rs => {
      // in weka bagging does bootstrapping with replacement (true)
      val bootstrap = dataset.sample(true, $(subsamplingRate), rand.nextLong()).cache()
      val trainData = resample(bootstrap, rand).cache()
      bootstrap.unpersist()

      val rfc = new RandomForestClassifier()
        .setLabelCol($(labelCol))
        .setFeaturesCol($(featuresCol))
        .setNumTrees($(numTrees))
        .setSubsamplingRate(1.0)
        .setMaxDepth($(maxDepth))
        .setMaxBins($(maxBins))
        .setMinInstancesPerNode($(minInstancesPerNode))
        .setMinInfoGain($(minInfoGain))
        .setCheckpointInterval($(checkpointInterval))
        .setSeed(rand.nextLong())
        .setMaxMemoryInMB($(maxMemoryInMB))
        .setCacheNodeIds($(cacheNodeIds))
        .setFeatureSubsetStrategy($(featureSubsetStrategy))
        .fit(trainData)

      trainData.unpersist()
      val dT = ju.Calendar.getInstance()
      System.out.println("[" + dT.getTime() + "] Trained resample: " + rs)
      rfc.trees
    }).toArray
    
    new RandomForestClassificationModel(trees, numFeatures, 2)
  }

  def resample(dataset: Dataset[_], rand: Random): Dataset[_]
}

trait AbstractImbalanceClassifierParams extends RandomForestClassifierParams {

  /** @group param */
  final val numResamples: IntParam = new IntParam(this, "numResamples", 
    "Number of resamples of the input data", ParamValidators.gtEq(1))

  setDefault(numTrees -> 1, numResamples -> 100)

}