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

package org.ubu.admirable.exec
import org.apache.spark.ml.classification._
import org.apache.spark.ml.Estimator
import org.apache.spark.mllib.util.MLUtils
import org.ubu.admirable.util.LibSVMReader
import org.apache.spark.sql.functions.rand
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.param.Param
import java.lang.reflect.MalformedParametersException
import java.security.InvalidParameterException
import org.apache.spark.ml.param.ParamMap
import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.ml.Model
import org.apache.spark.sql.functions._
import org.apache.spark.ml.param.LongParam
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.param.BooleanParam
import java.util.Calendar
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.util.LabelUtils

/**
 * An experiment launcher that runs experiments for resampling
 * and then training an ensemble with the resampled (balanced)
 * data set.
 * 
 * Usage:
 * spark-submit --class org.ubu.admirable.exec.ResampleEnsembleLauncher \
 * ./ensemble-classification-imbalance-bigdata.jar [Resampler class name] \
 * [Ensemble class name] "[string containing resampler parameters]" \
 * "[string containing ensemble parameters]" [weight instances? true/false] \
 * [random seed] [number of repetitions] [number of folds] \
 * [libSVM data set path] [output results path]
 * 
 * Example: Random Forest without resampling
 * spark-submit --class org.ubu.admirable.exec.ResampleEnsembleLauncher 
 * ./ensemble-classification-imbalance-bigdata.jar NoResampler 
 * RandomForestClassifier "" "numTrees=100,maxDepth=5" true 46 5 2 
 * datasets/susy_ir4_nonorm.libsvm susy_ir4_nonorm_none_rf_100
 * 
 * @author Mario Juez-Gil <mariojg@ubu.es>
 */
object ResampleEnsembleLauncher {
  
  def main(args: Array[String]) {
    val session = SparkSession.builder.appName("Experiment Launcher")
      .getOrCreate()
    session.sparkContext.setLogLevel("WARN")

    val resamplerClass = args(0)
    val classifierClass = args(1)
    val resamplerParams = args(2)
    val classifierParams = args(3)
    val weightInstances = args(4).toBoolean
    val seed = args(5).toLong
    val cvReps = args(6).toInt
    val cvFolds = args(7).toInt
    val inputDataset = args(8)
    val outputPath = args(9)
    val skip = try {
      args(10).toInt
    }catch{
      case _: Exception => 0
    }

    val pkgPrefix = "org.apache.spark.ml."
    val resampler = Class
      .forName(pkgPrefix + "instance." + resamplerClass)
      .newInstance()
      .asInstanceOf[Transformer]
    val classifier = Class
      .forName(pkgPrefix + "classification." + classifierClass)
      .newInstance()
      .asInstanceOf[Estimator[_]]

    val resamplerParamMap = toParamMap(resamplerParams, resampler, seed)
    val classifierParamMap = toParamMap(classifierParams, classifier, seed)

    val randCol = rand(seed)
    val rnd = new Random(seed)

    val dataset = LibSVMReader.libSVMToML(inputDataset, session)
      .orderBy(randCol)
      .select(col("label"), col("features"))

    val minMaxScaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("sf")
    val model = minMaxScaler.fit(dataset)
    val scaledDataSet = model.transform(dataset)
      .select("label", "sf")
      .withColumnRenamed("sf", "features")

    val schema = dataset.schema

    val randomSuffix = Random.alphanumeric.take(10).mkString // to avoid existing folders

    /*
     * Results is a DataFrame with the following Schema:
     * -----------------------------------------------------------------------
     * | true | predicted | fold | repetition | train_time | prediction_time |
     * -----------------------------------------------------------------------
     */
    var skipped = 0
    val results = (1 to cvReps).map { case(rep) =>
      val run = (trainSplit: RDD[Row], testSplit: RDD[Row], foldID: Int) => {
        val dT = Calendar.getInstance()
        System.out.println("[" + dT.getTime() + "] Repetition " + rep + " / Fold " + foldID)
        // training
        val trainDS = session.createDataFrame(trainSplit, schema).cache
        
        val trainDF = if (weightInstances){
          // weightcol
          val classCounts = LabelUtils.getLabelsInfo(trainDS, "label")
          val (minorityLabel, minoritySize) = classCounts(0)
          val (majorityLabel, majoritySize) = classCounts(1)
          val totalSize = minoritySize + majoritySize
          val minoTrainDF = trainDS
            .filter(col("label") === minorityLabel)
            .select(col("label"), col("features"), lit((totalSize / 2.0) / minoritySize).as("weight"))
          val majoTrainDF = trainDS
            .filter(col("label") === majorityLabel)
            .select(col("label"), col("features"), lit((totalSize / 2.0) / majoritySize).as("weight"))
          majoTrainDF.union(minoTrainDF).toDF
        }else{
          trainDS.toDF
        }
        
        trainDF.count // force caches
        val startTraining = System.nanoTime()

        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(trainDF)
        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labelsArray(0))

        val pipeline = new Pipeline()
          .setStages(Array(labelIndexer, classifier, labelConverter))

        val resampledDF = resampler.transform(trainDF, resamplerParamMap)
        val model = pipeline.fit(resampledDF, classifierParamMap)
        val endTraining = System.nanoTime()
        trainDF.unpersist(blocking = true)

        val testDF = session.createDataFrame(testSplit, schema).cache
        testDF.count
        val startTest = System.nanoTime()
        val testResults = model.transform(testDF).cache()
        val numPredictions = testResults.count() // for triggering cache
        val endTest = System.nanoTime()
        testDF.unpersist(blocking = true)

        val trainTime = endTraining - startTraining
        val predictionTime = (endTest - startTest) / numPredictions

        testResults
          .select(col("label"), col("predictedLabel"))
          .withColumnRenamed("label", "true")
          .withColumnRenamed("predictedLabel", "predicted")
          .withColumn("fold", lit(foldID))
          .withColumn("repetition", lit(rep))
          .withColumn("train_time", lit(trainTime))
          .withColumn("prediction_time", lit(predictionTime))
      }

      if(cvFolds > 1){
        val splits = MLUtils.kFold(scaledDataSet.toDF.rdd, cvFolds, rnd.nextLong)
        splits.zipWithIndex.map { case (params) =>
          if(skipped < skip){
            skipped += 1
            scaledDataSet.sparkSession.emptyDataFrame
          }else{
            val foldResults = run(params._1._1, params._1._2, params._2 + 1)
            foldResults.repartition(28).write.format("csv").save(outputPath + "_r" + rep + "_" + randomSuffix + "_f" + (params._2 + 1))
            foldResults
          }
        }.reduce { (allFoldsDF, currentFoldDF) =>
          if(!allFoldsDF.isEmpty && !currentFoldDF.isEmpty){
            allFoldsDF.union(currentFoldDF)
          }else{
            scaledDataSet.sparkSession.emptyDataFrame
          }
        }
      }else{
        if(skipped < skip){
          skipped += 1
          scaledDataSet.sparkSession.emptyDataFrame
        }else{
          run(scaledDataSet.toDF.rdd, scaledDataSet.toDF.rdd, 1)
        }
      }
    }.reduce { (resultsDF, partialResultsDF) =>
      if(!resultsDF.isEmpty && !partialResultsDF.isEmpty){
        resultsDF.union(partialResultsDF)
      }else{
        scaledDataSet.sparkSession.emptyDataFrame
      }
    }

    if(!results.isEmpty){
      results.repartition(28).write.format("csv").save(outputPath + "_" + randomSuffix)
    }
  }

  /**
    * Creates a ParamMap from a string containing the parameters.
    *
    * @param params A string containing the parameters and their values.
    *               Different parameters are separated by a comma.
    * @param obj The object with params (a pipeline stage)
    * @param seed Random seed
    * @return The param map.
    */
  def toParamMap(params: String, obj: PipelineStage, seed: Long): ParamMap = {
    val paramMap = new ParamMap()

    if(params.size > 0){
      if(obj.hasParam("seed")){
        val seedParam = obj.getParam("seed")
        paramMap.put(seedParam.asInstanceOf[LongParam]->(seed))
      }

      params.split(",").map { case(p) =>
        val paramData = p.split("=")
        if(paramData.size != 2) throw new MalformedParametersException
        val paramName = paramData(0)
        val paramValue = paramData(1)

        if(!obj.hasParam(paramName)) throw new InvalidParameterException
        val param = obj.getParam(paramName)
        
        try {
          if(paramValue.matches("\\d+[lL]")){
            paramMap.put(param.asInstanceOf[LongParam]->(paramValue.init.toLong))
          }else{
            paramMap.put(param.asInstanceOf[IntParam]->(paramValue.toInt))
          }
        } catch {
          case _: Exception => try {
            paramMap.put(param.asInstanceOf[DoubleParam]->(paramValue.toDouble))
          } catch {
            case _: Exception => try {
              paramMap.put(param.asInstanceOf[BooleanParam]->(paramValue.toBoolean))
            } catch {
              case _: Exception => print(paramValue); paramMap.put(param->(paramValue))
            }
          }
        }
      }
    } 

    paramMap
  }
}