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
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline

/**
 * An experiment launcher that runs experiments for training an ensemble that
 * resamples (balances) the data set for each base classifier.
 * 
 * Usage:
 * spark-submit --class org.ubu.admirable.exec.EnsembleResampleLauncher \
 * ./ensemble-classification-imbalance-bigdata.jar [Ensemble class name] \
 * "[string containing ensemble parameters]" [weight instances? true/false] \
 * [random seed] [number of repetitions] [number of folds] \
 * [libSVM data set path] [output results path]
 * 
 * Example: Random Forest without resampling
 * spark-submit --class org.ubu.admirable.exec.EnsembleResampleLauncher 
 * ./ensemble-classification-imbalance-bigdata.jar ImbalanceGBTClassifier 
 * "randomUnderSampling=true,percSampling=0,maxIter=100,featureSubsetStrategy=auto,maxDepth=5"
 * true 46 5 2 datasets/susy_ir4_nonorm.libsvm susy_ir4_nonorm_none_imbgbt_100
 * 
 * @author Mario Juez-Gil <mariojg@ubu.es>
 */
object EnsembleResampleLauncher {
  
  def main(args: Array[String]) {
    val session = SparkSession.builder.appName("Experiment Launcher")
      .getOrCreate()
    session.sparkContext.setLogLevel("WARN")

    val classifierClass = args(0)
    val classifierParams = args(1)
    val weightInstances = args(2).toBoolean
    val seed = args(3).toLong
    val cvReps = args(4).toInt
    val cvFolds = args(5).toInt
    val inputDataset = args(6)
    val outputPath = args(7)
    val skip = try {
      args(8).toInt
    }catch{
      case _: Exception => 0
    }

    val pkgPrefix = "org.apache.spark.ml.classification."
    val classifier = Class
      .forName(pkgPrefix + classifierClass)
      .newInstance()
      .asInstanceOf[Estimator[_]]

    val paramMap = new ParamMap

    if(classifier.hasParam("seed")){
      val seedParam = classifier.getParam("seed")
      paramMap.put(seedParam.asInstanceOf[LongParam]->(seed))
    }

    classifierParams.split(",").map { case(p) =>
      val paramData = p.split("=")
      if(paramData.size != 2) throw new MalformedParametersException
      val paramName = paramData(0)
      val paramValue = paramData(1)

      if(!classifier.hasParam(paramName)) throw new InvalidParameterException
      val param = classifier.getParam(paramName)
      
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

    val randCol = rand(seed)
    val rnd = new Random(seed)

    val dataset = LibSVMReader.libSVMToML(inputDataset, session)
      .orderBy(randCol)
      .select(col("label"), col("features"))
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
        val trainDF = session.createDataFrame(trainSplit, schema).cache
        trainDF.count
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

        val model = pipeline.fit(trainDF, paramMap)
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
          .select(col("label"), col("prediction"))
          .withColumnRenamed("label", "true")
          .withColumnRenamed("prediction", "predicted")
          .withColumn("fold", lit(foldID))
          .withColumn("repetition", lit(rep))
          .withColumn("train_time", lit(trainTime))
          .withColumn("prediction_time", lit(predictionTime))
      }

      if(cvFolds > 1){
        val splits = MLUtils.kFold(dataset.toDF.rdd, cvFolds, rnd.nextLong)
        splits.zipWithIndex.map { case (params) =>
          if(skipped < skip){
            skipped += 1
            dataset.sparkSession.emptyDataFrame
          }else{
            val foldResults = run(params._1._1, params._1._2, params._2 + 1)
            foldResults.repartition(28).write.format("csv").save(outputPath + "_r" + rep + "_" + randomSuffix + "_f" + (params._2 + 1))
            foldResults
          }
        }.reduce { (allFoldsDF, currentFoldDF) =>
          if(!allFoldsDF.isEmpty && !currentFoldDF.isEmpty){
            allFoldsDF.union(currentFoldDF)
          }else{
            dataset.sparkSession.emptyDataFrame
          }
        }
      }else{
        if(skipped < skip){
          skipped += 1
          dataset.sparkSession.emptyDataFrame
        }else{
          run(dataset.toDF.rdd, dataset.toDF.rdd, 1)
        }
      }
    }.reduce { (resultsDF, partialResultsDF) =>
      if(!resultsDF.isEmpty && !partialResultsDF.isEmpty){
        resultsDF.union(partialResultsDF)
      }else{
        dataset.sparkSession.emptyDataFrame
      }
    }

    if(!results.isEmpty){
      results.repartition(28).write.format("csv").save(outputPath + "_" + randomSuffix)
    }
  }

}