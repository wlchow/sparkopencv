package com.cloudera.se.wchow.sparkopencv

import java.io.File
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IOUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkFiles
import org.bytedeco.javacpp.opencv_core.Mat
import org.bytedeco.javacpp.opencv_core.RectVector
import org.bytedeco.javacpp.opencv_core.Scalar
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgcodecs.imwrite
import org.bytedeco.javacpp.opencv_imgproc.rectangle
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier

// This will download the image locally to the node and then process it
// For Testing only, do not use this one
object SparkOpenCVLocal {
  def main (args: Array[String]){
    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val timestamp1 = System.currentTimeMillis()

    // The 1st argument is the path of the classifier.
    // The 2nd argument is the directory where the input images are located.
    // The optional 3rd argument is the directory where the output images will be written.
    if (args.length < 2 || args.length > 3) {
      println("usage: <classifier-path> <input-directory> [output-directory]")
      System.exit(-1)
    }
    val classifierPath = args(0)
    val inputDirName = args(1)
    val outputDirName = if (args.length == 2) inputDirName else args(2)


    // The classifier file is on HDFS and needs to be a local in order for OpenCV to use it
    // so we need to call addFile which will add a file to be downloaded with this Spark job on every node.
    sc.addFile(classifierPath)
    // It will be indexed using the fileName so that you can retrieve it later using SparkFiles.get
    val fileName = classifierPath.substring(classifierPath.lastIndexOf("/") + 1)

    // Go through the list of HDFS files
    val inputDir = new Path(inputDirName)
    val fs = inputDir.getFileSystem(new Configuration())
    val files = fs.listStatus(inputDir)
    val filePaths = files.map(_.getPath().toString())
    println("Number of files found: " + filePaths.length)

    // Using Spark to go scan the HDFS directory and then use OpenCV on each executor node
    sc.parallelize(filePaths).foreach({
      processImage(_, fileName, outputDirName)
    })

    val timestamp2 = System.currentTimeMillis()

    println("\n\n")
    println("The output files can be found at " + outputDirName)
    println("Total time: " + (timestamp2-timestamp1) + " milliseconds.")
    println("\n\n")
  }


  def processImage(inPath: String, classifierFileName: String, outputDir: String): String = {

    // This will be called in the Spark executor

    // Get the absolute path of a file added through SparkContext.addFile()
    val classifierLocalFullPath = SparkFiles.get(classifierFileName)

    /*
    println("WCHOW Processing image file inPath: " + inPath)
    println("WCHOW classifierFileName: " + classifierFileName)
    println("WCHOW classifierLocalFullPath: " + classifierLocalFullPath)
    println("WCHOW outputDir: " + outputDir)
    */

    // OpenCV imread and imwrite expects that the files be local. i.e. it cannot process files off HDFS directly
    // Need to download the file off HDFS before it can be processed locally by OpenCV
    // Download the image file locally to the the Spark executor node and then process it
    val extension = inPath.substring(inPath.lastIndexOf("."))
    val localIn = File.createTempFile("pic_", extension, new File("/tmp/"))
    //println("WCHOW localIn.getPath():" + localIn.getPath())
    copyFile(inPath, "file://" + localIn.getPath())

    val inImg = imread(localIn.getPath())
    val detector = new CascadeClassifier(classifierLocalFullPath)
    val outlinedImg = detectFeatures(inImg, detector)

    val inMatSize = inImg.total() * inImg.elemSize()
    println("WCHOW inMatSize: " + inMatSize)
    val outMatSize = outlinedImg.total() * outlinedImg.elemSize()
    println("WCHOW outMatSize: " + outMatSize)

    // Write the output image to a local temp file
    val localOut = File.createTempFile("out_", extension, new File("/tmp/"))
    //println("WCHOW localOut.getPath():" + localOut.getPath())
    imwrite(localOut.getPath(),outlinedImg)

    // Copy the file to the HDFS output directory
    val fileName = inPath.substring(inPath.lastIndexOf("/") + 1)
    val name = fileName.substring(0, fileName.lastIndexOf("."))
    val outPath = outputDir + name +"_output" + extension
    //println("WCHOW outPath: " + outPath)
    copyFile("file://" + localOut.getPath(), outPath)

    // delete the temporary files on the Spark executor node
    localIn.delete()
    localOut.delete()

    return outPath
  }


  // Outlines the detected features in a given Mat.
  def detectFeatures(img: Mat, detector: CascadeClassifier): Mat = {
    val features = new RectVector()
    detector.detectMultiScale(img, features)
    val numFeatures = features.size().toInt
    val outlined = img.clone()


    // Draws the rectangles on the detected features.
    val green = new Scalar(0, 255, 0, 0)
    for (f <- 0 until numFeatures) {
      val currentFeature = features.get(f)
      rectangle(outlined, currentFeature, green)
    }
    return outlined
  }


  // Copies file to and from given locations.
  // The to and from locations can be local (file:///) or HDFS (hdfs:///)
  def copyFile(from: String, to: String) {
    val conf = new Configuration()
    val fromPath = new Path(from)
    val toPath = new Path(to)
    val is = fromPath.getFileSystem(conf).open(fromPath)
    val os = toPath.getFileSystem(conf).create(toPath)
    IOUtils.copyBytes(is, os, conf)
    is.close()
    os.close()
  }


}
