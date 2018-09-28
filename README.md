# Apache Spark and OpenCV

Using OpenCV with Apache Spark for image detection  
  
Tried this on CentOS 7.3, CDH 5.13, Spark 2.2

## Prerequisites

1) Install gtk2
Do this on ALL the nodes
```yum install gtk2```

2) Put the classifier and image files on HDFS  
```
[root@c168 sparkopencv]# pwd
/root/IdeaProjects/sparkopencv
[root@c168 sparkopencv]# hadoop fs -put classifiers/ /user/william/
[root@c168 sparkopencv]# hadoop fs -ls -R /user/william/classifiers
-rw-r--r--   3 william william     676709 2017-11-21 15:59 /user/william/classifiers/haarcascade_frontalface_alt.xml
-rw-r--r--   3 william william     540616 2017-11-21 15:59 /user/william/classifiers/haarcascade_frontalface_alt2.xml
-rw-r--r--   3 william william     930127 2017-11-21 15:59 /user/william/classifiers/haarcascade_frontalface_default.xml
[root@c168 sparkopencv]# hadoop fs -put images04 /user/william
[root@c168 sparkopencv]# hadoop fs -ls -R /user/william/images04
```
Create the output directory on HDFS
```
[root@c168 sparkopencv]# hadoop fs -mkdir /user/william/out
```

### Cleanup between runs
```hadoop fs -rm -skipTrash /user/william/out/*```

### Examples of how to run this
#### Run it in YARN mode Client Mode (default)
```
spark2-submit --class com.cloudera.se.wchow.sparkopencv.SparkOpenCV --master yarn target/sparkopencv-1.0-SNAPSHOT.jar hdfs:///user/william/classifiers/haarcascade_frontalface_alt.xml hdfs:///user/william/images04 hdfs:///user/william/out/
```
#### Run it in YARN mode Cluster Mode
```
spark2-submit --class com.cloudera.se.wchow.sparkopencv.SparkOpenCV --master yarn --deploy-mode cluster target/sparkopencv-1.0-SNAPSHOT.jar hdfs:///user/william/classifiers/haarcascade_frontalface_alt.xml hdfs:///user/william/images04 hdfs:///user/william/out/
```
#### Run it with a bunch of parameters
```
spark2-submit --class com.cloudera.se.wchow.sparkopencv.SparkOpenCV --master yarn --num-executors 4 --executor-cores 4 --executor-memory 2G --conf spark.yarn.executor.memoryOverhead=2048 target/sparkopencv-1.0-SNAPSHOT.jar hdfs:///user/william/classifiers/haarcascade_frontalface_alt.xml hdfs:///user/william/images04 hdfs:///user/william/out/
```
