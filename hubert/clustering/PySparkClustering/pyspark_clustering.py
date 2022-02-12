import datetime
import os

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, isnan, when, count


def do_my_logging(log_msg):
    # logger = logging.getLogger('__FILE__')
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('{} :: {}'.format(now, log_msg))

def check_nan(dataframe):
    nan_cnt = dataframe.select([
        count(
            when(
                col(c).contains('None') | col(c).contains('NULL') | (col(c) == '') | col(c).isNull() | isnan(c),
                c
            )
        ).alias(c) for c in dataframe.columns
    ]).groupBy().sum().collect()
    return nan_cnt


merged_dir = '/lnet/express/work/people/stankov/alignment/mfcc'
spark = SparkSession.builder.appName('name').getOrCreate()
spark.sparkContext.setLogLevel("DEBUG")
do_my_logging('Session started')

# reading all merged csv_files
do_my_logging('Iterating over merged dir')
dataset = spark.read.csv(merged_dir, header=True, inferSchema=True)

do_my_logging('checking nans')
nans = check_nan(dataset)
if nans != 0:
    do_my_logging(f'nans detected = {nans}')

final_data = VectorAssembler(inputCols=[f'{i}' for i in range(39)], outputCol='features').transform(dataset)

new_data = None
clusters = [5, 10, 15, 20, 25, 50, 75, 100, 200, 250, 300, 500]
do_my_logging(f'clusters are {clusters}')

for k in clusters:
    do_my_logging(f'Kmeans with k={k}')
    kmeans = KMeans(featuresCol='features', k=k, initSteps=5, seed=0xDEAD)
    km_model = kmeans.fit(final_data)

    if new_data is None:
        new_data = km_model.transform(final_data)
    else:
        new_data = km_model.transform(new_data)

    new_data = new_data.withColumnRenamed('prediction', f'km{k}')

    do_my_logging(f'>>> km={k}, wssse {km_model.computeCost(final_data):.3f}')

do_my_logging(f'new columns{new_data.columns}')

do_my_logging('finished clustering for all k')
#
#
# # save new_data
do_my_logging('saving the results')
# output_dir = os.path.join(merged_dir, 'clustering')
output_dir = os.path.join('/lnet/express/work/people/stankov/alignment/clustering_all')
new_data.select([f'km{k}' for k in clusters] + ['path']).write.csv(output_dir, header=True, mode='overwrite')
do_my_logging('done, exiting')
