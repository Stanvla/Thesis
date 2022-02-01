import logging
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import os


def do_my_logging(log_msg):
    # logger = logging.getLogger('__FILE__')
    print('log_msg = {}'.format(log_msg))
    
merged_dir = '/lnet/express/work/people/stankov/alignment/mfccs/merged'
# logging.basicConfig(filename=f"pyspark_example.log", filemode='w', level=logging.DEBUG, format='[%(levelname)s] %(asctime)s - %(message)s', datefmt='%H:%M:%S '
#                                                                                                                                               '%d.%m.%Y')
spark = SparkSession.builder.appName('name').getOrCreate()
spark.sparkContext.setLogLevel("DEBUG")
do_my_logging('Session started')

# reading all merged csv_files
dataset = None
do_my_logging('Iterating over merged dir')
for i, csv_file in enumerate(os.listdir(merged_dir)):

    if not csv_file.endswith('.csv'):
        continue

    do_my_logging(f'{i:3}/{len(os.listdir(merged_dir))} Loading {csv_file}')

    full_path = os.path.join(merged_dir, csv_file)
    tmp_df = spark.read.csv(full_path, header=True, inferSchema=True)

    for i in range(39):
        col_name = f'{i}'
        tmp_df = tmp_df.withColumn(col_name, tmp_df[col_name].cast('float'))

    if dataset is None:
        dataset = tmp_df
    else:
        dataset = dataset.union(tmp_df)


assembler = VectorAssembler(inputCols=[f'{i}' for i in range(39)], outputCol='features')
final_data = assembler.transform(dataset)

new_data = None
# clusters = [2, 5]
clusters = [2, 5, 10, 25, 50, 100, 250, 500]
for k in clusters:
    do_my_logging(f'Kmeans with k={k}')
    kmeans = KMeans(featuresCol='features', k=k, initSteps=5, seed=0xDEAD)
    km_model = kmeans.fit(final_data)

    if new_data is None:
        new_data = km_model.transform(final_data)
    else:
        new_data = km_model.transform(new_data)

    new_data = new_data.withColumnRenamed('prediction', f'km{k}')

    do_my_logging(f'new columns{new_data.columns}')
    do_my_logging(f'wssse {km_model.computeCost(final_data):.3f}')

do_my_logging('finished clustering for all k')
#
#
# # save new_data
do_my_logging('saving the results')
output_dir = os.path.join(merged_dir, 'clustering')
new_data.select([f'km{k}' for k in clusters] + ['path']).write.csv(output_dir, header=True, mode='overwrite')

