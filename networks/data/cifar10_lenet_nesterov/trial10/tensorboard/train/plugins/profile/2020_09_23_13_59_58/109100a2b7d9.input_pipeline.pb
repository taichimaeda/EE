	�#�@jB@�#�@jB@!�#�@jB@	��F+`�?��F+`�?!��F+`�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�#�@jB@g�ba��;@1�1�%$ @A1� O!�?I?���?Y�6T��7�?*	!�rh�-S@2F
Iterator::ModelaP���b�?!��3���I@)8K�rJ�?1�9�:��C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���ɩ��?!��U�l6@)�Q��?1���?�1@:Preprocessing2U
Iterator::Model::ParallelMapV2�an��?!�z%(@)�an��?1�z%(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate!?�nJ�?!'M-0@)��:��T~?1I��jTN#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipN�����?!h'�}8H@)�z�2Q�t?1ڌT�S@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����(@t?!�ߏ�@)����(@t?1�ߏ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,�,�}l?!'(�-�"@),�,�}l?1'(�-�"@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapu��?!F���#3@)zUg��c?10��#W@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��F+`�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	g�ba��;@g�ba��;@!g�ba��;@      ��!       "	�1�%$ @�1�%$ @!�1�%$ @*      ��!       2	1� O!�?1� O!�?!1� O!�?:	?���??���?!?���?B      ��!       J	�6T��7�?�6T��7�?!�6T��7�?R      ��!       Z	�6T��7�?�6T��7�?!�6T��7�?JGPUY��F+`�?b 