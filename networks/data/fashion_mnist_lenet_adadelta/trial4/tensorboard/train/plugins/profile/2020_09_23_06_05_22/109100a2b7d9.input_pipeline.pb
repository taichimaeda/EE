	�u7A@�u7A@!�u7A@	/e�$���?/e�$���?!/e�$���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�u7A@wKr���:@1��&���@A�{G�	1�?I�ݓ��Z�?YF��(&o�?*	�I+�R@2F
Iterator::Model��&2s��?!8 ���I@)I�Q}�?1<	k���C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��~�7�?!�7m� w2@)��n�;2�?1�ܽn7-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�NҌ?!�o9S��2@)���K��?1P���Q,@:Preprocessing2U
Iterator::Model::ParallelMapV2R�o&��?!���aO&@)R�o&��?1���aO&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip5{���?!��O�LxH@)���3.|?1��L�+q"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicef��(ϼl?!/�T�@)f��(ϼl?1/�T�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�A��h?!�Kr�)�@)�A��h?1�Kr�)�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapU�]=�?!�i�A5@)W!�'�>]?1�їR�#@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9/e�$���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	wKr���:@wKr���:@!wKr���:@      ��!       "	��&���@��&���@!��&���@*      ��!       2	�{G�	1�?�{G�	1�?!�{G�	1�?:	�ݓ��Z�?�ݓ��Z�?!�ݓ��Z�?B      ��!       J	F��(&o�?F��(&o�?!F��(&o�?R      ��!       Z	F��(&o�?F��(&o�?!F��(&o�?JGPUY/e�$���?b 