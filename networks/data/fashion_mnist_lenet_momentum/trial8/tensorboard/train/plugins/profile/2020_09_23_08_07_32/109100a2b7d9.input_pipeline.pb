	'3�Vz�A@'3�Vz�A@!'3�Vz�A@	��A����?��A����?!��A����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6'3�Vz�A@^G��(<@1�a�@AS]���?I�~�d�p�?Y�g@��?*	��/ݴY@2F
Iterator::Model`L8��?!�*�\��D@)r��7��?1�:�nw<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat+N�f��?! &;�$<@)p|�%�?1L���r�8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��3w��?!p�8q8@)�!p$�`�?1R��[fg2@:Preprocessing2U
Iterator::Model::ParallelMapV2��jGq��?!�׭�8)@)��jGq��?1�׭�8)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicei���ny?!�w�F'@)i���ny?1�w�F'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziptb�c�?!�l�vM@)��k��r?10W2z�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorڍ>�m?!���ꨌ@)ڍ>�m?1���ꨌ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap �t����?!/���W:@)��	��`?1r�4�k�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��A����?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	^G��(<@^G��(<@!^G��(<@      ��!       "	�a�@�a�@!�a�@*      ��!       2	S]���?S]���?!S]���?:	�~�d�p�?�~�d�p�?!�~�d�p�?B      ��!       J	�g@��?�g@��?!�g@��?R      ��!       Z	�g@��?�g@��?!�g@��?JGPUY��A����?b 