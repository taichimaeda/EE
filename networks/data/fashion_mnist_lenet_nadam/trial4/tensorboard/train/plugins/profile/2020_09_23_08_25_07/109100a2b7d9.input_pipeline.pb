	�LjhHN@�LjhHN@!�LjhHN@	��0���?��0���?!��0���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�LjhHN@��7�I@1�'�ڑ@A�	MK�?I���>9
�?Y�r.�U�?*	�n��*T@2F
Iterator::Model4h��b�?!�m�}��I@)���5[�?1P�?�E@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate,��̰�?!C��p�j5@);�bFx�?1 f���0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatE�u����?!��^��/3@)�C p�?1�"B*	b/@:Preprocessing2U
Iterator::Model::ParallelMapV2�̱���?!��ƹ�#@)�̱���?1��ƹ�#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2v�Kp�?!?�	�$H@)��S��q?1f�1I�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicet(CUL�o?!����'@)t(CUL�o?1����'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�����g?!9�9k�@)�����g?19�9k�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�,��o��?!�P���7@)��z`?1�J�Sҁ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 85.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��0���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��7�I@��7�I@!��7�I@      ��!       "	�'�ڑ@�'�ڑ@!�'�ڑ@*      ��!       2	�	MK�?�	MK�?!�	MK�?:	���>9
�?���>9
�?!���>9
�?B      ��!       J	�r.�U�?�r.�U�?!�r.�U�?R      ��!       Z	�r.�U�?�r.�U�?!�r.�U�?JGPUY��0���?b 