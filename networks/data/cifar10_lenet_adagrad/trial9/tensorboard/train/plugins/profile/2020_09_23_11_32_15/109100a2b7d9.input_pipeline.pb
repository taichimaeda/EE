	�a���A@�a���A@!�a���A@	���W��?���W��?!���W��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�a���A@��Xm�;@1l%t��) @A�! _B�?I&�`6�?YP�mp��?*	*��ΧT@2F
Iterator::Modelܸ���Ф?!��曆�H@)�"�Ƥ�?1�#�i�B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ�+�ϒ?!e2D�:;6@)�P�B�y�?1�Y�2@:Preprocessing2U
Iterator::Model::ParallelMapV2�
����?!��*�s�'@)�
����?1��*�s�'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�~K�|�?!�5dyeI@)�W�B�?1+��"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��� �?!]���j0@)��x'?15:\i"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceC��w?!>r��B@)C��w?1>r��B@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor<hv�[�h?!��p�	@)<hv�[�h?1��p�	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��?�0�?!����"3@)P�<�e?1��)S�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��W��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Xm�;@��Xm�;@!��Xm�;@      ��!       "	l%t��) @l%t��) @!l%t��) @*      ��!       2	�! _B�?�! _B�?!�! _B�?:	&�`6�?&�`6�?!&�`6�?B      ��!       J	P�mp��?P�mp��?!P�mp��?R      ��!       Z	P�mp��?P�mp��?!P�mp��?JGPUY��W��?b 