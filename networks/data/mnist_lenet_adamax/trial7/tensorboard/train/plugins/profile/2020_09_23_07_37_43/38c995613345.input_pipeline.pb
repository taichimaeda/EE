	Y��;�B@Y��;�B@!Y��;�B@	���Q��?���Q��?!���Q��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Y��;�B@��N�X>@1��+� @A@��
/�?I[$�F��?YR_�vj��?*	��ʡm^@2F
Iterator::Model��P�n�?!ʡ�l�1Q@)9d�bӪ?1�곜	�E@:Preprocessing2U
Iterator::Model::ParallelMapV2:�%��?!��2z�9@):�%��?1��2z�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���O=�?!��D-@)�S��э?1�~D�i�'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[��X��?!&�o";}$@)*��g\8�?1�n�H@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�|]��t�?!�xeL�8?@)��t �u?1��lz�a@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec����r?!L�T\�@)c����r?1L�T\�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr��	�j?!#b�z�a@)r��	�j?1#b�z�a@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��k]j��?!Ln��G|(@)
�F�c?1/���d��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 81.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9���Q��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��N�X>@��N�X>@!��N�X>@      ��!       "	��+� @��+� @!��+� @*      ��!       2	@��
/�?@��
/�?!@��
/�?:	[$�F��?[$�F��?![$�F��?B      ��!       J	R_�vj��?R_�vj��?!R_�vj��?R      ��!       Z	R_�vj��?R_�vj��?!R_�vj��?JGPUY���Q��?b 