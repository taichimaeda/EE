	c�~��C@c�~��C@!c�~��C@	]��*�@]��*�@!]��*�@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6c�~��C@c����=@1�����& @A�N��Z�?I!�Ky �?Yd�C�7�?*	֣p=
�l@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�?��?!Z��V�Q@)�Y�h9��?1+Y���P@:Preprocessing2F
Iterator::ModelG�,Ҥ?!n.{�%�1@)��A�p�?1G��_�,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��N���?!#
�+= @))[$�F�?19g!L��@:Preprocessing2U
Iterator::Model::ParallelMapV2�7U��?!Vʩ��@)�7U��?1Vʩ��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipE�e�?�?!e4����T@)a��+ey?1���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice$�����w?!���\@)$�����w?1���\@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��	�yk?!.�ګ�"�?)��	�yk?1.�ګ�"�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��9��?!_�6���Q@)�
�.�f?1�T-�q�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9\��*�@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	c����=@c����=@!c����=@      ��!       "	�����& @�����& @!�����& @*      ��!       2	�N��Z�?�N��Z�?!�N��Z�?:	!�Ky �?!�Ky �?!!�Ky �?B      ��!       J	d�C�7�?d�C�7�?!d�C�7�?R      ��!       Z	d�C�7�?d�C�7�?!d�C�7�?JGPUY\��*�@b 