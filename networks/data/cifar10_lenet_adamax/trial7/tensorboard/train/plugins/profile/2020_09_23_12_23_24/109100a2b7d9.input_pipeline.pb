	3�f�mC@3�f�mC@!3�f�mC@	�.3��?�.3��?!�.3��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails63�f�mC@ǡ~��=@1O�}�v @A����	��?I�y�0��?Y�i2�m��?*	�p=
��U@2F
Iterator::Model�aL�{)�?!�.�?lF@)� �6qr�?1��R�|A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateD�ͩd �?!���>_�:@)�ْUn�?1�]4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat㥛� ��?!QW��4@)���_vO�?1�FA��0@:Preprocessing2U
Iterator::Model::ParallelMapV2j��{��?!��p��#@)j��{��?1��p��#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceު�PMIv?!ح	��@)ު�PMIv?1ح	��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��9��˨?!��q��K@)���"Rs?1�տ|@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorlxz�,Cl?!���vn@)lxz�,Cl?1���vn@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���<�?!TO[Լ�<@)��ǘ��`?1~Fƫ�r@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 76.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�.3��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ǡ~��=@ǡ~��=@!ǡ~��=@      ��!       "	O�}�v @O�}�v @!O�}�v @*      ��!       2	����	��?����	��?!����	��?:	�y�0��?�y�0��?!�y�0��?B      ��!       J	�i2�m��?�i2�m��?!�i2�m��?R      ��!       Z	�i2�m��?�i2�m��?!�i2�m��?JGPUY�.3��?b 