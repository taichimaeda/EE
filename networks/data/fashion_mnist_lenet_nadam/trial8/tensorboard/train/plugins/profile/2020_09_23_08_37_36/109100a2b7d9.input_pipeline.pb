	���zi M@���zi M@!���zi M@	ى3-��?ى3-��?!ى3-��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���zi M@�)�D/�H@1�i�L�@A���Bt�?I!O!W��?YN����?*	-��利�@2Z
#Iterator::Model::ParallelMapV2::Zip�>��?!̜�T��V@)�'Hlw�?1<��4T@:Preprocessing2F
Iterator::Model�@׾�^�?!��Z�b#@)�3��E`�?1�~++P� @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatec����?!����]b@)YLl>��?1R�	��@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�.�H��?!QoP#L@)�&��鳓?12J=1U@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�lt�Oq�?!p�V	g� @)�lt�Oq�?1p�V	g� @:Preprocessing2U
Iterator::Model::ParallelMapV2P6�
�r�?!3�<{��?)P6�
�r�?13�<{��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice%���wz?!�2�gFX�?)%���wz?1�2�gFX�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��0E�4�?!�9�*��@)���`�Hd?1ՒiY6�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9؉3-��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�)�D/�H@�)�D/�H@!�)�D/�H@      ��!       "	�i�L�@�i�L�@!�i�L�@*      ��!       2	���Bt�?���Bt�?!���Bt�?:	!O!W��?!O!W��?!!O!W��?B      ��!       J	N����?N����?!N����?R      ��!       Z	N����?N����?!N����?JGPUY؉3-��?b 