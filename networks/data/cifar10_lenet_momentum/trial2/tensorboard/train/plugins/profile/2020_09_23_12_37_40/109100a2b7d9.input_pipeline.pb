	�mR�X�C@�mR�X�C@!�mR�X�C@	
���Z��?
���Z��?!
���Z��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�mR�X�C@B�v��g>@1���r @Ai�G5���?I���@��?Y7P��|z�?*	|?5^�AX@2F
Iterator::Model��Ü�?!N��ޟ�H@)�Op���?1���9&C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�.4�i��?!oQ%@X�5@)�q�d��?1�T��0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat'N�w(
�?!���m+4@)+�`�?1s
-�{0@:Preprocessing2U
Iterator::Model::ParallelMapV2�OU��X�?!���	�}&@)�OU��X�?1���	�}&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip������?!�!`:I@)�릔w?1��i4ͻ@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�o
+Tt?!��K0�u@)�o
+Tt?1��K0�u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorʩ�ajKm?!���3|@)ʩ�ajKm?1���3|@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]��X32�?!�A3_Z8@)	��Lnd?1v��7�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 77.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9
���Z��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B�v��g>@B�v��g>@!B�v��g>@      ��!       "	���r @���r @!���r @*      ��!       2	i�G5���?i�G5���?!i�G5���?:	���@��?���@��?!���@��?B      ��!       J	7P��|z�?7P��|z�?!7P��|z�?R      ��!       Z	7P��|z�?7P��|z�?!7P��|z�?JGPUY
���Z��?b 