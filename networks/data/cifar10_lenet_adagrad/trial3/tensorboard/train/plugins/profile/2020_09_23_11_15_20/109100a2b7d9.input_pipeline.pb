	H4�"�A@H4�"�A@!H4�"�A@	�2�>��?�2�>��?!�2�>��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6H4�"�A@�\6:�;@1��{�	 @A�1^�?I�~2Ƈ��?Yt�?Pn۳?*	Q���-j@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�s��q5�?!Qpi��P@)v���_w�?1���qsiL@:Preprocessing2F
Iterator::Model�q��rg�?!�sw���4@)C���-�?1 :b��0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{���Η?! h]!3&@)�C5%Y��?1� �,<6"@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�*n�b�?!�>�FRV@)(~��k	�?1���t9Y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��Yh�4�?!#�A��S@)+�&�|��?1���e@:Preprocessing2U
Iterator::Model::ParallelMapV2	N} y�?!c��T�@)	N} y�?1c��T�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice1{�v�q?!E;"�)��?)1{�v�q?1E;"�)��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoryx��ee?!5Gc��?)yx��ee?15Gc��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�2�>��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�\6:�;@�\6:�;@!�\6:�;@      ��!       "	��{�	 @��{�	 @!��{�	 @*      ��!       2	�1^�?�1^�?!�1^�?:	�~2Ƈ��?�~2Ƈ��?!�~2Ƈ��?B      ��!       J	t�?Pn۳?t�?Pn۳?!t�?Pn۳?R      ��!       Z	t�?Pn۳?t�?Pn۳?!t�?Pn۳?JGPUY�2�>��?b 