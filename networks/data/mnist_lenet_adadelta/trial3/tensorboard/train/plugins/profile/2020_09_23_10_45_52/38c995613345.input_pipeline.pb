	gs�B@gs�B@!gs�B@	�_�ݴ�?�_�ݴ�?!�_�ݴ�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6gs�B@X}w�<@1!����@A�y��C5�?I�F� \�?Y�q�&"�?*	�Zd;7X@2F
Iterator::Model��B���?!�����@F@)[A�+��?1��k%�A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV]��?!�ԃ9:@)Cp\�M�?1)t��W74@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+��A�?!�P�wl4@)�!��T2�?1򔤡IT0@:Preprocessing2U
Iterator::Model::ParallelMapV2�]��a��?!J����!@)�]��a��?1J����!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipɯb���?!hzl56�K@)A��h:;y?1�q p@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceI��Z��w?!_��@)I��Z��w?1_��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorc'��>p?!���`@)c'��>p?1���`@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`��9z�?!w`^��<@)O��'�c?17��JL�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�_�ݴ�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	X}w�<@X}w�<@!X}w�<@      ��!       "	!����@!����@!!����@*      ��!       2	�y��C5�?�y��C5�?!�y��C5�?:	�F� \�?�F� \�?!�F� \�?B      ��!       J	�q�&"�?�q�&"�?!�q�&"�?R      ��!       Z	�q�&"�?�q�&"�?!�q�&"�?JGPUY�_�ݴ�?b 