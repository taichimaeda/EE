	��릔_C@��릔_C@!��릔_C@	�$�<V�?�$�<V�?!�$�<V�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��릔_C@�wD��?@1���l��@A�G���\�?I��B=}��?YWд��h�?*	��/�dY@2F
Iterator::Modelb��vKr�?!���ʀG@)���q��?1.��h!C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateb���4�?!#ƥ2�e;@)��y�Cn�?1=�����5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!��6�1@)КiQ�?1^KV�.@:Preprocessing2U
Iterator::Model::ParallelMapV2F�@1�?!OnI��}!@)F�@1�?1OnI��}!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�/��Cx?!���oT@)�/��Cx?1���oT@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�'�y��?!iv�h5J@)M�O�t?1_d��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT��b�h?!q/���@)T��b�h?1q/���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�E;�I�?!sI�v>@)46<�Rf?1�<6v@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 81.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�$�<V�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�wD��?@�wD��?@!�wD��?@      ��!       "	���l��@���l��@!���l��@*      ��!       2	�G���\�?�G���\�?!�G���\�?:	��B=}��?��B=}��?!��B=}��?B      ��!       J	Wд��h�?Wд��h�?!Wд��h�?R      ��!       Z	Wд��h�?Wд��h�?!Wд��h�?JGPUY�$�<V�?b 