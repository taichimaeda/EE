	��O��A@��O��A@!��O��A@	pP69�?pP69�?!pP69�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��O��A@��W:r<@1\:�<c@A���h��?I�f*�#��?Y���_�|�?*	���Q d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatesG�˵h�?!�z ��O@)l�<*��?1Pt�JIM@:Preprocessing2F
Iterator::Model5��,�?!�'Vw].6@)����?1�W��S	1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���^Dۑ?!�bڮ��%@)�+��ص�?1ƚ6+�!"@:Preprocessing2U
Iterator::Model::ParallelMapV2��>V�ۀ?!�?�&�@)��>V�ۀ?1�?�&�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��M�v?!��x��@)��M�v?1��x��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipwi�ai�?!v*�htS@)W?6ɏ�u?1��}�
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�]���h?!A�N�?)�]���h?1A�N�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�zM
�?!�7�}��O@)S�r/0+d?1^��{S��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9pP69�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��W:r<@��W:r<@!��W:r<@      ��!       "	\:�<c@\:�<c@!\:�<c@*      ��!       2	���h��?���h��?!���h��?:	�f*�#��?�f*�#��?!�f*�#��?B      ��!       J	���_�|�?���_�|�?!���_�|�?R      ��!       Z	���_�|�?���_�|�?!���_�|�?JGPUYpP69�?b 