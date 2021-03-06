�	�A%�cvA@�A%�cvA@!�A%�cvA@	����Z��?����Z��?!����Z��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�A%�cvA@g��}qY;@1r4GV~�@Aߨ��5�?I� ����?Y�t��.��?*	:��v��^@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����`�?!����qG@)߉Y/��?1T�K��*E@:Preprocessing2F
Iterator::Model������?!����Vp?@)QO�?��?1�⤵�!7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat:#/kb�?!���R,�+@)2��z��?1��W!��&@:Preprocessing2U
Iterator::Model::ParallelMapV2;�?l�ф?!�c��s� @);�?l�ф?1�c��s� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT�*�gz�?!�ZZ�#Q@)q��]P?1\��+U�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceg|_\��v?!����6@)g|_\��v?1����6@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	�L�nh?!;���5@)	�L�nh?1;���5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[��Y�?!�Z�^8H@)ŏ1w-!_?1��=���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 78.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9����Z��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	g��}qY;@g��}qY;@!g��}qY;@      ��!       "	r4GV~�@r4GV~�@!r4GV~�@*      ��!       2	ߨ��5�?ߨ��5�?!ߨ��5�?:	� ����?� ����?!� ����?B      ��!       J	�t��.��?�t��.��?!�t��.��?R      ��!       Z	�t��.��?�t��.��?!�t��.��?JGPUY����Z��?b �"k
Agradient_tape/sequential_30/conv2d_91/Conv2D/Conv2DBackpropFilterConv2DBackpropFilters��Y��?!s��Y��?"i
@gradient_tape/sequential_30/conv2d_91/Conv2D/Conv2DBackpropInputConv2DBackpropInput�+����?!V�5�>��?">
sequential_30/conv2d_91/Relu_FusedConv2Db3@��߷?!+��?"k
Agradient_tape/sequential_30/conv2d_92/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter䃽���?!�c����?"k
Agradient_tape/sequential_30/conv2d_90/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.��	EI�?!#�#�a�?">
sequential_30/conv2d_92/Relu_FusedConv2D��(�)��?!~�J+�P�?"i
@gradient_tape/sequential_30/conv2d_92/Conv2D/Conv2DBackpropInputConv2DBackpropInput��wC�?!��bB��?"9
sequential_30/dense_60/MatMulMatMull�F��K�?!K?2���?"a
@gradient_tape/sequential_30/max_pooling2d_60/MaxPool/MaxPoolGradMaxPoolGradr�9�SΙ?!�\[�?"G
+gradient_tape/sequential_30/dense_60/MatMulMatMuldD�?!/��?Q      Y@Y      0@a      U@q�偗�C@y�i�8�?"�	
both�Your program is POTENTIALLY input-bound because 78.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�39.7038% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 