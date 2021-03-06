�	�w�WC@�w�WC@!�w�WC@	��`�7�?��`�7�?!��`�7�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�w�WC@Ŏơ~�<@1��:�" @Ak��� ��?I��s���?Y�L�J��?*	G�z��{@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatd���q�?!�d����T@)g_y��"�?16�F�ąT@:Preprocessing2F
Iterator::Model9DܜJ�?!����'�#@)膦��?1��p�l @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate 9a�hV�?!�H���@)��ۂ���?1:�". �@:Preprocessing2U
Iterator::Model::ParallelMapV2h��|?5~?!r�vz���?)h��|?5~?1r�vz���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�_!se�?!�&;�V@)�b.�z?1poRg��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice5D�ov?!��ڐz��?)5D�ov?1��ڐz��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��
�s?!�o����?)��
�s?1�o����?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��'����?!2'�j�@)J]2���a?1�i�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��`�7�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ŏơ~�<@Ŏơ~�<@!Ŏơ~�<@      ��!       "	��:�" @��:�" @!��:�" @*      ��!       2	k��� ��?k��� ��?!k��� ��?:	��s���?��s���?!��s���?B      ��!       J	�L�J��?�L�J��?!�L�J��?R      ��!       Z	�L�J��?�L�J��?!�L�J��?JGPUY��`�7�?b �"m
Cgradient_tape/sequential_117/conv2d_352/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterr��5���?!r��5���?"m
Cgradient_tape/sequential_117/conv2d_353/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter
�DP��?!|���?"@
sequential_117/conv2d_352/Relu_FusedConv2D�/�2CT�?!s^�x5�?"k
Bgradient_tape/sequential_117/conv2d_352/Conv2D/Conv2DBackpropInputConv2DBackpropInput�A����?!Ӯ��%�?"-
IteratorGetNext/_1_Send\Q� <��?!��OS	�?"m
Cgradient_tape/sequential_117/conv2d_351/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�ڝ�P�?!�j�H]N�?"@
sequential_117/conv2d_353/Relu_FusedConv2D�J�$��?!~�7k��?"k
Bgradient_tape/sequential_117/conv2d_353/Conv2D/Conv2DBackpropInputConv2DBackpropInputrY�\ū�?!�����?"I
-gradient_tape/sequential_117/dense_234/MatMulMatMul��t��?! ͯ���?";
sequential_117/dense_234/MatMulMatMul��u+BҚ?!>�(�+��?Q      Y@Y�4_�g�0@a�2(&�T@q��gH·6@yO��En"�?"�	
both�Your program is POTENTIALLY input-bound because 75.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�22.7178% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 