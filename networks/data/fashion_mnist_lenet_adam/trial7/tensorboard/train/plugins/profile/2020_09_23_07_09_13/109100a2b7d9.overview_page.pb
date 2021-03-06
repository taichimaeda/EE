�	�,����B@�,����B@!�,����B@	�5Lц�?�5Lц�?!�5Lц�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�,����B@��W� >@1Ϊ��V@A�;ۣ7ܗ?I8������?Y�� x|�?*	�G�z]@2F
Iterator::Model���H�?!�؈R+3E@)n��4Ң?1�Q��Џ?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�x@ٔ+�?!��F~�@@)��B=}�?1���-1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceC p�ٓ?!xD�p�0@)C p�ٓ?1xD�p�0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr�CQ�O�?!����1@)���U��?1�S�%=�+@:Preprocessing2U
Iterator::Model::ParallelMapV2�Tkaډ?!1�(>�%@)�Tkaډ?11�(>�%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipǛ��,�?!1'w���L@)�?�,u?1�G=��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT8�T�m?!�N���@)T8�T�m?1�N���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapeo)狥?!R���B@)���%f?1L�C�v@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�5Lц�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��W� >@��W� >@!��W� >@      ��!       "	Ϊ��V@Ϊ��V@!Ϊ��V@*      ��!       2	�;ۣ7ܗ?�;ۣ7ܗ?!�;ۣ7ܗ?:	8������?8������?!8������?B      ��!       J	�� x|�?�� x|�?!�� x|�?R      ��!       Z	�� x|�?�� x|�?!�� x|�?JGPUY�5Lц�?b �"k
Agradient_tape/sequential_26/conv2d_79/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteriG4%�8�?!iG4%�8�?"i
@gradient_tape/sequential_26/conv2d_79/Conv2D/Conv2DBackpropInputConv2DBackpropInput�VVo�M�?!a�o�/�?">
sequential_26/conv2d_79/Relu_FusedConv2D��Gm`�?!���	�(�?"k
Agradient_tape/sequential_26/conv2d_80/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�Aǐ�۱?!q�����?"k
Agradient_tape/sequential_26/conv2d_78/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�i�됛�?!:>n0��?">
sequential_26/conv2d_80/Relu_FusedConv2DЭ��D�?!��^��?"i
@gradient_tape/sequential_26/conv2d_80/Conv2D/Conv2DBackpropInputConv2DBackpropInput��'���?!�| 1���?"9
sequential_26/dense_52/MatMulMatMulp?�ۛ?!�Fy)���?"a
@gradient_tape/sequential_26/max_pooling2d_52/MaxPool/MaxPoolGradMaxPoolGrad��&h�}�?!�}�T���?"<
sequential_26/conv2d_78/BiasAddBiasAdd��q��~�?!�	~d�X�?Q      Y@Y      0@a      U@q-��A�xJ@y�6c�>��?"�	
both�Your program is POTENTIALLY input-bound because 79.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�52.9451% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 