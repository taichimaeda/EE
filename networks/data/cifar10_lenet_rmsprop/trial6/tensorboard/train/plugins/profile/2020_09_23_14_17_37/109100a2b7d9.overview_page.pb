�	ak��/H@ak��/H@!ak��/H@	n�C�'��?n�C�'��?!n�C�'��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ak��/H@ط���3C@1����#!@AP��|zl�?I�X5s;�?Y������?*	���Ʒn@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR+L�k�?!��[��O@)gDio���?1�SZ�O@:Preprocessing2F
Iterator::Model1��*��?!o�E�|�6@)5
If��?1�x�S2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�P�,�?!r�~^$@)���q��?1�?:�@:Preprocessing2U
Iterator::Model::ParallelMapV2�'*�T�?!t�3˥�@)�'*�T�?1t�3˥�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipRb��vK�?!�.� OS@)x'�{?1D�e��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��$xCu?!�u�]�� @)��$xCu?1�u�]�� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorQ�\�mOp?!�^�2 ��?)Q�\�mOp?1�^�2 ��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��{�P�?!�w��%@)j��%!a?1|�o{:�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9n�C�'��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ط���3C@ط���3C@!ط���3C@      ��!       "	����#!@����#!@!����#!@*      ��!       2	P��|zl�?P��|zl�?!P��|zl�?:	�X5s;�?�X5s;�?!�X5s;�?B      ��!       J	������?������?!������?R      ��!       Z	������?������?!������?JGPUYn�C�'��?b �"m
Cgradient_tape/sequential_165/conv2d_496/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterjt�����?!jt�����?"m
Cgradient_tape/sequential_165/conv2d_497/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteruC_�瀵?!���*\�?"@
sequential_165/conv2d_496/Relu_FusedConv2D����0�?!xI��^(�?"k
Bgradient_tape/sequential_165/conv2d_496/Conv2D/Conv2DBackpropInputConv2DBackpropInput����郲?!���]Y��?"-
IteratorGetNext/_1_SendްV �?!Т?_]�?"m
Cgradient_tape/sequential_165/conv2d_495/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�����?!��I+�9�?"@
sequential_165/conv2d_497/Relu_FusedConv2D)�ė<ڢ?!���Zg�?"k
Bgradient_tape/sequential_165/conv2d_497/Conv2D/Conv2DBackpropInputConv2DBackpropInput����R��?!�N�! s�?"I
-gradient_tape/sequential_165/dense_330/MatMulMatMul�����s�?!=�Q>�f�?";
sequential_165/dense_330/MatMulMatMul���� #�?!��C�/�?Q      Y@Y�ΐ��3$@a$���yV@qIY��B@y-��-��?"�	
both�Your program is POTENTIALLY input-bound because 79.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�37.3289% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 