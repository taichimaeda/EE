�	�;�_�M@�;�_�M@!�;�_�M@	́4�n�?́4�n�?!́4�n�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�;�_�M@���ѥH@1t]����@A(~��k	�?I����Te�?Y)?���x�?*	���S�@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat E�����?!T�J�<U@)�46<�?1&�,=U@:Preprocessing2F
Iterator::Model��26t��?!i���J"@)��×��?1�<���@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate`w���s�?!F&�`S@)�G��Q�?1�-?̹
@:Preprocessing2U
Iterator::Model::ParallelMapV2��p�q��?!���DP��?)��p�q��?1���DP��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��w�Go�?!3Ed佶V@)��ϛ�Tx?1F	�@_��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice
��ϛ�t?!�{�bϳ�?)
��ϛ�t?1�{�bϳ�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_Pjr?!�!5��?)_Pjr?1�!5��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�#�&ݖ�?!�K]��@)sd��a?1�UBW�b�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9́4�n�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���ѥH@���ѥH@!���ѥH@      ��!       "	t]����@t]����@!t]����@*      ��!       2	(~��k	�?(~��k	�?!(~��k	�?:	����Te�?����Te�?!����Te�?B      ��!       J	)?���x�?)?���x�?!)?���x�?R      ��!       Z	)?���x�?)?���x�?!)?���x�?JGPUY́4�n�?b �"l
Bgradient_tape/sequential_50/conv2d_151/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����O��?!����O��?"j
Agradient_tape/sequential_50/conv2d_151/Conv2D/Conv2DBackpropInputConv2DBackpropInput������?!�8"6��?"?
sequential_50/conv2d_151/Relu_FusedConv2D��3 ;m�?!�c^��?"l
Bgradient_tape/sequential_50/conv2d_152/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�~�" �?!��-�b��?"l
Bgradient_tape/sequential_50/conv2d_150/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`d���?! W����?"?
sequential_50/conv2d_152/Relu_FusedConv2D�S�.QG�?!����i�?"j
Agradient_tape/sequential_50/conv2d_152/Conv2D/Conv2DBackpropInputConv2DBackpropInput�˄���?![�u���?":
sequential_50/dense_100/MatMulMatMul:Z��[�?!Z��0�
�?"b
Agradient_tape/sequential_50/max_pooling2d_100/MaxPool/MaxPoolGradMaxPoolGrad�Sǿ�g�?!�9�����?"H
,gradient_tape/sequential_50/dense_100/MatMulMatMul<�zn�$�?!	J_�?Q      Y@Y@m�Kz@a,��O[hW@q���sG@y�$�VW��?"�	
both�Your program is POTENTIALLY input-bound because 84.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�46.9038% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 