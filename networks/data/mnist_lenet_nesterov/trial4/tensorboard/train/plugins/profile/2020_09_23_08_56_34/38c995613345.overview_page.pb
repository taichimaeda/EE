�	g�E(�B@g�E(�B@!g�E(�B@	D�
xF�?D�
xF�?!D�
xF�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6g�E(�B@���u�<@1��S9�i@A�� !��?I�mp���?Y>{.S��?*	�$��sU@2F
Iterator::Model���խ�?!��AZ��I@)��o�h��?1&5�� E@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat)x
�Rϒ?!�ByXh5@)���l �?1JW]��0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateW�'��?!�5LDF 2@)J���?1�׿�e�'@:Preprocessing2U
Iterator::Model::ParallelMapV2W|C�u�?!�a����"@)W|C�u�?1�a����"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8��@�?!ur��Y0H@)c}�Ev?1�ρ�uX@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����`u?!'�MT@)����`u?1'�MT@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr1�q�p?!k�o��T@)r1�q�p?1k�o��T@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��oB!�?!_.#�=�4@)��cw�b?1$ŷ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9C�
xF�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���u�<@���u�<@!���u�<@      ��!       "	��S9�i@��S9�i@!��S9�i@*      ��!       2	�� !��?�� !��?!�� !��?:	�mp���?�mp���?!�mp���?B      ��!       J	>{.S��?>{.S��?!>{.S��?R      ��!       Z	>{.S��?>{.S��?!>{.S��?JGPUYC�
xF�?b �"l
Bgradient_tape/sequential_63/conv2d_190/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter1Dt���?!1Dt���?"?
sequential_63/conv2d_190/Relu_FusedConv2D=9�(��?!h0��Pt�?"j
Agradient_tape/sequential_63/conv2d_190/Conv2D/Conv2DBackpropInputConv2DBackpropInput�1M!_��?!�|o�b�?"l
Bgradient_tape/sequential_63/conv2d_191/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteryW7:L�?![�����?"l
Bgradient_tape/sequential_63/conv2d_189/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter �#Ϫ?!�Yh]��?"?
sequential_63/conv2d_191/Relu_FusedConv2Dq��9�?!�#�Ќ��?"j
Agradient_tape/sequential_63/conv2d_191/Conv2D/Conv2DBackpropInputConv2DBackpropInputXf�J�?!8�	.�?":
sequential_63/dense_126/MatMulMatMul���P�?!|�3?���?"b
Agradient_tape/sequential_63/max_pooling2d_126/MaxPool/MaxPoolGradMaxPoolGrad�����#�?!�SS�λ�?"H
,gradient_tape/sequential_63/dense_126/MatMulMatMul� _�`z�?!�K���w�?Q      Y@Ymާ�d0@a�d���T@q�)]��J@y]΅ф"�?"�	
both�Your program is POTENTIALLY input-bound because 80.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�52.1761% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 