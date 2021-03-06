�	�:�2G@�:�2G@!�:�2G@	|u��fR�?|u��fR�?!|u��fR�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�:�2G@)u�8F.C@1u��l�@An1?74e�?I�f��j��?Yg�R@���?*	L7�A`�U@2F
Iterator::Modelv�1<��?!~�JF@)��a��4�?1�Uo�%lA@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenater�	�OƘ?!���ʩ;@)"rl=�?1,b��{5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=ڨN�?!���dm!4@)Ý#���?1L��u�0@:Preprocessing2U
Iterator::Model::ParallelMapV2���Co�?!r��w#@)���Co�?1r��w#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�£�#v?!�\�@)�£�#v?1�\�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipjܛ�0Ѩ?!��Z�K@)ADj��4s?1I�
�
r@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor܂����i?!��w_�@)܂����i?1��w_�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���͚?!� ���=@)��%�<`?1����/!@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 82.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9|u��fR�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	)u�8F.C@)u�8F.C@!)u�8F.C@      ��!       "	u��l�@u��l�@!u��l�@*      ��!       2	n1?74e�?n1?74e�?!n1?74e�?:	�f��j��?�f��j��?!�f��j��?B      ��!       J	g�R@���?g�R@���?!g�R@���?R      ��!       Z	g�R@���?g�R@���?!g�R@���?JGPUY|u��fR�?b �"l
Bgradient_tape/sequential_72/conv2d_217/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter���ȸ��?!���ȸ��?"j
Agradient_tape/sequential_72/conv2d_217/Conv2D/Conv2DBackpropInputConv2DBackpropInputL��g��?!�o��l?�?"?
sequential_72/conv2d_217/Relu_FusedConv2D!W2T�%�?!�M�'2��?"l
Bgradient_tape/sequential_72/conv2d_218/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�;��Ø�?!|�c��?"l
Bgradient_tape/sequential_72/conv2d_216/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterֹ��XD�?!��,���?"?
sequential_72/conv2d_218/Relu_FusedConv2D{3*����?!&ZD���?"j
Agradient_tape/sequential_72/conv2d_218/Conv2D/Conv2DBackpropInputConv2DBackpropInput��N�΀�?!�G�Ϳ�?":
sequential_72/dense_144/MatMulMatMul~�'u�?!�{��v��?"b
Agradient_tape/sequential_72/max_pooling2d_144/MaxPool/MaxPoolGradMaxPoolGrad�+�����?!��$Y�?"H
,gradient_tape/sequential_72/dense_144/MatMulMatMul�⩚�!�?!.�S3
�?Q      Y@Y߈�N�#@a䮟-V�V@q��/��%N@y,��9���?"�	
both�Your program is POTENTIALLY input-bound because 82.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�60.2968% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 