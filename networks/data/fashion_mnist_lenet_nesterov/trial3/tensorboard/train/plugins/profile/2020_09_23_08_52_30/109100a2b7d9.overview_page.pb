�	�Ù_�qB@�Ù_�qB@!�Ù_�qB@	n��^���?n��^���?!n��^���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�Ù_�qB@�ʉv�=@1$���(@A��+H3�?I��j��?YTq��s�?*	/�$���@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenaten/�?!rBԷ>�U@)ȳ˷>��?12�; �U@:Preprocessing2F
Iterator::Model���x�?!G�Xbd&@)�߃�.m�?1��E7M@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatE�J�E�?!��A�)�@)D�Ac&�?1���H@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#,*�t�?!;tڹ�-W@)���م?1o�A��?:Preprocessing2U
Iterator::Model::ParallelMapV2�fI-�?!��J|�d�?)�fI-�?1��J|�d�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceAJ�i�w?!��3�G�?)AJ�i�w?1��3�G�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�HP�h?!�2X���?)�HP�h?1�2X���?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�����?!<k1f��U@)�O ��b?1z�(]���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o��^���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ʉv�=@�ʉv�=@!�ʉv�=@      ��!       "	$���(@$���(@!$���(@*      ��!       2	��+H3�?��+H3�?!��+H3�?:	��j��?��j��?!��j��?B      ��!       J	Tq��s�?Tq��s�?!Tq��s�?R      ��!       Z	Tq��s�?Tq��s�?!Tq��s�?JGPUYo��^���?b �"l
Bgradient_tape/sequential_62/conv2d_187/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�H7�k�?!�H7�k�?"j
Agradient_tape/sequential_62/conv2d_187/Conv2D/Conv2DBackpropInputConv2DBackpropInput%���C
�?!|W��8�?"?
sequential_62/conv2d_187/Relu_FusedConv2DY<���[�?!�&6��?"l
Bgradient_tape/sequential_62/conv2d_188/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��!�:��?!E��N��?"l
Bgradient_tape/sequential_62/conv2d_186/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�Ra�٪?!��*�T�?"?
sequential_62/conv2d_188/Relu_FusedConv2D~�P�)�?!y�!nF��?"j
Agradient_tape/sequential_62/conv2d_188/Conv2D/Conv2DBackpropInputConv2DBackpropInput8�m_�?!�btG��?":
sequential_62/dense_124/MatMulMatMul���,��?!3�@�`��?"b
Agradient_tape/sequential_62/max_pooling2d_124/MaxPool/MaxPoolGradMaxPoolGrad��֌_�?!s��NY��?"=
 sequential_62/conv2d_186/BiasAddBiasAddm#;�ޖ�?!��Eb�?Q      Y@Y�4_�g�0@a�2(&�T@qU��hwC@yO��}h��?"�	
both�Your program is POTENTIALLY input-bound because 80.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�38.9329% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 