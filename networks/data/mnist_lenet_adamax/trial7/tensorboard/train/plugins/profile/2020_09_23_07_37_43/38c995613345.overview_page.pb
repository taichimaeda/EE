�	Y��;�B@Y��;�B@!Y��;�B@	���Q��?���Q��?!���Q��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Y��;�B@��N�X>@1��+� @A@��
/�?I[$�F��?YR_�vj��?*	��ʡm^@2F
Iterator::Model��P�n�?!ʡ�l�1Q@)9d�bӪ?1�곜	�E@:Preprocessing2U
Iterator::Model::ParallelMapV2:�%��?!��2z�9@):�%��?1��2z�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���O=�?!��D-@)�S��э?1�~D�i�'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[��X��?!&�o";}$@)*��g\8�?1�n�H@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�|]��t�?!�xeL�8?@)��t �u?1��lz�a@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec����r?!L�T\�@)c����r?1L�T\�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr��	�j?!#b�z�a@)r��	�j?1#b�z�a@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��k]j��?!Ln��G|(@)
�F�c?1/���d��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 81.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9���Q��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��N�X>@��N�X>@!��N�X>@      ��!       "	��+� @��+� @!��+� @*      ��!       2	@��
/�?@��
/�?!@��
/�?:	[$�F��?[$�F��?![$�F��?B      ��!       J	R_�vj��?R_�vj��?!R_�vj��?R      ��!       Z	R_�vj��?R_�vj��?!R_�vj��?JGPUY���Q��?b �"l
Bgradient_tape/sequential_36/conv2d_109/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��?B���?!��?B���?"j
Agradient_tape/sequential_36/conv2d_109/Conv2D/Conv2DBackpropInputConv2DBackpropInputx&�-��?!9v�8i�?"?
sequential_36/conv2d_109/Relu_FusedConv2D�L��?!��_�a��?"l
Bgradient_tape/sequential_36/conv2d_110/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�p�*Zl�?!��ix��?"?
sequential_36/conv2d_110/Relu_FusedConv2D�9�n�a�?!������?"l
Bgradient_tape/sequential_36/conv2d_108/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteri�0A�?!q쇮i"�?"j
Agradient_tape/sequential_36/conv2d_110/Conv2D/Conv2DBackpropInputConv2DBackpropInput (tЭ�?!�.��4��?"9
sequential_36/dense_72/MatMulMatMuld�CW$��?!�NI�R�?"a
@gradient_tape/sequential_36/max_pooling2d_72/MaxPool/MaxPoolGradMaxPoolGrad�~}^�d�?!�:=:�?"M
/gradient_tape/sequential_36/conv2d_109/ReluGradReluGrad����z��?!ž��=��?Q      Y@Y      0@a      U@q�Y�ރ=@yW�F���?"�	
both�Your program is POTENTIALLY input-bound because 81.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�29.5151% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 