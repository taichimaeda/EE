�	鷯�L@鷯�L@!鷯�L@	n���l��?n���l��?!n���l��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6鷯�L@nk�KaI@1c~nh�>@AL������?Ig�;p�?Y��yS�
�?*	����̼`@2F
Iterator::Model�
F%u�?!���EJ@)��}�p�?1����xA@:Preprocessing2U
Iterator::Model::ParallelMapV2���Z(�?!���Y2@)���Z(�?1���Y2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=�+J	��?!��g�'�3@)��󬤕?1�HLC��/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��Z�?!;���0�3@)�: �^�?1W[,�+/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0*��D�?!�yf8��G@)�GnM�-�?1x䯸@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice.����w?!=&(�2u@).����w?1=&(�2u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor%�c\qqt?!b���@)%�c\qqt?1b���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����Y��?!Zl� �5@)�D�A�c?1��G���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o���l��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	nk�KaI@nk�KaI@!nk�KaI@      ��!       "	c~nh�>@c~nh�>@!c~nh�>@*      ��!       2	L������?L������?!L������?:	g�;p�?g�;p�?!g�;p�?B      ��!       J	��yS�
�?��yS�
�?!��yS�
�?R      ��!       Z	��yS�
�?��yS�
�?!��yS�
�?JGPUYo���l��?b �"j
@gradient_tape/sequential_7/conv2d_22/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�����?!�����?"h
?gradient_tape/sequential_7/conv2d_22/Conv2D/Conv2DBackpropInputConv2DBackpropInput����>7�?!L,�z `�?"=
sequential_7/conv2d_22/Relu_FusedConv2D�j��(�?!�a�tht�?"j
@gradient_tape/sequential_7/conv2d_23/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��G�S�?!��U�d�?"j
@gradient_tape/sequential_7/conv2d_21/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!��Fk�?!v>���?"h
?gradient_tape/sequential_7/conv2d_23/Conv2D/Conv2DBackpropInputConv2DBackpropInputq�'t��?!���v}�?"=
sequential_7/conv2d_23/Relu_FusedConv2D���ѱ˝?!窱"1Z�?"K
-gradient_tape/sequential_7/conv2d_21/ReluGradReluGrad��ghƕ?!`#8钶�?"K
-gradient_tape/sequential_7/conv2d_22/ReluGradReluGrad��ghƕ?!ٛ����?"8
sequential_7/dense_14/MatMulMatMul��ghƕ?!REvVo�?Q      Y@YA�A�@a\��[�eW@q���w�O@y�w���0�?"�	
both�Your program is POTENTIALLY input-bound because 88.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�63.3396% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 