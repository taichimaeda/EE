�	{.S��	C@{.S��	C@!{.S��	C@	v�m˽~�?v�m˽~�?!v�m˽~�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6{.S��	C@���[��>@1\;Q�@A��-Y�?I��+H3��?Y���ۂ��?*	#��~jЂ@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���\���?!�*�.��U@)���\���?1�*�.��U@:Preprocessing2F
Iterator::Modelհ��T�?!���}@)�1!撚?1����=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=ByG�?!��
	@)����&�?1}�*��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ӀAR�?!�%�wzV@)��	Q�?1V?*��@:Preprocessing2U
Iterator::Model::ParallelMapV2vq�-�?!Al���?)vq�-�?1Al���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn�ݳ.�?!��oA'�W@)V�F�?x?1��v9w�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@4��i?!:J��':�?)@4��i?1:J��':�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2 {��c�?!j��}�V@)�;P�<�a?1Й��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 81.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9u�m˽~�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���[��>@���[��>@!���[��>@      ��!       "	\;Q�@\;Q�@!\;Q�@*      ��!       2	��-Y�?��-Y�?!��-Y�?:	��+H3��?��+H3��?!��+H3��?B      ��!       J	���ۂ��?���ۂ��?!���ۂ��?R      ��!       Z	���ۂ��?���ۂ��?!���ۂ��?JGPUYu�m˽~�?b �"l
Bgradient_tape/sequential_38/conv2d_115/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter}9���?!}9���?"j
Agradient_tape/sequential_38/conv2d_115/Conv2D/Conv2DBackpropInputConv2DBackpropInput�p�G���?!��K]���?"?
sequential_38/conv2d_115/Relu_FusedConv2D�8u���?!�2�����?"l
Bgradient_tape/sequential_38/conv2d_116/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�H0�8�?!P3��)�?"l
Bgradient_tape/sequential_38/conv2d_114/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter� Ky�?!f5�L�x�?"?
sequential_38/conv2d_116/Relu_FusedConv2D�"+�#��?!�L�e6X�?"j
Agradient_tape/sequential_38/conv2d_116/Conv2D/Conv2DBackpropInputConv2DBackpropInputG_�w��?!��x���?"9
sequential_38/dense_76/MatMulMatMul_KP*��?!2eː�x�?"a
@gradient_tape/sequential_38/max_pooling2d_76/MaxPool/MaxPoolGradMaxPoolGrad�ӅSǙ?!���,�F�?"G
+gradient_tape/sequential_38/dense_76/MatMulMatMul�Q�HgS�?!D@g��?Q      Y@Y!�B!0@a��{��T@q�Q~_��@@y4A׮��?"�	
both�Your program is POTENTIALLY input-bound because 81.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�33.9364% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 