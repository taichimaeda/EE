�	�Ƕ8B@�Ƕ8B@!�Ƕ8B@	�qq�2g�?�qq�2g�?!�qq�2g�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�Ƕ8B@_\��#;@1���>�0 @A~�.rO�?I���8aB�?YU�wE��?*	��ʡ�c@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatv3���?!uzcO��N@)!yv�ַ?1vI/U�M@:Preprocessing2F
Iterator::Model�@+0du�?!�$a��U8@)9��� �?1���*4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��Udt@�?!ƶ�ݓ�R@)*��g\8�?1�ƞ�H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatejg��R�?!�"��L@)z�W�|?1���e�@:Preprocessing2U
Iterator::Model::ParallelMapV2�%��:�z?!LkeEj�@)�%��:�z?1LkeEj�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice[�*�MFu?!{+���
@)[�*�MFu?1{+���
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�
F%uj?!�E��C @)�
F%uj?1�E��C @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�ص�ݒ�?!�h`$��!@)P��W\\?1��&�ܻ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�qq�2g�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	_\��#;@_\��#;@!_\��#;@      ��!       "	���>�0 @���>�0 @!���>�0 @*      ��!       2	~�.rO�?~�.rO�?!~�.rO�?:	���8aB�?���8aB�?!���8aB�?B      ��!       J	U�wE��?U�wE��?!U�wE��?R      ��!       Z	U�wE��?U�wE��?!U�wE��?JGPUY�qq�2g�?b �"m
Cgradient_tape/sequential_106/conv2d_319/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterv�zs��?!v�zs��?"m
Cgradient_tape/sequential_106/conv2d_320/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�p����?!~@{]h�?"@
sequential_106/conv2d_319/Relu_FusedConv2DW8~�,^�?!��Z��/�?"k
Bgradient_tape/sequential_106/conv2d_319/Conv2D/Conv2DBackpropInputConv2DBackpropInputL�%����?!'Dd7��?"-
IteratorGetNext/_1_Send�Zb<,U�?!��|F���?"m
Cgradient_tape/sequential_106/conv2d_318/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter���%�?!��[�7�?"@
sequential_106/conv2d_320/Relu_FusedConv2D�M�]:��?!q�c�v�?"k
Bgradient_tape/sequential_106/conv2d_320/Conv2D/Conv2DBackpropInputConv2DBackpropInput	$B��?!2�%#���?"I
-gradient_tape/sequential_106/dense_212/MatMulMatMul�?~p (�?!-�-*A��?";
sequential_106/dense_212/MatMulMatMul����@��?!�B.;j�?Q      Y@Y��1@a������T@qJ<��?@y�S���ަ?"�	
both�Your program is POTENTIALLY input-bound because 75.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�31.9202% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 