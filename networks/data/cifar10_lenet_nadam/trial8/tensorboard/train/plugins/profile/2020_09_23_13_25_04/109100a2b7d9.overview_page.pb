�	���SɯP@���SɯP@!���SɯP@	/�����?/�����?!/�����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���SɯP@!�> �+L@1R�Hڍ~"@AW@�ի�?I�����?Y�r����?*	�x�&1�X@2F
Iterator::Model��&��?!��&UG@)�	m9��?1�h��{VB@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�Z����?!&��eF9@)��w�-;�?1ަ۹�3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatd��A%��?!3�k�i3@)��_ �?1BC_U��/@:Preprocessing2U
Iterator::Model::ParallelMapV2�uT5A�?!���\��#@)�uT5A�?1���\��#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��B���?!�4� ٪J@)VW@�u?1�c��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�}͑u?!!�M��F@)�}͑u?1!�M��F@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(�8'0m?!���	��@)(�8'0m?1���	��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�'���?!�nq��<@)�d��7ij?1��ʎ>
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9/�����?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	!�> �+L@!�> �+L@!!�> �+L@      ��!       "	R�Hڍ~"@R�Hڍ~"@!R�Hڍ~"@*      ��!       2	W@�ի�?W@�ի�?!W@�ի�?:	�����?�����?!�����?B      ��!       J	�r����?�r����?!�r����?R      ��!       Z	�r����?�r����?!�r����?JGPUY/�����?b �"m
Cgradient_tape/sequential_147/conv2d_442/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��B���?!��B���?"m
Cgradient_tape/sequential_147/conv2d_443/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�E�6��?!�3�4��?"@
sequential_147/conv2d_442/Relu_FusedConv2D�㤪�ɱ?!�Ҭ��h�?"k
Bgradient_tape/sequential_147/conv2d_442/Conv2D/Conv2DBackpropInputConv2DBackpropInput��A+�?!��n�`��?"-
IteratorGetNext/_1_SendW��+t��?!��c���?"m
Cgradient_tape/sequential_147/conv2d_441/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�K�ࢡ?!�pi|��?"@
sequential_147/conv2d_443/Relu_FusedConv2D.Dg\��?!^�Q	���?"k
Bgradient_tape/sequential_147/conv2d_443/Conv2D/Conv2DBackpropInputConv2DBackpropInput������?!O�%�W��?"I
-gradient_tape/sequential_147/dense_294/MatMulMatMul�q$6E�?!77I����?";
sequential_147/dense_294/MatMulMatMul��{�1�?!Ƕ'���?Q      Y@Y�9�s�@ac�1�cW@q�#�9C@yB��B �?"�	
both�Your program is POTENTIALLY input-bound because 84.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�38.4511% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 