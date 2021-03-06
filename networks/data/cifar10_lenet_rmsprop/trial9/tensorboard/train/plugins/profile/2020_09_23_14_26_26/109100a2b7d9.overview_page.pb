�	�cϞ��F@�cϞ��F@!�cϞ��F@	���[��?���[��?!���[��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�cϞ��F@���s�A@1� ���� @A��?k~��?I�N>=�e�?Y�HM��f�?*	\���(�X@2F
Iterator::Model��@��?!dɡ"�`H@)���-=��?1i�:�j-B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�U�@�?!0���8@)����Ǔ?1zo��T3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatē���G�?!��,�2@)`��Ù�?11'��.@:Preprocessing2U
Iterator::Model::ParallelMapV2G9�M�a�?!����(@)G9�M�a�?1����(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�[���u?!�C��c@)�[���u?1�C��c@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�1��8�?!�6^�S�I@)O��'�s?1O�7s�F@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���:8�k?!�sp�T5@)���:8�k?1�sp�T5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM��E;�?!��'��;@)�R����g?1Dĵ�(G@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 78.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9���[��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���s�A@���s�A@!���s�A@      ��!       "	� ���� @� ���� @!� ���� @*      ��!       2	��?k~��?��?k~��?!��?k~��?:	�N>=�e�?�N>=�e�?!�N>=�e�?B      ��!       J	�HM��f�?�HM��f�?!�HM��f�?R      ��!       Z	�HM��f�?�HM��f�?!�HM��f�?JGPUY���[��?b �"m
Cgradient_tape/sequential_168/conv2d_505/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�������?!�������?"m
Cgradient_tape/sequential_168/conv2d_506/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�V�+��?!!j3?�?"@
sequential_168/conv2d_505/Relu_FusedConv2D\x���H�?!8�m]�?"k
Bgradient_tape/sequential_168/conv2d_505/Conv2D/Conv2DBackpropInputConv2DBackpropInput<�&y���?!Go�7K��?"-
IteratorGetNext/_1_Send6�//�?!�|���?"m
Cgradient_tape/sequential_168/conv2d_504/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh~yn/)�?!KV�x�3�?"@
sequential_168/conv2d_506/Relu_FusedConv2D0����?!L�q��b�?"k
Bgradient_tape/sequential_168/conv2d_506/Conv2D/Conv2DBackpropInputConv2DBackpropInput���۠?!��N�p�?"I
-gradient_tape/sequential_168/dense_336/MatMulMatMul-�V�-��?!W��N)f�?";
sequential_168/dense_336/MatMulMatMul�?%7�?!��x�/�?Q      Y@Y�ΐ��3$@a$���yV@q��C�n�E@y_:��¥?"�	
both�Your program is POTENTIALLY input-bound because 78.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�43.3784% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 