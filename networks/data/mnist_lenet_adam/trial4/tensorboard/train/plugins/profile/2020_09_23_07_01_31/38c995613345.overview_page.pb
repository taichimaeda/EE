�	�q��B@�q��B@!�q��B@	O��t�v�?O��t�v�?!O��t�v�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�q��B@6!�1�<@1�K��τ@Af��t牗?I���؉�?Y�iP4��?*	D�l���a@2F
Iterator::Model�/L�
F�?!�v`���G@)��U�3�?1��%~}C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatex�҆�Ҩ?!�Moz�'A@)?�a�'�?1��,d!�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�G7?!}F��Q�)@)�/��乎?1ozӛ�;%@:Preprocessing2U
Iterator::Model::ParallelMapV2�-�s`�?!��%~�!@)�-�s`�?1��%~�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�sE)!Xu?!���,�@)�sE)!Xu?1���,�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�<L��?!F��Q" J@)�0�*�t?1Mozӛr@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor!���'*k?!;0���@)!���'*k?1;0���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapO�Z�7ک?!���,��A@)e�z�Fw`?1Y�B��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9P��t�v�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	6!�1�<@6!�1�<@!6!�1�<@      ��!       "	�K��τ@�K��τ@!�K��τ@*      ��!       2	f��t牗?f��t牗?!f��t牗?:	���؉�?���؉�?!���؉�?B      ��!       J	�iP4��?�iP4��?!�iP4��?R      ��!       Z	�iP4��?�iP4��?!�iP4��?JGPUYP��t�v�?b �"k
Agradient_tape/sequential_23/conv2d_70/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter5.|\u%�?!5.|\u%�?">
sequential_23/conv2d_70/Relu_FusedConv2Doc_l�=�?!��U�)��?"i
@gradient_tape/sequential_23/conv2d_70/Conv2D/Conv2DBackpropInputConv2DBackpropInput2��e��?!�����?"k
Agradient_tape/sequential_23/conv2d_71/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter+)Ԉ��?!M��r3��?"k
Agradient_tape/sequential_23/conv2d_69/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter4�W�Q��?!t��9�?">
sequential_23/conv2d_71/Relu_FusedConv2D_*M+��?!���_��?"i
@gradient_tape/sequential_23/conv2d_71/Conv2D/Conv2DBackpropInputConv2DBackpropInputm��<o5�?!g�wȶ�?"9
sequential_23/dense_46/MatMulMatMul*�V�K�?!�+.��?"a
@gradient_tape/sequential_23/max_pooling2d_46/MaxPool/MaxPoolGradMaxPoolGrad!*�b�?!�|_ ���?"G
+gradient_tape/sequential_23/dense_46/MatMulMatMul-��. N�?!Y��!0��?Q      Y@Y������/@a� � U@q��\�2F@y�����?"�	
both�Your program is POTENTIALLY input-bound because 79.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�44.395% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 