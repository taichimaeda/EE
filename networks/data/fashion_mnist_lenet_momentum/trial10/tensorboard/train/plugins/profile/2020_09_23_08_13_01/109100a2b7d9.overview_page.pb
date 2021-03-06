�	TS�u8�@@TS�u8�@@!TS�u8�@@	a�B����?a�B����?!a�B����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6TS�u8�@@��/�1�:@1N|��8�@A�u�|�H�?I�t><K��?Y"�^F�ܾ?*	J+�FU@2F
Iterator::Modelw�$��?!`�:>;L@)4���lɚ?1�k�N�>@:Preprocessing2U
Iterator::Model::ParallelMapV2�-@�j�?!�>z�s�9@)�-@�j�?1�>z�s�9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�K�^I�?!�*؊��4@)T���f�?1髮�*>1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateAgҦ��?!�����,@)���1>�~?1�Z�ҩ�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu�BY��?!�*����E@)F�-t%u?1�W�ׇ@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���qs?!F��r�O@)���qs?1F��r�O@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR���Tj?!��KI_�@)R���Tj?1��KI_�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapTUh �͌?!ɔ�H��0@)�p��[u]?1B���� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9b�B����?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��/�1�:@��/�1�:@!��/�1�:@      ��!       "	N|��8�@N|��8�@!N|��8�@*      ��!       2	�u�|�H�?�u�|�H�?!�u�|�H�?:	�t><K��?�t><K��?!�t><K��?B      ��!       J	"�^F�ܾ?"�^F�ܾ?!"�^F�ܾ?R      ��!       Z	"�^F�ܾ?"�^F�ܾ?!"�^F�ܾ?JGPUYb�B����?b �"l
Bgradient_tape/sequential_49/conv2d_148/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�Yl�?!�Yl�?"j
Agradient_tape/sequential_49/conv2d_148/Conv2D/Conv2DBackpropInputConv2DBackpropInput��r-�"�?! _��>�?"?
sequential_49/conv2d_148/Relu_FusedConv2Dd�Y���?!Y�|yN(�?"l
Bgradient_tape/sequential_49/conv2d_149/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�7�T��?!Bf�NT��?"l
Bgradient_tape/sequential_49/conv2d_147/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��ȏx�?!S��`�?"?
sequential_49/conv2d_149/Relu_FusedConv2DR����?!�~�� ��?"j
Agradient_tape/sequential_49/conv2d_149/Conv2D/Conv2DBackpropInputConv2DBackpropInput���m��?!��d"���?"9
sequential_49/dense_98/MatMulMatMul'�&,Û?!Y������?"a
@gradient_tape/sequential_49/max_pooling2d_98/MaxPool/MaxPoolGradMaxPoolGrad�p{�ꠚ?!�ٴ���?"G
+gradient_tape/sequential_49/dense_98/MatMulMatMul�Nr�O��?!Ul8We�?Q      Y@Y�4_�g�0@a�2(&�T@q��į2H@y��-��?"�	
both�Your program is POTENTIALLY input-bound because 79.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�48.391% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 