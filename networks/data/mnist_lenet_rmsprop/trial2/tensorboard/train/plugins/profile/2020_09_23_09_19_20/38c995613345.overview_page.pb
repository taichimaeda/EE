�	Zd;�5G@Zd;�5G@!Zd;�5G@	IB-O���?IB-O���?!IB-O���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Zd;�5G@G�0}�!C@1�:�4@A6�U���?I��5!���?YK�!q���?*	P��n�~@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��Τ�?!9^LU@){��a�?1�`���U@:Preprocessing2F
Iterator::Model����Y��?!D恣�^!@)����s�?1q&�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�r��{�?!L�o�5�@)Iط���?151�cnn
@:Preprocessing2U
Iterator::Model::ParallelMapV2:X��0_~?!Z���F�?):X��0_~?1Z���F�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip
��O��?!7Ï�!�V@)l��[{?1o����?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU�����u?!�&l
�K�?)U�����u?1�&l
�K�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor �M���p?![�RB��?) �M���p?1[�RB��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_�\6:�?!
i�mD@)����[_?1�k����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 82.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9JB-O���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	G�0}�!C@G�0}�!C@!G�0}�!C@      ��!       "	�:�4@�:�4@!�:�4@*      ��!       2	6�U���?6�U���?!6�U���?:	��5!���?��5!���?!��5!���?B      ��!       J	K�!q���?K�!q���?!K�!q���?R      ��!       Z	K�!q���?K�!q���?!K�!q���?JGPUYJB-O���?b �"l
Bgradient_tape/sequential_71/conv2d_214/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�#|-�?!�#|-�?"?
sequential_71/conv2d_214/Relu_FusedConv2Dn�9m"��?!;�?"j
Agradient_tape/sequential_71/conv2d_214/Conv2D/Conv2DBackpropInputConv2DBackpropInput�mXj)�?!���3���?"l
Bgradient_tape/sequential_71/conv2d_215/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��¡���?!������?"l
Bgradient_tape/sequential_71/conv2d_213/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter$d}ۨ?!; E>B	�?"?
sequential_71/conv2d_215/Relu_FusedConv2D6���z�?!��U��?"j
Agradient_tape/sequential_71/conv2d_215/Conv2D/Conv2DBackpropInputConv2DBackpropInput��=Z�?!��/LW��?":
sequential_71/dense_142/MatMulMatMul��I���?!�/x����?"b
Agradient_tape/sequential_71/max_pooling2d_142/MaxPool/MaxPoolGradMaxPoolGrad/�����?!B�^g�l�?"H
,gradient_tape/sequential_71/dense_142/MatMulMatMulٛrS�Z�?!!���q�?Q      Y@Y� ��U�#@a��Gu�V@q-xF+�I@y�gA���?"�	
both�Your program is POTENTIALLY input-bound because 82.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�51.4544% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 