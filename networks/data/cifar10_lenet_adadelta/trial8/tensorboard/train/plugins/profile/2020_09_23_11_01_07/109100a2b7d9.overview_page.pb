�	q��[uyB@q��[uyB@!q��[uyB@	��5�?��5�?!��5�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6q��[uyB@@Qٰ�
<@1K���Jd @A�T�z��?I��w}�,�?Y����8�?*	ףp=�a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�2��?!��qQ.=@)>x�҆â?1[�n-��9@:Preprocessing2F
Iterator::Model/0+�~�?!a���&?@)q㊋��?1>C���9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��A{�?!�V�[K�=@)���C�?1PH���2@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-_��?ݐ?!�N�N'@)-_��?ݐ?1�N�N'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipе/��?!�AX6:Q@)��o{�Ć?1˖[PWw@:Preprocessing2U
Iterator::Model::ParallelMapV2�@J��~?!���bR@)�@J��~?1���bR@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor5ӽN��r?!}� Y�	@)5ӽN��r?1}� Y�	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB@��
�?!dH};��?@)�P�,i?1b���^e@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 75.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��5�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	@Qٰ�
<@@Qٰ�
<@!@Qٰ�
<@      ��!       "	K���Jd @K���Jd @!K���Jd @*      ��!       2	�T�z��?�T�z��?!�T�z��?:	��w}�,�?��w}�,�?!��w}�,�?B      ��!       J	����8�?����8�?!����8�?R      ��!       Z	����8�?����8�?!����8�?JGPUY��5�?b �"l
Bgradient_tape/sequential_97/conv2d_292/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter� �����?!� �����?"l
Bgradient_tape/sequential_97/conv2d_293/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����$u�?!�wv��?"?
sequential_97/conv2d_292/Relu_FusedConv2DPr+ѣ"�?!���j���?"j
Agradient_tape/sequential_97/conv2d_292/Conv2D/Conv2DBackpropInputConv2DBackpropInput%����t�?!�s�����?"-
IteratorGetNext/_1_Send��7zM�?!!w�eQ_�?"l
Bgradient_tape/sequential_97/conv2d_291/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����.�?!��ҟ���?"?
sequential_97/conv2d_293/Relu_FusedConv2Dޡ�ź�?!���w)�?"j
Agradient_tape/sequential_97/conv2d_293/Conv2D/Conv2DBackpropInputConv2DBackpropInput�S"��?!�
B�?"H
,gradient_tape/sequential_97/dense_194/MatMulMatMulX�5���?!q-�sB�?":
sequential_97/dense_194/MatMulMatMul)�q:V�?!f��y%�?Q      Y@Y��1@a������T@qv��
A@y�1Fd��?"�	
both�Your program is POTENTIALLY input-bound because 75.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�34.9925% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 