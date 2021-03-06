�	"9��UNH@"9��UNH@!"9��UNH@	���E���?���E���?!���E���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6"9��UNH@�X�C@1�+,�� @A� ��ǧ?I������?Y��
��?*	㥛� ��@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate=HO�C��?!`����NW@)�]L3ݫ�?1�+]��,W@:Preprocessing2F
Iterator::Model�L����?!��H�S�@)�Nϻ���?1����*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(B�v��?!{r���?)O�S�{F�?1�Qȃ}g�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�z�"0V�?![��auX@)g�CV�?1N��]8�?:Preprocessing2U
Iterator::Model::ParallelMapV2��Gߤi�?!�k���?)��Gߤi�?1�k���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��^fx?!��+�}��?)��^fx?1��+�}��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorP�s'�m?!<��Jڀ�?)P�s'�m?1<��Jڀ�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�	/����?!�ͷu^W@).8��_�f?1p3�L��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9���E���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�X�C@�X�C@!�X�C@      ��!       "	�+,�� @�+,�� @!�+,�� @*      ��!       2	� ��ǧ?� ��ǧ?!� ��ǧ?:	������?������?!������?B      ��!       J	��
��?��
��?!��
��?R      ��!       Z	��
��?��
��?!��
��?JGPUY���E���?b �"m
Cgradient_tape/sequential_161/conv2d_484/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterp@ib� �?!p@ib� �?"m
Cgradient_tape/sequential_161/conv2d_485/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterq;����?!��w\w�?"@
sequential_161/conv2d_484/Relu_FusedConv2D���U�?!�'�L�?"k
Bgradient_tape/sequential_161/conv2d_484/Conv2D/Conv2DBackpropInputConv2DBackpropInputx�d0
��?!a�@n#��?"-
IteratorGetNext/_1_Send�R|!i.�?!�qp����?"m
Cgradient_tape/sequential_161/conv2d_483/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��ˡ�B�?!o�Tc�$�?"@
sequential_161/conv2d_485/Relu_FusedConv2D��`%d �?!]����T�?"k
Bgradient_tape/sequential_161/conv2d_485/Conv2D/Conv2DBackpropInputConv2DBackpropInput����ݠ?!WZ��b�?"I
-gradient_tape/sequential_161/dense_322/MatMulMatMul��3���?!�({	X�?";
sequential_161/dense_322/MatMulMatMul�cn���?!�fs�$�?Q      Y@Y�ΐ��3$@a$���yV@q��L��>@yS�o����?"�	
both�Your program is POTENTIALLY input-bound because 80.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�30.8632% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 