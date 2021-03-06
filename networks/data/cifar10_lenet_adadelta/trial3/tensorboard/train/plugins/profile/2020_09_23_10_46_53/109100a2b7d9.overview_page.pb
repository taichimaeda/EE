�	B^&ţB@B^&ţB@!B^&ţB@	��kG��?��kG��?!��kG��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6B^&ţB@@��r�V<@1�c���g @A$	�P��?I-��o�c�?Y�9@0G��?*	X9�H��@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX9��v��?!��k�w!W@)�:U�g��?1�*
x�V@:Preprocessing2F
Iterator::Model��^fب?!Ym0�<)@)^�����?1X�]�
@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?n�|�b�?!�{���� @)V��6o��?1Q�=�y�?:Preprocessing2U
Iterator::Model::ParallelMapV2������?!���h�?)������?1���h�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipR,��R�?!*�L7l�W@))[$�F�?1�B��j�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��^
z?!�U� ���?)��^
z?1�U� ���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorLo.2n?!�5|2P��?)Lo.2n?1�5|2P��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���\Q��?!�d���1W@)nYk(�g?1t)x`�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 76.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��kG��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	@��r�V<@@��r�V<@!@��r�V<@      ��!       "	�c���g @�c���g @!�c���g @*      ��!       2	$	�P��?$	�P��?!$	�P��?:	-��o�c�?-��o�c�?!-��o�c�?B      ��!       J	�9@0G��?�9@0G��?!�9@0G��?R      ��!       Z	�9@0G��?�9@0G��?!�9@0G��?JGPUY��kG��?b �"l
Bgradient_tape/sequential_92/conv2d_277/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��Xl���?!��Xl���?"l
Bgradient_tape/sequential_92/conv2d_278/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterxq��n�?!6e����?"?
sequential_92/conv2d_277/Relu_FusedConv2D�Pj{x�?!r����?"j
Agradient_tape/sequential_92/conv2d_277/Conv2D/Conv2DBackpropInputConv2DBackpropInputG>�p�?!��|A��?"-
IteratorGetNext/_1_Send��W��?!�cm��i�?"l
Bgradient_tape/sequential_92/conv2d_276/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�
��أ?!R�)���?"?
sequential_92/conv2d_278/Relu_FusedConv2D�n�:��?!�2��,-�?"j
Agradient_tape/sequential_92/conv2d_278/Conv2D/Conv2DBackpropInputConv2DBackpropInputS�t�o�?!x,�)D�?"H
,gradient_tape/sequential_92/dense_184/MatMulMatMul�S��?!����C�?":
sequential_92/dense_184/MatMulMatMul�5�i!]�?!���s�?Q      Y@Y��1@a������T@q��J��:@y"`�]æ?"�	
both�Your program is POTENTIALLY input-bound because 76.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�26.9884% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 