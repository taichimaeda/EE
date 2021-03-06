�	�-y<kC@�-y<kC@!�-y<kC@	������?������?!������?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�-y<kC@�����=@17+1� @A/��C�?Ia3��2�?Y�GS=��?*	G�z�3z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatJ����?!oiZ�P(U@)4��s�?1K��J��T@:Preprocessing2F
Iterator::Model�-�l�I�?!PHj�B�$@)�u?T�?1�D�+�@:Preprocessing2U
Iterator::Model::ParallelMapV2�බ�?!����R@)�බ�?1����R@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate
H�`��?!ʻ�Ud�@)i��>�Q~?1�}��?�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��)1	w?!��S�v�?)��)1	w?1��S�v�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip]�Fx�?!��ҧwgV@)��8���v?1�Z5�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�bFx{p?!ڑQ�!��?)�bFx{p?1ڑQ�!��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��5�Ko�?!T?�#6J@)S[� �c?1(&8G��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 76.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9������?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����=@�����=@!�����=@      ��!       "	7+1� @7+1� @!7+1� @*      ��!       2	/��C�?/��C�?!/��C�?:	a3��2�?a3��2�?!a3��2�?B      ��!       J	�GS=��?�GS=��?!�GS=��?R      ��!       Z	�GS=��?�GS=��?!�GS=��?JGPUY������?b �"m
Cgradient_tape/sequential_116/conv2d_349/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter���]�A�?!���]�A�?"m
Cgradient_tape/sequential_116/conv2d_350/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterƼYy"��?!��I��]�?"@
sequential_116/conv2d_349/Relu_FusedConv2DS��MU�?!�74"s�?"k
Bgradient_tape/sequential_116/conv2d_349/Conv2D/Conv2DBackpropInputConv2DBackpropInputRL؂ ��?!��T�b�?"-
IteratorGetNext/_1_Send �b5>V�?!>�E�9��?"m
Cgradient_tape/sequential_116/conv2d_348/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterx8:TUI�?!��fF�@�?"@
sequential_116/conv2d_350/Relu_FusedConv2D��� ���?!�Nv�n��?"k
Bgradient_tape/sequential_116/conv2d_350/Conv2D/Conv2DBackpropInputConv2DBackpropInput'���ֿ�?!8rVl��?"I
-gradient_tape/sequential_116/dense_232/MatMulMatMul�}M�<�?!��6��?";
sequential_116/dense_232/MatMulMatMul��w�Ě?!|�ԋ^v�?Q      Y@Y�4_�g�0@a�2(&�T@qK�bHG1@yν̿��?"�	
both�Your program is POTENTIALLY input-bound because 76.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�17.2784% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 