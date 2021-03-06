�	*�-90B@*�-90B@!*�-90B@	�drR��?�drR��?!�drR��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6*�-90B@�K��Ϥ=@1�}:3�@A�~�~�d�?I���Co�?Y7QKs+��?*	gffff]�@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceӅX��?!�6^�CS@)ӅX��?1�6^�CS@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�k����?!>Yۛ�j"@)��9� �?1�~�b!@:Preprocessing2F
Iterator::Model��p�q�?!�+*�-�"@)r��rg&�?1��-¬�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate� 5�l��?!��ŘT�S@)���m�?1��Q�J@:Preprocessing2U
Iterator::Model::ParallelMapV2�e���-�?!˩M�\}@)�e���-�?1˩M�\}@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�a��h��?!��:LڤV@)}\*��{?1�$Yi���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�/K;5�k?!l��]Ή�?)�/K;5�k?1l��]Ή�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�D�eݿ�?!���oT@)��Cl�pb?1��TR�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 81.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�drR��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�K��Ϥ=@�K��Ϥ=@!�K��Ϥ=@      ��!       "	�}:3�@�}:3�@!�}:3�@*      ��!       2	�~�~�d�?�~�~�d�?!�~�~�d�?:	���Co�?���Co�?!���Co�?B      ��!       J	7QKs+��?7QKs+��?!7QKs+��?R      ��!       Z	7QKs+��?7QKs+��?!7QKs+��?JGPUY�drR��?b �"k
Agradient_tape/sequential_26/conv2d_79/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��t���?!��t���?">
sequential_26/conv2d_79/Relu_FusedConv2D;�!��?!/Yv�?"i
@gradient_tape/sequential_26/conv2d_79/Conv2D/Conv2DBackpropInputConv2DBackpropInputOo=��?!u�����?"k
Agradient_tape/sequential_26/conv2d_80/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�1�='j�?!r��]U�?">
sequential_26/conv2d_80/Relu_FusedConv2DW�;�
�?!�4����?"k
Agradient_tape/sequential_26/conv2d_78/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterj��)�Щ?!�tbh�?"i
@gradient_tape/sequential_26/conv2d_80/Conv2D/Conv2DBackpropInputConv2DBackpropInputۅ�T�?!C}%���?"9
sequential_26/dense_52/MatMulMatMul���c^6�?!��^��?"a
@gradient_tape/sequential_26/max_pooling2d_52/MaxPool/MaxPoolGradMaxPoolGrad�\����?!�ԕwo�?"L
.gradient_tape/sequential_26/conv2d_79/ReluGradReluGrad�^Ö��?!��K4\6�?Q      Y@Y������/@a� � U@q�{@���F@yPX�!��?"�	
both�Your program is POTENTIALLY input-bound because 81.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�45.6314% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 