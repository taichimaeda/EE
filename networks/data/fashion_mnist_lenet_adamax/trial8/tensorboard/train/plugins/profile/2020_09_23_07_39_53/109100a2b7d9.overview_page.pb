�	%��ID B@%��ID B@!%��ID B@	6�5k��?6�5k��?!6�5k��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6%��ID B@Q/�4'�<@1��K�@A�9y�	��?I���ْ��?Y�r��?*	��ʡmT@2F
Iterator::Model�(��/��?!�q���cF@)ٖg)Y�?1�� �"B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!���ycx5@)�=\r�)�?1R���Em1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateSy=��?!7l����2@)w��N#-�?1���N)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipù��?!#�w�K@)u�~?1a���v"@:Preprocessing2U
Iterator::Model::ParallelMapV2�鲘�||?!/��!@)�鲘�||?1/��!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapI�\߇��?!�-��+�8@)�����s?1��I��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��~���s?!���CP�@)��~���s?1���CP�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor=�E~�k?!���\v,@)=�E~�k?1���\v,@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no97�5k��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Q/�4'�<@Q/�4'�<@!Q/�4'�<@      ��!       "	��K�@��K�@!��K�@*      ��!       2	�9y�	��?�9y�	��?!�9y�	��?:	���ْ��?���ْ��?!���ْ��?B      ��!       J	�r��?�r��?!�r��?R      ��!       Z	�r��?�r��?!�r��?JGPUY7�5k��?b �"l
Bgradient_tape/sequential_37/conv2d_112/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterG&�J�?!G&�J�?"j
Agradient_tape/sequential_37/conv2d_112/Conv2D/Conv2DBackpropInputConv2DBackpropInputh^��è�?!{վ2v��?"?
sequential_37/conv2d_112/Relu_FusedConv2D5i~�Ã�?!,��?"l
Bgradient_tape/sequential_37/conv2d_113/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterTn�Qy�?!� ���+�?"l
Bgradient_tape/sequential_37/conv2d_111/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter-��p��?!�ߝ>u{�?"?
sequential_37/conv2d_113/Relu_FusedConv2D������?!�^j+cY�?"j
Agradient_tape/sequential_37/conv2d_113/Conv2D/Conv2DBackpropInputConv2DBackpropInput&5���z�?!Nr���?"9
sequential_37/dense_74/MatMulMatMul��Q��?!�J�G�y�?"a
@gradient_tape/sequential_37/max_pooling2d_74/MaxPool/MaxPoolGradMaxPoolGrad�>lJa��?!���Q�I�?"G
+gradient_tape/sequential_37/dense_74/MatMulMatMul�K���?!��gAH�?Q      Y@Y!�B!0@a��{��T@qY	���D@ysQ��f��?"�	
both�Your program is POTENTIALLY input-bound because 79.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�41.2786% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 