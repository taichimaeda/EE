�	�ȓ�kH@�ȓ�kH@!�ȓ�kH@	K��!��?K��!��?!K��!��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�ȓ�kH@�đ"MC@1�N���!@A+MJA���?IܜJ���?Y��0��?*	�����e@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceٴR��?!�9���I@)ٴR��?1�9���I@:Preprocessing2F
Iterator::Model��q�哥?!��� 8@)�A|`��?1�D��B}3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�[t��z�?!A�$��N@)g����?1�.X�#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��m3�?!�JF]�#$@)xB�?�ύ?1�2���� @:Preprocessing2U
Iterator::Model::ParallelMapV2���kzP�?!�/uw+@)���kzP�?1�/uw+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8j��{�?!��D���R@)ޏ�/��x?1ݖJ�{@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorL��pvki?!���u�O�?)L��pvki?1���u�O�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�5��
�?!��c$T;O@)�@�C�b?1������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9K��!��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�đ"MC@�đ"MC@!�đ"MC@      ��!       "	�N���!@�N���!@!�N���!@*      ��!       2	+MJA���?+MJA���?!+MJA���?:	ܜJ���?ܜJ���?!ܜJ���?B      ��!       J	��0��?��0��?!��0��?R      ��!       Z	��0��?��0��?!��0��?JGPUYK��!��?b �"m
Cgradient_tape/sequential_163/conv2d_490/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��֧��?!��֧��?"m
Cgradient_tape/sequential_163/conv2d_491/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh�Zݨ�?!��BE�?"@
sequential_163/conv2d_490/Relu_FusedConv2Dǃ�mOH�?!����?"k
Bgradient_tape/sequential_163/conv2d_490/Conv2D/Conv2DBackpropInputConv2DBackpropInput��}ͯ�?!x�|��?"-
IteratorGetNext/_1_Send���N�$�?!�ǦP3�?"m
Cgradient_tape/sequential_163/conv2d_489/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��2O�?!~/"�;�?"@
sequential_163/conv2d_491/Relu_FusedConv2D��uU���?!I�y�Wj�?"k
Bgradient_tape/sequential_163/conv2d_491/Conv2D/Conv2DBackpropInputConv2DBackpropInput^�P����?!�N�uv�?"I
-gradient_tape/sequential_163/dense_326/MatMulMatMulV�y��?!��s�j�?";
sequential_163/dense_326/MatMulMatMulK��&�2�?!�$RԊ4�?Q      Y@Y�ΐ��3$@a$���yV@q���4�@@y�KK�{�?"�	
both�Your program is POTENTIALLY input-bound because 80.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�33.4469% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 