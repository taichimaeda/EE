�	?�g͏=H@?�g͏=H@!?�g͏=H@	���c(�?���c(�?!���c(�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?�g͏=H@����XC@1=Զa!@A�B�_�+�?IOqN`�?Y���[���?*	�(\���e@2U
Iterator::Model::ParallelMapV2G��Ȯ��?!�u�ywK@)G��Ȯ��?1�u�ywK@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateC�Y����?!�~0��2@)������?1�VNx��/@:Preprocessing2F
Iterator::Model���?�?!��).&P@)�z�ۡa�?1v�y��R#@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatN��1�M�?!��Ҍ�u%@)��26t��?1`ʶ/�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/��د?!���ͳA@)�����L�?1���, @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���m3u?!��J��@)���m3u?1��J��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�����k?!;�ߠ۵�?)�����k?1;�ߠ۵�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapϺFˁ�?!�Ƚ��$4@)Έ���c?1F����,�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 79.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9���c(�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����XC@����XC@!����XC@      ��!       "	=Զa!@=Զa!@!=Զa!@*      ��!       2	�B�_�+�?�B�_�+�?!�B�_�+�?:	OqN`�?OqN`�?!OqN`�?B      ��!       J	���[���?���[���?!���[���?R      ��!       Z	���[���?���[���?!���[���?JGPUY���c(�?b �"m
Cgradient_tape/sequential_162/conv2d_487/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�~Ĩ��?!�~Ĩ��?"m
Cgradient_tape/sequential_162/conv2d_488/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter`�D|��?!ֶfe�@�?"@
sequential_162/conv2d_487/Relu_FusedConv2D�GQsD�?!��9��?"k
Bgradient_tape/sequential_162/conv2d_487/Conv2D/Conv2DBackpropInputConv2DBackpropInput�5�ߩ�?!� 1'��?"-
IteratorGetNext/_1_Send��>�r��?!����C��?"m
Cgradient_tape/sequential_162/conv2d_486/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter2'�
�9�?!�@�(�?"@
sequential_162/conv2d_488/Relu_FusedConv2D���O��?!���4W�?"k
Bgradient_tape/sequential_162/conv2d_488/Conv2D/Conv2DBackpropInputConv2DBackpropInput�<����?!�CQf&c�?"I
-gradient_tape/sequential_162/dense_324/MatMulMatMul��K\��?!��3Q�W�?";
sequential_162/dense_324/MatMulMatMul%#Y�0�?!�kN�-!�?Q      Y@Y�ΐ��3$@a$���yV@q(�<@@y?$	��?"�	
both�Your program is POTENTIALLY input-bound because 79.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�32.4695% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 