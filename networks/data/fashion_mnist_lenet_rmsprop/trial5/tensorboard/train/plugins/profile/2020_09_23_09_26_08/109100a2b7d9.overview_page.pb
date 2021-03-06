�	:W���G@:W���G@!:W���G@	X΄0��?X΄0��?!X΄0��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6:W���G@$c���C@1�+J	��@A,am���?I�0|DL�?Y��q���?*	�I+*�@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�n��?!�����U@)�h�^`��?1��:fsU@:Preprocessing2F
Iterator::Model�{/�h�?!~�R�1�@)��)��z�?1���@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4��@��?!�y9Pm@)��.�5�?1�_;�x@:Preprocessing2U
Iterator::Model::ParallelMapV2��DR��?!�Y6���?)��DR��?1�Y6���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�3��X�?!�<H�(�?)�3��X�?1�<H�(�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��^)�?!��J��"W@)ݲC�Ö~?1��{�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���J̳r?!Kf@+5��?)���J̳r?1Kf@+5��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap3�-�?!��g��U@)�%��og?1����`��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 83.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9W΄0��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	$c���C@$c���C@!$c���C@      ��!       "	�+J	��@�+J	��@!�+J	��@*      ��!       2	,am���?,am���?!,am���?:	�0|DL�?�0|DL�?!�0|DL�?B      ��!       J	��q���?��q���?!��q���?R      ��!       Z	��q���?��q���?!��q���?JGPUYW΄0��?b �"l
Bgradient_tape/sequential_74/conv2d_223/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterI�;&t&�?!I�;&t&�?"j
Agradient_tape/sequential_74/conv2d_223/Conv2D/Conv2DBackpropInputConv2DBackpropInput86q�&��?!e=t���?"?
sequential_74/conv2d_223/Relu_FusedConv2D�.�K�:�?![j*ּ��?"l
Bgradient_tape/sequential_74/conv2d_224/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter+�%ʳ��?!��ȩ��?"l
Bgradient_tape/sequential_74/conv2d_222/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterX��u$�?!���y8�?"?
sequential_74/conv2d_224/Relu_FusedConv2D�þ4���?!wh�H+�?"j
Agradient_tape/sequential_74/conv2d_224/Conv2D/Conv2DBackpropInputConv2DBackpropInput2�{�?!�������?":
sequential_74/dense_148/MatMulMatMul� {.�ϙ?!��da#��?"b
Agradient_tape/sequential_74/max_pooling2d_148/MaxPool/MaxPoolGradMaxPoolGrad~��cĘ?!���Cc�?"H
,gradient_tape/sequential_74/dense_148/MatMulMatMul.�i��7�?!�3� �?Q      Y@Y߈�N�#@a䮟-V�V@qR��R�VA@y>;mQ鐮?"�	
both�Your program is POTENTIALLY input-bound because 83.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�34.6771% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 