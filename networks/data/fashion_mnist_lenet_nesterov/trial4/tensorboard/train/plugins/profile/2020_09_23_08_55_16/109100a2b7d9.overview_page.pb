�	�SW>CA@�SW>CA@!�SW>CA@	�ʟ�/O�?�ʟ�/O�?!�ʟ�/O�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�SW>CA@���e�;@1=b�ܢ@A˄_��M�?I¢"N'��?Y������?*	ףp=��@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����?!P����S@)���ᱟ�?1\��<d�S@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+�z���?!Go��&@)��Z��?1��`�%@:Preprocessing2F
Iterator::Model�A�۽ܧ?!�3�!��@)�;P�<��?1�'���@:Preprocessing2U
Iterator::Model::ParallelMapV2� ���?!*0�s�~�?)� ���?1*0�s�~�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�lu9%��?!���}�%W@)-��VЄ?1G��1���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceO崧�x?!&z8"S��?)O崧�x?1&z8"S��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor,���cZk?!𛽔��?),���cZk?1𛽔��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapM�O��?!0#j ��S@)臭���c?1��,�=_�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�ʟ�/O�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���e�;@���e�;@!���e�;@      ��!       "	=b�ܢ@=b�ܢ@!=b�ܢ@*      ��!       2	˄_��M�?˄_��M�?!˄_��M�?:	¢"N'��?¢"N'��?!¢"N'��?B      ��!       J	������?������?!������?R      ��!       Z	������?������?!������?JGPUY�ʟ�/O�?b �"l
Bgradient_tape/sequential_63/conv2d_190/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�K�{�5�?!�K�{�5�?"j
Agradient_tape/sequential_63/conv2d_190/Conv2D/Conv2DBackpropInputConv2DBackpropInput��*^�?!i{r|2�?"?
sequential_63/conv2d_190/Relu_FusedConv2D�G�t�?!^Q��~�?"l
Bgradient_tape/sequential_63/conv2d_191/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter-9��?!`��#t�?"?
sequential_63/conv2d_191/Relu_FusedConv2D�el�Y0�?! f]��?"l
Bgradient_tape/sequential_63/conv2d_189/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter$�����?!l���s~�?"j
Agradient_tape/sequential_63/conv2d_191/Conv2D/Conv2DBackpropInputConv2DBackpropInput��Ғ�?!����?":
sequential_63/dense_126/MatMulMatMul(�;@?�?!E�[����?"b
Agradient_tape/sequential_63/max_pooling2d_126/MaxPool/MaxPoolGradMaxPoolGradhX�,��?!��n!�h�?"M
/gradient_tape/sequential_63/conv2d_190/ReluGradReluGrad�����?!f��Q�.�?Q      Y@Y�4_�g�0@a�2(&�T@q�t�PD@y~�S'�?"�	
both�Your program is POTENTIALLY input-bound because 80.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�40.6292% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 