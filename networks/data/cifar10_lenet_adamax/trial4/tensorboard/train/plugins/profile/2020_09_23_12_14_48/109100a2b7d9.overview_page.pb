�	�����jC@�����jC@!�����jC@	p��.V~�?p��.V~�?!p��.V~�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�����jC@:[@h=�=@1�.4�i< @AՖ:����?IW$&��[�?Y�������?*	m����W@2F
Iterator::Model��q4GV�?!!��
G@)����	�?1��}3@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[@h=|��?!M�l.`9@)3Mg'��?1F�Ia� 4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ӝ'��?!gZ��<4@)5E�ӻx�?1(%=�d;0@:Preprocessing2U
Iterator::Model::ParallelMapV2øDkE�?!� 5�!,@)øDkE�?1� 5�!,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipsePmp"�?!���th�J@)! _B�w?1y:'��D@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��lXSYt?!�4��@)��lXSYt?1�4��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorÜ�Mo?!��l�@)Ü�Mo?1��l�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�G5��Ě?!�����;@))<hv�[a?1W�X4��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 76.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9q��.V~�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	:[@h=�=@:[@h=�=@!:[@h=�=@      ��!       "	�.4�i< @�.4�i< @!�.4�i< @*      ��!       2	Ֆ:����?Ֆ:����?!Ֆ:����?:	W$&��[�?W$&��[�?!W$&��[�?B      ��!       J	�������?�������?!�������?R      ��!       Z	�������?�������?!�������?JGPUYq��.V~�?b �"m
Cgradient_tape/sequential_123/conv2d_370/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteraz>Q0��?!az>Q0��?"m
Cgradient_tape/sequential_123/conv2d_371/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter������?!2'
J��?"@
sequential_123/conv2d_370/Relu_FusedConv2DGMƙpA�?!��{p"�?"k
Bgradient_tape/sequential_123/conv2d_370/Conv2D/Conv2DBackpropInputConv2DBackpropInputozms㋳?! WM �?"-
IteratorGetNext/_1_Send�#�
�}�?!"Џ_�?"m
Cgradient_tape/sequential_123/conv2d_369/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����?!�zF���?"@
sequential_123/conv2d_371/Relu_FusedConv2D����ܣ?!C7ep.�?"k
Bgradient_tape/sequential_123/conv2d_371/Conv2D/Conv2DBackpropInputConv2DBackpropInput���˳�?!�ǏͬI�?"I
-gradient_tape/sequential_123/dense_246/MatMulMatMulH�Ƣ��?!r7��J�?";
sequential_123/dense_246/MatMulMatMul&x�i�{�?!3;��?Q      Y@Y������0@aVUUUU�T@q�����7@y+�Etg�?"�	
both�Your program is POTENTIALLY input-bound because 76.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�23.5667% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 