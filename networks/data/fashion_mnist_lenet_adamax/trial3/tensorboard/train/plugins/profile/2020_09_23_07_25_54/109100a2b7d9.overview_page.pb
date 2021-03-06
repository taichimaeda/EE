�	���Z�sB@���Z�sB@!���Z�sB@	U�j�H��?U�j�H��?!U�j�H��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6���Z�sB@��f�R�=@1���߆�@A�5�:�?I��`���?Y�a0���?*	�p=
�T@2F
Iterator::Modeld�ء?!�i��E@)m<�b�Ϛ?1F��#Z@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��!p�?!P��8@)C��fڎ?1�Ok�/�2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat#�3��E�?!��O�3@)鹅�D��?1�3=0@:Preprocessing2U
Iterator::Model::ParallelMapV2��W�2ā?!����֫%@)��W�2ā?1����֫%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%vmo�$�?!j��Z�:L@)���ӝ'~?16��&d"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice5_%�t?!Ck�s@)5_%�t?1Ck�s@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorx���Ĭg?!������@)x���Ĭg?1������@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�8�t�y�?!ڛI�jj;@)��hUM`?1MD��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9T�j�H��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��f�R�=@��f�R�=@!��f�R�=@      ��!       "	���߆�@���߆�@!���߆�@*      ��!       2	�5�:�?�5�:�?!�5�:�?:	��`���?��`���?!��`���?B      ��!       J	�a0���?�a0���?!�a0���?R      ��!       Z	�a0���?�a0���?!�a0���?JGPUYT�j�H��?b �"k
Agradient_tape/sequential_32/conv2d_97/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�k"����?!�k"����?">
sequential_32/conv2d_97/Relu_FusedConv2Dd�B��ʷ?!��CG��?"i
@gradient_tape/sequential_32/conv2d_97/Conv2D/Conv2DBackpropInputConv2DBackpropInputQ2����?!�u..���?"k
Agradient_tape/sequential_32/conv2d_98/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��ns��?!|�	!"�?"k
Agradient_tape/sequential_32/conv2d_96/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�^Rc�?!�'�K�n�?">
sequential_32/conv2d_98/Relu_FusedConv2D&���©?!��NnS�?"i
@gradient_tape/sequential_32/conv2d_98/Conv2D/Conv2DBackpropInputConv2DBackpropInputSctݩy�?!K����?"9
sequential_32/dense_64/MatMulMatMul-�|#B��?!�2��
w�?"a
@gradient_tape/sequential_32/max_pooling2d_64/MaxPool/MaxPoolGradMaxPoolGrad�6���?!��|3YF�?"G
+gradient_tape/sequential_32/dense_64/MatMulMatMul���3�o�?!H����?Q      Y@Y!�B!0@a��{��T@qh0�6иH@yX��b٬?"�	
both�Your program is POTENTIALLY input-bound because 80.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�49.4439% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 