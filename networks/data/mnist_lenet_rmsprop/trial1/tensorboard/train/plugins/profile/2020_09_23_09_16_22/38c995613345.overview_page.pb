�	�f�R@0H@�f�R@0H@!�f�R@0H@	���M�e�?���M�e�?!���M�e�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�f�R@0H@A C�"D@1�j���,@Ab���LL�?I�?3���?Y[A�+��?*	_�I�X@2F
Iterator::Model�9?�q�?!ʇ�3M@)��"�ng�?1��J��F@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��Ӝ�Ȕ?!�~��P4@)�{H��ߐ?1e����}0@:Preprocessing2U
Iterator::Model::ParallelMapV2��r��?!N�ӳ�M)@)��r��?1N�ӳ�M)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ky�z�?!�B��`�*@)m7�7M�}?1�_NY�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���}Vy?!�h�h�@)���}Vy?1�h�h�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�
/�H�?!5x�H��D@)˟ov?1;�IVx�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor)[$�Fo?!��G'^�@))[$�Fo?1��G'^�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapLqU�wE�?!�v��/@)����(@d?1� 㢛�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 83.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9���M�e�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	A C�"D@A C�"D@!A C�"D@      ��!       "	�j���,@�j���,@!�j���,@*      ��!       2	b���LL�?b���LL�?!b���LL�?:	�?3���?�?3���?!�?3���?B      ��!       J	[A�+��?[A�+��?![A�+��?R      ��!       Z	[A�+��?[A�+��?![A�+��?JGPUY���M�e�?b �"l
Bgradient_tape/sequential_70/conv2d_211/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter+�q7�?!+�q7�?"?
sequential_70/conv2d_211/Relu_FusedConv2D��5��?!n���?"j
Agradient_tape/sequential_70/conv2d_211/Conv2D/Conv2DBackpropInputConv2DBackpropInput�5��+k�?!�Q�\���?"l
Bgradient_tape/sequential_70/conv2d_212/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��1\��?!������?"l
Bgradient_tape/sequential_70/conv2d_210/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterzǃ���?!�C���?"?
sequential_70/conv2d_212/Relu_FusedConv2D�1��?!�l(��?"j
Agradient_tape/sequential_70/conv2d_212/Conv2D/Conv2DBackpropInputConv2DBackpropInput�ޕ�dѤ?!|��\��?":
sequential_70/dense_140/MatMulMatMul��큮��?!s(��A��?"b
Agradient_tape/sequential_70/max_pooling2d_140/MaxPool/MaxPoolGradMaxPoolGradU���r�?!��C��j�?"H
,gradient_tape/sequential_70/dense_140/MatMulMatMulzS�;�?!�܉��?Q      Y@Y� ��U�#@a��Gu�V@qzro�(gN@y�)�����?"�	
both�Your program is POTENTIALLY input-bound because 83.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�60.8059% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 