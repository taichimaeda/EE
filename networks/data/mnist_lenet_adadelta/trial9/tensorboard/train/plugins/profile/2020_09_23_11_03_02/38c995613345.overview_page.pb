�	ٕ��TD@ٕ��TD@!ٕ��TD@	w�NPe�?w�NPe�?!w�NPe�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ٕ��TD@dY0�@@1���9�@A �4�O�?I�~��Γ�?Y˟o��?*	X9��z�@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateO"¿�?!�]���EU@)࢓����?1qXւwT@:Preprocessing2F
Iterator::Model�DR���?!,M�=DP@)�W<�H��?1T��p�@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�,��;�?!FT��Sr@)�9� U�?1a���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�
����?!R�����	@)�
����?1R�����	@:Preprocessing2U
Iterator::Model::ParallelMapV2��Ss���?!b�M#�?)��Ss���?1b�M#�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS��.q�?!-�'��
W@)ٳ�25	~?1ǲ��<��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZ_&�p?!��7���?)Z_&�p?1��7���?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapjhwH1�?!�E�ydU@)����?g?1]�F��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 81.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9w�NPe�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	dY0�@@dY0�@@!dY0�@@      ��!       "	���9�@���9�@!���9�@*      ��!       2	 �4�O�? �4�O�?! �4�O�?:	�~��Γ�?�~��Γ�?!�~��Γ�?B      ��!       J	˟o��?˟o��?!˟o��?R      ��!       Z	˟o��?˟o��?!˟o��?JGPUYw�NPe�?b �"l
Bgradient_tape/sequential_98/conv2d_295/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter]�+�*�?!]�+�*�?"?
sequential_98/conv2d_295/Relu_FusedConv2Dr��`�ݷ?!�Ѣ���?"j
Agradient_tape/sequential_98/conv2d_295/Conv2D/Conv2DBackpropInputConv2DBackpropInputk"�v`K�?!&�F���?"l
Bgradient_tape/sequential_98/conv2d_296/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��9@~��?!NU�	�?"l
Bgradient_tape/sequential_98/conv2d_294/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter����ܪ?!φ���v�?"?
sequential_98/conv2d_296/Relu_FusedConv2D����?!f���BZ�?"j
Agradient_tape/sequential_98/conv2d_296/Conv2D/Conv2DBackpropInputConv2DBackpropInput�^&��?!P=;p��?":
sequential_98/dense_196/MatMulMatMul�bϜ>�?!g�#0ʑ�?"b
Agradient_tape/sequential_98/max_pooling2d_196/MaxPool/MaxPoolGradMaxPoolGrad؎ �2��?!ސ���^�?"[
0Adadelta/Adadelta/update_6/ResourceApplyAdadeltaResourceApplyAdadelta?����2�?!�-[�q �?Q      Y@Ymާ�d0@a�d���T@q_�\OjJF@y(R�nT�?"�	
both�Your program is POTENTIALLY input-bound because 81.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�44.5814% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 