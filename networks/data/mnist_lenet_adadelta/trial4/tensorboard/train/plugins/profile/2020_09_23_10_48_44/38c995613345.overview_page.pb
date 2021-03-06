�	�ܘ��DC@�ܘ��DC@!�ܘ��DC@	��F���?��F���?!��F���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�ܘ��DC@��;#?@1�A{�@At��;�?I0o���?Y�c!:��?*	����3f@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�vR~R�?!Ud��K@)Y�n�ͷ?1��S�,J@:Preprocessing2F
Iterator::ModelC��fڦ?!Qv*!9@);%�Ρ?1B+짔3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���i�?!�>Kf��)@)hx�㮝?1�n:��$@:Preprocessing2U
Iterator::Model::ParallelMapV2ܝ��.�?!A\��d1@)ܝ��.�?1A\��d1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&ݖ��?!kbx���R@)>���4`�?1�Ǽ�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicecAJx?!��4F�
@)cAJx?1��4F�
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensore�I)��r?!,?��`�@)e�I)��r?1,?��`�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�"�J %�?!7\&s��L@)��u?Tj?1A���w��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��F���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��;#?@��;#?@!��;#?@      ��!       "	�A{�@�A{�@!�A{�@*      ��!       2	t��;�?t��;�?!t��;�?:	0o���?0o���?!0o���?B      ��!       J	�c!:��?�c!:��?!�c!:��?R      ��!       Z	�c!:��?�c!:��?!�c!:��?JGPUY��F���?b �"l
Bgradient_tape/sequential_93/conv2d_280/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter>�[���?!>�[���?"?
sequential_93/conv2d_280/Relu_FusedConv2D[ې�lͷ?!�Pd�Q��?"j
Agradient_tape/sequential_93/conv2d_280/Conv2D/Conv2DBackpropInputConv2DBackpropInput���j�~�?!�	\;Q��?"l
Bgradient_tape/sequential_93/conv2d_281/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�`(��?!�!&@	�?"l
Bgradient_tape/sequential_93/conv2d_279/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~I7�'�?!�
��dK�?"?
sequential_93/conv2d_281/Relu_FusedConv2DN��mM��?!�dETW?�?"j
Agradient_tape/sequential_93/conv2d_281/Conv2D/Conv2DBackpropInputConv2DBackpropInput[�z%�?!��񫙝�?":
sequential_93/dense_186/MatMulMatMul��H_Ś?!
+9��s�?"b
Agradient_tape/sequential_93/max_pooling2d_186/MaxPool/MaxPoolGradMaxPoolGrad�P��=��?!��N��@�?"[
0Adadelta/Adadelta/update_6/ResourceApplyAdadeltaResourceApplyAdadelta��3�ʘ?!�a����?Q      Y@Ymާ�d0@a�d���T@q<���"H@y1�N>�B�?"�	
both�Your program is POTENTIALLY input-bound because 80.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�48.2703% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 