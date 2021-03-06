�	�mO��,A@�mO��,A@!�mO��,A@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�mO��,A@��I�;@1�3���@A�	�8��?I��ฌ��?*	effff~Y@2F
Iterator::Model��e��a�?!R�؞�NH@)KW��x��?1K�d���B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateݵ�|г�?!��h�&�8@)�Gqh�?1��oÊ3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath�ej��?!��@���0@)�B����?1�E��H,@:Preprocessing2U
Iterator::Model::ParallelMapV2��֦���?!,ϕ#�%@)��֦���?1,ϕ#�%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice5��-</u?!z	'�I@)5��-</u?1z	'�I@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��<,Ԫ?!�9'aL�I@)���!��t?1t^шD@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/�e����?!�V��e=@)H��'��s?1t]�yT"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoreo)狽g?!�i�'�@)eo)狽g?1�i�'�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 78.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��I�;@��I�;@!��I�;@      ��!       "	�3���@�3���@!�3���@*      ��!       2	�	�8��?�	�8��?!�	�8��?:	��ฌ��?��ฌ��?!��ฌ��?B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �"l
Bgradient_tape/sequential_44/conv2d_133/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter{���?!{���?"j
Agradient_tape/sequential_44/conv2d_133/Conv2D/Conv2DBackpropInputConv2DBackpropInput�ԏ0�?!�2+y�?"?
sequential_44/conv2d_133/Relu_FusedConv2DN�����?!R��@1��?"l
Bgradient_tape/sequential_44/conv2d_134/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�f�,ͱ?!���wn�?"?
sequential_44/conv2d_134/Relu_FusedConv2Dd�`���?!
�ٽP��?"l
Bgradient_tape/sequential_44/conv2d_132/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�נ\a�?!��t�x�?"j
Agradient_tape/sequential_44/conv2d_134/Conv2D/Conv2DBackpropInputConv2DBackpropInput�3TKs��?!86l����?"9
sequential_44/dense_88/MatMulMatMul^%"�pC�?!cG�0���?"a
@gradient_tape/sequential_44/max_pooling2d_88/MaxPool/MaxPoolGradMaxPoolGrad�K^K�?!
���{e�?"M
/gradient_tape/sequential_44/conv2d_133/ReluGradReluGrad���˘?!�h p�+�?Q      Y@Y�4_�g�0@a�2(&�T@q��TNv�Q@y�h����?"�
both�Your program is POTENTIALLY input-bound because 78.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�71.3041% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 