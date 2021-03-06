�	�(#.� d@�(#.� d@!�(#.� d@	0Sa��W�?0Sa��W�?!0Sa��W�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�(#.� d@>�n�K�U@1=�K�ebF@A�ui���?I%�S;�;@Y��:ǀ��?*	H�z��_@2F
Iterator::Model���@�m�?!�\�nCH@),*�t���?1�����A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat� [���?!QS�_5@)��Ȯ���?1�+􊴺/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��6S!�?!�}��*d3@)5���k�?1(&��-@:Preprocessing2U
Iterator::Model::ParallelMapV2IG9�M��?!nZ-ʼ+@)IG9�M��?1nZ-ʼ+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��O��?!f��/��I@)h�>��?1�ob ~�!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_���|?!�!(6�	@)��_���|?1�!(6�	@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq:�v?!T�k{Y�@)q:�v?1T�k{Y�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]n0�a��?!�'?5@)}�:c?1e)Ǫ���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�17.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no90Sa��W�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	>�n�K�U@>�n�K�U@!>�n�K�U@      ��!       "	=�K�ebF@=�K�ebF@!=�K�ebF@*      ��!       2	�ui���?�ui���?!�ui���?:	%�S;�;@%�S;�;@!%�S;�;@B      ��!       J	��:ǀ��?��:ǀ��?!��:ǀ��?R      ��!       Z	��:ǀ��?��:ǀ��?!��:ǀ��?JGPUY0Sa��W�?b �"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop ��Jg�?! ��Jg�?"&
CudnnRNNCudnnRNNlEb��?!0�(�V��?"J
/sequential_14/embedding_14/embedding_lookup/_13_SendF94r|�?!�m�Hc�?"`
Egradient_tape/sequential_14/embedding_14/embedding_lookup/Reshape/_31_Send1���¤�?!��4����?"(
gradients/AddNAddN!d��h8g?!B�����?"C
$gradients/transpose_9_grad/transpose	Transpose�����c?!Qx��� �?"*
transpose_9	Transpose����c?!�K��?"A
"gradients/transpose_grad/transpose	Transposew_c�c9T?!�@����?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGradM��6�yS?!���Ͱ'�?"?
&sequential_14/dropout_28/dropout/Mul_1Mul���K?!Ɇ#��.�?Q      Y@Y�;�� �?ax�A|�X@q1K�}YD@y��{X�|?"�
both�Your program is POTENTIALLY input-bound because 54.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�17.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�40.6954% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 