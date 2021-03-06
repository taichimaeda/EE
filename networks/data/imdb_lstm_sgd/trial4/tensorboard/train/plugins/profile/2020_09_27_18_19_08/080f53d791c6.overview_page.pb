�	�b�B�f@�b�B�f@!�b�B�f@	s�����?s�����?!s�����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�b�B�f@ |(�yX@1J��F@AD�.l�V�?I��Ơ\B@Y���=�$�?*	]���(b@2F
Iterator::Model�ǁW˭?!��'D@);S�Ʀ?1���>��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�%r���?!zIx@��=@)f/�N[�?1_�`/:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�Z_$��?!�I��3@).�R���?1��$�+@:Preprocessing2U
Iterator::Model::ParallelMapV2�g�ej�?!<�����"@)�g�ej�?1<�����"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�N�j�?!�a���M@)�b�=y�?1�뿴� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceW�"���?!���tj@)W�"���?1���tj@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��2Wu?!ݸтq@)��2Wu?1ݸтq@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���*l�?!�U0��5@)����!9i?1�0�3@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 54.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�20.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9s�����?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	 |(�yX@ |(�yX@! |(�yX@      ��!       "	J��F@J��F@!J��F@*      ��!       2	D�.l�V�?D�.l�V�?!D�.l�V�?:	��Ơ\B@��Ơ\B@!��Ơ\B@B      ��!       J	���=�$�?���=�$�?!���=�$�?R      ��!       Z	���=�$�?���=�$�?!���=�$�?JGPUYs�����?b �"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropͿ�A�?!Ϳ�A�?"&
CudnnRNNCudnnRNNJ��X�?!���s��?"J
/sequential_83/embedding_83/embedding_lookup/_13_SendѡԑÆ�?!+!U��k�?"`
Egradient_tape/sequential_83/embedding_83/embedding_lookup/Reshape/_29_Send3�ڷW�?!l�4���?"(
gradients/AddNAddN��S8�f?!Q=�C~��?"C
$gradients/transpose_9_grad/transpose	TransposeoI!X�>c?!�^�ϼ�?"*
transpose_9	Transpose�^�!c?!�w}���?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad'w�J�S?!�׌��#�?"A
"gradients/transpose_grad/transpose	TransposeoI!X�>S?!Z��9-�?"L
3gradient_tape/sequential_83/dropout_166/dropout/MulMul���4�lH?!LU3�?Q      Y@Y��ތ�?a� �̹X@qV��a�9@y�dW0��{?"�
both�Your program is POTENTIALLY input-bound because 54.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�20.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�25.8296% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 