�	�A
��(f@�A
��(f@!�A
��(f@	�i#S*�?�i#S*�?!�i#S*�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�A
��(f@����Y@1 %vmo�F@A��z�?I2�m�׮;@Y R�8���?*	&1��`@2F
Iterator::ModelЛ�T[�?!t��y��G@)���H�?1� ʺ�A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata3�ٲ�?!9@=��4@)�u��S�?1��|;��0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate#ڎ����?!���J4@)O$�jf-�?1(��i*�.@:Preprocessing2U
Iterator::Model::ParallelMapV2�Sr3ܐ?!���m(@)�Sr3ܐ?1���m(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��o�4(�?!�t:�NJ@)S=��M�?1B�Ұ�#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceR���Tz?!|F����@)R���Tz?1|F����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@�z��{u?!���@)@�z��{u?1���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�1 Ǟ?!>Q�vIK6@)*�"�h?1V$����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 58.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�15.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�i#S*�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����Y@����Y@!����Y@      ��!       "	 %vmo�F@ %vmo�F@! %vmo�F@*      ��!       2	��z�?��z�?!��z�?:	2�m�׮;@2�m�׮;@!2�m�׮;@B      ��!       J	 R�8���? R�8���?! R�8���?R      ��!       Z	 R�8���? R�8���?! R�8���?JGPUY�i#S*�?b �"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropP<w^%��?!P<w^%��?"&
CudnnRNNCudnnRNN������?!�ɺ��>�?"J
/sequential_52/embedding_52/embedding_lookup/_21_SendG�Wp|��?!�(|�7��?"`
Egradient_tape/sequential_52/embedding_52/embedding_lookup/Reshape/_45_Send�hh��?!vzLR)�?"(
gradients/AddNAddN�C{ˢf?! ���?�?"*
transpose_9	TransposeO���#c?!o���S�?"C
$gradients/transpose_9_grad/transpose	Transpose��A��b?!vU�Nf�?"A
"gradients/transpose_grad/transpose	Transpose��x�T?!�_ފ[p�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGradO���#S?!aJ�_�y�?"L
3gradient_tape/sequential_52/dropout_104/dropout/MulMul13���JH?!��R ��?Q      Y@Yw��Q��@a$�q��CX@q�zCdo;@y�.�7g�?"�
both�Your program is POTENTIALLY input-bound because 58.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�15.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�27.4351% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 