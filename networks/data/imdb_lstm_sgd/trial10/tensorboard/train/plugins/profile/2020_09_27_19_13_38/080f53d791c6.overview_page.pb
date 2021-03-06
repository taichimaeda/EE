�	6����h@6����h@!6����h@	��7$�'�?��7$�'�?!��7$�'�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails66����h@��Q�]@1�6��n`F@A��rf�B�?I�s���B@Y���Q�?*	+���b@2F
Iterator::Model@3����?!?5�JǛE@)rP�Lۧ?1ź=%�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj��_=�?!��0��:@)]�mO�؞?1��)9��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�z�I|�?!�(���4@)�8�� n�?1�nkwvQ.@:Preprocessing2U
Iterator::Model::ParallelMapV2o�o�>;�?!��u���%@)o�o�>;�?1��u���%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�ó�?!��d�8dL@)�4}v��?1@�<��f@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor섗���?!�h�7_@)섗���?1�h�7_@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceٵ�ݒ�?!�ęd/�@)ٵ�ݒ�?1�ęd/�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�p=
ף�?!��6I~6@)+l� [f?1��E�7�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 58.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�19.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��7$�'�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Q�]@��Q�]@!��Q�]@      ��!       "	�6��n`F@�6��n`F@!�6��n`F@*      ��!       2	��rf�B�?��rf�B�?!��rf�B�?:	�s���B@�s���B@!�s���B@B      ��!       J	���Q�?���Q�?!���Q�?R      ��!       Z	���Q�?���Q�?!���Q�?JGPUY��7$�'�?b �"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop���?Q�?!���?Q�?"&
CudnnRNNCudnnRNN��ڲ!�?!�k��?"J
/sequential_89/embedding_89/embedding_lookup/_13_Send���'t�?!�
N��e�?"`
Egradient_tape/sequential_89/embedding_89/embedding_lookup/Reshape/_29_Sendj��_Q)�?!���A_��?"(
gradients/AddNAddN��xg?!E���l��?"C
$gradients/transpose_9_grad/transpose	Transpose�o��׭c?!�CU�	�?"*
transpose_9	Transpose*4"�Mc?!�wwth�?"A
"gradients/transpose_grad/transpose	Transpose�rT��=T?!�aX�&�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad��m��}S?!�X�FF0�?" 
AddN_1AddNT���I�H?!�5�u6�?Q      Y@Y��ތ�?a� �̹X@q ��ڋ8@yv����|?"�
both�Your program is POTENTIALLY input-bound because 58.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�19.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�24.5463% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 