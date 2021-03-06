�	��"�zd@��"�zd@!��"�zd@	��rUD�?��rUD�?!��rUD�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��"�zd@�?���iV@1jkD0FF@A�F���?I�h:;8=@Y��˚X��?*	;�O��.c@2F
Iterator::Model�kC�8�?!��JƝA@),)w�㣥?1e�N��;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��,'���?!�ƻ��;@)~R���?1�el�e6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateeM.���?!��mt:�7@)�FY����?1*�f��x3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�H�H��?!3��1@P@)~5��?1��R�x�&@:Preprocessing2U
Iterator::Model::ParallelMapV2�	1�Tm�?!:�'�@)�	1�Tm�?1:�'�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��L�n�?!�Z��/@)��L�n�?1�Z��/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�uʣ{?!�\�@)�uʣ{?1�\�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�aL�{)�?!HNS�9@)6w��\�f?1�Us����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�17.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��rUD�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�?���iV@�?���iV@!�?���iV@      ��!       "	jkD0FF@jkD0FF@!jkD0FF@*      ��!       2	�F���?�F���?!�F���?:	�h:;8=@�h:;8=@!�h:;8=@B      ��!       J	��˚X��?��˚X��?!��˚X��?R      ��!       Z	��˚X��?��˚X��?!��˚X��?JGPUY��rUD�?b �"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�S|6�R�?!�S|6�R�?"&
CudnnRNNCudnnRNN��w��!�?!�L����?"J
/sequential_42/embedding_42/embedding_lookup/_15_Send�c���?!]���b�?"`
Egradient_tape/sequential_42/embedding_42/embedding_lookup/Reshape/_33_Send���'�Ԍ?!�yf�e��?"(
gradients/AddNAddN�#�p)g?!_u����?"C
$gradients/transpose_9_grad/transpose	Transpose4��/Mec?!	:�K���?"*
transpose_9	Transpose4��/Mec?!�՘Y�?"A
"gradients/transpose_grad/transpose	Transpose��ʿ�T?!q����?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad�?h!��S?!'%˹'�?" 
AddN_1AddN>�܌L�H?!c\-�?Q      Y@Yxi��$��?aZN�l3�X@qq��`�C@y3[�Ԃ?"�
both�Your program is POTENTIALLY input-bound because 54.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�17.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�39.9092% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 