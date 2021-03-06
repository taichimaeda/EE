�	�el��\e@�el��\e@!�el��\e@	���%W�?���%W�?!���%W�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�el��\e@�3�9A�W@1(ђ��JF@A����	�?IձJ�v?@Y-wf��\�?*	�Zd;�`@2F
Iterator::Model������?!vL��H@)�M+�@.�?1B0�f�ZB@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatj��4ӽ�?!^�i�ah6@)H�)s�?1xj!��1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��V`Ț?!��aի�3@)@M-[닔?1�7� �-@:Preprocessing2U
Iterator::Model::ParallelMapV2;S��.�?!�p�x�)@);S��.�?1�p�x�)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipE/�Xni�?!��{bI@)M�7�Q��?1�EAvXo@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceR~R���x?!�w86�.@)R~R���x?1�w86�.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�\p�x?!� �K�	@)�\p�x?1� �K�	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�!�[='�?!C]A�@5@)e�I)��b?1�8�퐥�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 55.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�18.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���%W�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�3�9A�W@�3�9A�W@!�3�9A�W@      ��!       "	(ђ��JF@(ђ��JF@!(ђ��JF@*      ��!       2	����	�?����	�?!����	�?:	ձJ�v?@ձJ�v?@!ձJ�v?@B      ��!       J	-wf��\�?-wf��\�?!-wf��\�?R      ��!       Z	-wf��\�?-wf��\�?!-wf��\�?JGPUY���%W�?b �"N
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop��,vU�?!��,vU�?"&
CudnnRNNCudnnRNNr]|��?!�4�5p��?"J
/sequential_23/embedding_23/embedding_lookup/_19_Send�UC���?!�A���]�?"`
Egradient_tape/sequential_23/embedding_23/embedding_lookup/Reshape/_37_Send1'�z�?!���R��?"(
gradients/AddNAddN�O�|�g?!�mo���?"C
$gradients/transpose_9_grad/transpose	Transpose�s��+�c?!�W�_��?"*
transpose_9	Transposeؙ	@]c?!aY���?"A
"gradients/transpose_grad/transpose	Transpose0�T?!�q��?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad����,S?!B���#�?"K
2gradient_tape/sequential_23/dropout_46/dropout/MulMul�Chm��H?!SA�@�)�?Q      Y@YWx\�@aG=�pX@q����8@y����̂?"�
both�Your program is POTENTIALLY input-bound because 55.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�18.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�24.7488% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 