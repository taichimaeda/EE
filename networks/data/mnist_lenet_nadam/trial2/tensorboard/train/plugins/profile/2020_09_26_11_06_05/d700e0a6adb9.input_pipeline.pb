	/�r�]sI@/�r�]sI@!/�r�]sI@	d���|�?d���|�?!d���|�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6/�r�]sI@L�{)<F@1�!H{@A�^'�ei�?I�,
�(:�?Y�6�Nx	�?*	�A`�Вu@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicem��?!��T�+J@)m��?1��T�+J@:Preprocessing2F
Iterator::Model�F�@�?!3��%^8@)�����H�?1���	��3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateS"�^�?!4�,T_�M@)��[���?1���b@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���X��?!s��v��R@)�� �S��?1ٌo��b@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatm�y�ؘ�?!��(=�@)V��W9�?1x��]�@:Preprocessing2U
Iterator::Model::ParallelMapV2~ R�8��?!�m�pD�@)~ R�8��?1�m�pD�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9DܜJ�?!��Bv�O@)�1��|�?1�Ua!��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor]�E�~u?!����}S�?)]�E�~u?1����}S�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9c���|�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	L�{)<F@L�{)<F@!L�{)<F@      ��!       "	�!H{@�!H{@!�!H{@*      ��!       2	�^'�ei�?�^'�ei�?!�^'�ei�?:	�,
�(:�?�,
�(:�?!�,
�(:�?B      ��!       J	�6�Nx	�?�6�Nx	�?!�6�Nx	�?R      ��!       Z	�6�Nx	�?�6�Nx	�?!�6�Nx	�?JGPUYc���|�?b 