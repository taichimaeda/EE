	*��g\C@*��g\C@!*��g\C@	u��d�$�?u��d�$�?!u��d�$�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6*��g\C@I�[�\?@1`X�|�@A�W���T�?I�߽���?YiV�y˥?*	��~j�у@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate5F�j�?!R�I{D/V@)܁:���?1�9iW�U@:Preprocessing2F
Iterator::Model�9��ȥ?!��N2��@)��-�v��?1WP佘|@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�����?!�v����@)����?1���ϴ�@:Preprocessing2U
Iterator::Model::ParallelMapV2Ӿ��z܇?!&�ѡd�?)Ӿ��z܇?1&�ѡd�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$ӡ����?!���RW@)DkE��|?1�gT��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices�4�Bx?!�-N���?)s�4�Bx?1�-N���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���Q�n?!#t�����?)���Q�n?1#t�����?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��%�?!���kjGV@)2��|�c?1�J���%�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 81.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9u��d�$�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	I�[�\?@I�[�\?@!I�[�\?@      ��!       "	`X�|�@`X�|�@!`X�|�@*      ��!       2	�W���T�?�W���T�?!�W���T�?:	�߽���?�߽���?!�߽���?B      ��!       J	iV�y˥?iV�y˥?!iV�y˥?R      ��!       Z	iV�y˥?iV�y˥?!iV�y˥?JGPUYu��d�$�?b 