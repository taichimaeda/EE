	e����S=@e����S=@!e����S=@	D	�V-6�?D	�V-6�?!D	�V-6�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6e����S=@-�cyW�8@1�Ss��p@Am��?Ir���b�?Yi�A'��?*	�$���g@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ݮ���?!Zb�w�L@)����ɍ�?1�%��&CK@:Preprocessing2F
Iterator::Model}\*���?!�bnٸ<@)5`��i�?1i?F�j�5@:Preprocessing2U
Iterator::Model::ParallelMapV2 �+�p��?!�o��M@) �+�p��?1�o��M@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate4���5�?!;KU���@)�	g��ɀ?1D�u?<@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipi�7>[�?!�~g���Q@)����?1ۛC��{@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_ѭw?!��O@)��_ѭw?1��O@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicet	4�t?!컀��f@)t	4�t?1컀��f@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�(z�c��?!��~	0#@)���	.Vt?1��g��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 83.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9D	�V-6�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	-�cyW�8@-�cyW�8@!-�cyW�8@      ��!       "	�Ss��p@�Ss��p@!�Ss��p@*      ��!       2	m��?m��?!m��?:	r���b�?r���b�?!r���b�?B      ��!       J	i�A'��?i�A'��?!i�A'��?R      ��!       Z	i�A'��?i�A'��?!i�A'��?JGPUYD	�V-6�?b 