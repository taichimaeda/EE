	{�f��Cd@{�f��Cd@!{�f��Cd@	 �k����? �k����?! �k����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6{�f��Cd@�{dsնU@1c��K�IF@A,����?Iݚt["?>@Y�����?*	^�I+c@2F
Iterator::Model*U��-�?!m ���fB@)����·�?1È@?t�=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateEg�E(��?!~O*+�7@)ϡU1��?1F�y3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�mnLOX�?!���Ԣ]7@)2=a���?1u[�#4�2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip@��r�θ?!��f	&�O@) ��h�'�?1��/�e(@:Preprocessing2U
Iterator::Model::ParallelMapV2��Z}u�?!_�ŷ�T@)��Z}u�?1_�ŷ�T@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorx�=\r|?!1T�ú@)x�=\r|?11T�ú@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice겘�|\{?!��%��l@)겘�|\{?1��%��l@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�?�dȱ�?!��v&ʡ;@)��v� �w?1�:��d@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 53.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�18.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9 �k����?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�{dsնU@�{dsնU@!�{dsնU@      ��!       "	c��K�IF@c��K�IF@!c��K�IF@*      ��!       2	,����?,����?!,����?:	ݚt["?>@ݚt["?>@!ݚt["?>@B      ��!       J	�����?�����?!�����?R      ��!       Z	�����?�����?!�����?JGPUY �k����?b 