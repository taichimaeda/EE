	uv28��c@uv28��c@!uv28��c@	Q8-��8�?Q8-��8�?!Q8-��8�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6uv28��c@������T@1�9��	G@Ap��s���?IGsd�>@Y-�����?*	V-Fa@2F
Iterator::Model(a��_Y�?!�/�W"�H@)7�xͫ:�?1&��=C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��dV�p�?!�s��Gd3@)��~m��?1�\3o�-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�C�|�?!b��l�e2@)�tF^֔?1����:s-@:Preprocessing2U
Iterator::Model::ParallelMapV2f�O7P��?!�&���%@)f�O7P��?1�&���%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����=�?!.�&��zI@)^���?1Q���)#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice �o_�y?!NG�@<@) �o_�y?1NG�@<@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�]��y�t?!����_@)�]��y�t?1����_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps�4�B�?!Q��K�6@)YL�Qt?1RS�!�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 52.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�18.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Q8-��8�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	������T@������T@!������T@      ��!       "	�9��	G@�9��	G@!�9��	G@*      ��!       2	p��s���?p��s���?!p��s���?:	Gsd�>@Gsd�>@!Gsd�>@B      ��!       J	-�����?-�����?!-�����?R      ��!       Z	-�����?-�����?!-�����?JGPUYQ8-��8�?b 