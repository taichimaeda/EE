	�pz�>@�pz�>@!�pz�>@	�ÿb���?�ÿb���?!�ÿb���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�pz�>@�,g�8@1U1�~R@AkJ�GW�?IX S��?Y~�k�,	�?*	��Q�uc@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?��H��?!>�Т�/I@)�l���?1�T�tA�G@:Preprocessing2F
Iterator::Model�eo)�?!�M��e�A@)y�	�5��?1j�ٶ�9@:Preprocessing2U
Iterator::Model::ParallelMapV2�n��S�?!I���)f"@)�n��S�?1I���)f"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ؖg)�?!����E�@)Y|E�~?1��,E@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipb����?!=YM?P@)JΉ=��u?1�'&�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��k��?!+�<��w&@)��ΤMu?1�L|��
@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��"���s?!ݗ��2�@)��"���s?1ݗ��2�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��-�r?!^�j�R�@)��-�r?1^�j�R�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�ÿb���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�,g�8@�,g�8@!�,g�8@      ��!       "	U1�~R@U1�~R@!U1�~R@*      ��!       2	kJ�GW�?kJ�GW�?!kJ�GW�?:	X S��?X S��?!X S��?B      ��!       J	~�k�,	�?~�k�,	�?!~�k�,	�?R      ��!       Z	~�k�,	�?~�k�,	�?!~�k�,	�?JGPUY�ÿb���?b 