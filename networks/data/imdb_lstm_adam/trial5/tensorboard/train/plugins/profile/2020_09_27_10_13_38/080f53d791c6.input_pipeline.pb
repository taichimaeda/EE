	k���:f@k���:f@!k���:f@	~�5%Di�?~�5%Di�?!~�5%Di�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6k���:f@�׼��6Y@10��9\CF@A�����L�?I�����>@Y((E+��?*	+��b@2F
Iterator::Model��;�(A�?!>���r�H@)�E��(&�?1�{��2D@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��p�Ws�?!��D�}85@)|a2U0�?1�2x.�0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea�9��?![�Eh)�2@)>+N��?1���a~,@:Preprocessing2U
Iterator::Model::ParallelMapV2q=
ףp�?!��Xp�"@)q=
ףp�?1��Xp�"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%����?!�EX�)I@)�����?1�4�5@� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�@��_�{?!2Y_���@)�@��_�{?12Y_���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�%jj�z?!7�H�<Q@)�%jj�z?17�H�<Q@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�kA�!�?!Cl`��4@)�
F%uj?1>�2��� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 57.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�17.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9}�5%Di�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�׼��6Y@�׼��6Y@!�׼��6Y@      ��!       "	0��9\CF@0��9\CF@!0��9\CF@*      ��!       2	�����L�?�����L�?!�����L�?:	�����>@�����>@!�����>@B      ��!       J	((E+��?((E+��?!((E+��?R      ��!       Z	((E+��?((E+��?!((E+��?JGPUY}�5%Di�?b 