	����(�d@����(�d@!����(�d@	��-�a��?��-�a��?!��-�a��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����(�d@�_���W@1C:<��UF@Ar��Q���?I�%jj�:@Y1�{�O��?*	Zd;�O�\@2F
Iterator::ModelD�X�oC�?!o��J�H@)U���)�?1	��Ð�B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatBA)Z��?!��5�Fn;@),-#��ʙ?1��@�5@:Preprocessing2U
Iterator::Model::ParallelMapV2���:�f�?!�uc�$@)���:�f�?1�uc�$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���/g�?!�l�j�I@)��9̗�?1�E��$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate!W�Yʋ?!~��#�'@);��Tގ�?1��q]9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoraU��N�y?!xG���@)aU��N�y?1xG���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice͓k
dvv?!�h�]�$@)͓k
dvv?1�h�]�$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapPp��Ӑ?!�(����,@)�%��og?1�mx7��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 57.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�16.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��-�a��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�_���W@�_���W@!�_���W@      ��!       "	C:<��UF@C:<��UF@!C:<��UF@*      ��!       2	r��Q���?r��Q���?!r��Q���?:	�%jj�:@�%jj�:@!�%jj�:@B      ��!       J	1�{�O��?1�{�O��?!1�{�O��?R      ��!       Z	1�{�O��?1�{�O��?!1�{�O��?JGPUY��-�a��?b 