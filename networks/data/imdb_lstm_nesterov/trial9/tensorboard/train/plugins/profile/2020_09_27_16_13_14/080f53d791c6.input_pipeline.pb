	͔�ߒf@͔�ߒf@!͔�ߒf@	��
]4�?��
]4�?!��
]4�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6͔�ߒf@�tZ�A�Y@1LR�bPF@AVE�ɨ2�?I�!q���<@Y&m����?*	��K7��]@2F
Iterator::ModelPVW@�?!�kO*�YG@)s�<G令?1�M�!?#A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate� 3��O�?!�sAM�f7@)(G�`Ɣ?1�����+1@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��?�Ŋ�?!I\m�:�5@)7n1?7�?12O%ٛ�0@:Preprocessing2U
Iterator::Model::ParallelMapV2q>?��?!�wL!��(@)q>?��?1�wL!��(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����%~?!o�9���@)����%~?1o�9���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$�����?!j���Z�J@)�?x�=|?1���i�W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�fF?Ny?!Y4 }|�@)�fF?Ny?1Y4 }|�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�����?!yX��9@)��6�ُd?1U�Y�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 58.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�16.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��
]4�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�tZ�A�Y@�tZ�A�Y@!�tZ�A�Y@      ��!       "	LR�bPF@LR�bPF@!LR�bPF@*      ��!       2	VE�ɨ2�?VE�ɨ2�?!VE�ɨ2�?:	�!q���<@�!q���<@!�!q���<@B      ��!       J	&m����?&m����?!&m����?R      ��!       Z	&m����?&m����?!&m����?JGPUY��
]4�?b 