	9GWFe@9GWFe@!9GWFe@	]�W=���?]�W=���?!]�W=���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails69GWFe@�`7l[�W@1z6�>WiF@A���7/�?Iݴ�!�=@Y�I)����?*	�"��~�\@2F
Iterator::ModelLo.2�?!ŧ� ��I@)���p�?1hb���E@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��*�샜?!Ta59k8@)�я�S�?1�L5&�3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�up�71�?!<X�e$H@)��b���?14xZŚ�"@:Preprocessing2U
Iterator::Model::ParallelMapV2EJ�y�?!r�r�yJ @)EJ�y�?1r�r�yJ @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��e�c]�?!��4=9J(@)�;� �?1fe���T@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceH�Sȕzv?!����?@)H�Sȕzv?1����?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor͓k
dvv?!LR�XK<@)͓k
dvv?1LR�XK<@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapR�o&��?!&@��1-@)�_���f?1��-�E�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 56.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�17.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9\�W=���?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�`7l[�W@�`7l[�W@!�`7l[�W@      ��!       "	z6�>WiF@z6�>WiF@!z6�>WiF@*      ��!       2	���7/�?���7/�?!���7/�?:	ݴ�!�=@ݴ�!�=@!ݴ�!�=@B      ��!       J	�I)����?�I)����?!�I)����?R      ��!       Z	�I)����?�I)����?!�I)����?JGPUY\�W=���?b 