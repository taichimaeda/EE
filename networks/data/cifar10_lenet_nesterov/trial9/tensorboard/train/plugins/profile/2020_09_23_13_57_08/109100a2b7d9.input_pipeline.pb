	F[�D�qB@F[�D�qB@!F[�D�qB@	��c�!�?��c�!�?!��c�!�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6F[�D�qB@2U0*�<@1�9�m��@A�䠄��?Iд��hd�?Yd!:��?*	��Q�fV@2F
Iterator::Model�A�fէ?!�h[��I@)��f/�?1�h���D@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��rf��?!��Gj4@)���W:�?1��j0@:Preprocessing2U
Iterator::Model::ParallelMapV2^�����?!�3��C$@)^�����?1�3��C$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�M�»�?!,��P/@)��|y��?1��^ &�#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���N�?!@���^H@)Tƿϸ�?1�p25t9"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceϠ���u?!:Kn�{@)Ϡ���u?1:Kn�{@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoruv28J^m?!M�H� @)uv28J^m?1M�H� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!���B��2@) ��ce?1o���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 76.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��c�!�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	2U0*�<@2U0*�<@!2U0*�<@      ��!       "	�9�m��@�9�m��@!�9�m��@*      ��!       2	�䠄��?�䠄��?!�䠄��?:	д��hd�?д��hd�?!д��hd�?B      ��!       J	d!:��?d!:��?!d!:��?R      ��!       Z	d!:��?d!:��?!d!:��?JGPUY��c�!�?b 