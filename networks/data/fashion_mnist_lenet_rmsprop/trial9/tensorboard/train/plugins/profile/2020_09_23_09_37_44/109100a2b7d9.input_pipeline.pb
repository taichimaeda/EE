	�{�?m�G@�{�?m�G@!�{�?m�G@	�=I"'��?�=I"'��?!�=I"'��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�{�?m�G@��
~�C@1�U�p�@A�� �=�?I'�E'K-�?Yd���?*	�G�z�S@2F
Iterator::Model��z6��?!��$|��K@)s�69|ҡ?1�x�q@�E@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat`��i��?!+NtIZ 6@)'�����?1�U%�)2@:Preprocessing2U
Iterator::Model::ParallelMapV2�8d�b�?!�
) �'@)�8d�b�?1�
) �'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate#��<�?!��%�j,@)}w+Kt�y?1L�,��K@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�nf���t?!𨖊�@)�nf���t?1𨖊�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�o%;6�?! ۃ?FF@)Ly �Hs?1�r�A�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorf.py�i?!��#�@)f.py�i?1��#�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_A��h:�?!O�ƭ��0@)�q����_?1�ְ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 82.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�=I"'��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��
~�C@��
~�C@!��
~�C@      ��!       "	�U�p�@�U�p�@!�U�p�@*      ��!       2	�� �=�?�� �=�?!�� �=�?:	'�E'K-�?'�E'K-�?!'�E'K-�?B      ��!       J	d���?d���?!d���?R      ��!       Z	d���?d���?!d���?JGPUY�=I"'��?b 