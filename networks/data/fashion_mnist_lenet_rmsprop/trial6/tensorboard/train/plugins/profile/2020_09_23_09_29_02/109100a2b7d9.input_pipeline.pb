	��7�TG@��7�TG@!��7�TG@	�
e�L2�?�
e�L2�?!�
e�L2�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��7�TG@�)(C@1�ȑ���@A��+I��?I�d8�?Y��t?�?*	p=
ף�V@2F
Iterator::Modelr��&OY�?!h���m�F@)��|	�?1 5�l�=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg|_\�Җ?!��8��R8@)r�t��ϑ?1��.���2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(���ǒ?!�`dB�4@)ˢ����?1�<= d 0@:Preprocessing2U
Iterator::Model::ParallelMapV2����);�?!es}��&/@)����);�?1es}��&/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{��9y��?!�O
�?K@).s�,&6?1gʊP�� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�S���
t?!�'��[@)�S���
t?1�'��[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor^��6S!n?!����@)^��6S!n?1����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��l;m��?!`K��o*:@)��-�[?1�l���}�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 82.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�
e�L2�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�)(C@�)(C@!�)(C@      ��!       "	�ȑ���@�ȑ���@!�ȑ���@*      ��!       2	��+I��?��+I��?!��+I��?:	�d8�?�d8�?!�d8�?B      ��!       J	��t?�?��t?�?!��t?�?R      ��!       Z	��t?�?��t?�?!��t?�?JGPUY�
e�L2�?b 