	�C�r�GB@�C�r�GB@!�C�r�GB@	�Tm`�z�?�Tm`�z�?!�Tm`�z�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�C�r�GB@����}=@17o��@A�?�J���?I�
Ĳ��?Y`�5�!�?*	���x�c@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice&�\R�ݬ?!�R�̔qB@)&�\R�ݬ?1�R�̔qB@:Preprocessing2F
Iterator::Model|�Y�H��?!B��Z�A@)}�r�蜟?1"����24@:Preprocessing2U
Iterator::Model::ParallelMapV2z5@i�Q�?!��9�/@)z5@i�Q�?1��9�/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�U�6��?!j���#H@)>ϟ6�ӑ?1Zn�?��&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����3��?!��o��%@)o��ܚt�?1��ۊ!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�(�'�$�?!�s���P@)}w+Ktv?1������@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���1�i?!��}�� @)���1�i?1��}�� @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�����?!��Dv�H@)�+�j�c?1�C�.���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 80.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�Tm`�z�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����}=@����}=@!����}=@      ��!       "	7o��@7o��@!7o��@*      ��!       2	�?�J���?�?�J���?!�?�J���?:	�
Ĳ��?�
Ĳ��?!�
Ĳ��?B      ��!       J	`�5�!�?`�5�!�?!`�5�!�?R      ��!       Z	`�5�!�?`�5�!�?!`�5�!�?JGPUY�Tm`�z�?b 