	��S:�F@��S:�F@!��S:�F@	��w[~�?��w[~�?!��w[~�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��S:�F@C���B@1�F��@A��ӀAҗ?Iǃ-v�l�?Y����?*	���S)@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�BX�%��?!Z.+8�U@)u�yƾd�?1\�)GvU@:Preprocessing2F
Iterator::Model��bE�?!��
�F!@)Ҫ�t���?1/�%��q@:Preprocessing2U
Iterator::Model::ParallelMapV2Z�'��&�?!�2Q\4q�?)Z�'��&�?1�2Q\4q�?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipC�y�'�?!"]�#�V@)�z��9y�?1w	�XGa�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��x@�?!�����@)Kr��&Oy?1o
9��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�3g}�1y?!�7�S8��?)�3g}�1y?1�7�S8��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor������q?!P'i�p��?)������q?1P'i�p��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�6o��?!O��þl@)P�mp�b?1�l��0 �?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 82.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9��w[~�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	C���B@C���B@!C���B@      ��!       "	�F��@�F��@!�F��@*      ��!       2	��ӀAҗ?��ӀAҗ?!��ӀAҗ?:	ǃ-v�l�?ǃ-v�l�?!ǃ-v�l�?B      ��!       J	����?����?!����?R      ��!       Z	����?����?!����?JGPUY��w[~�?b 