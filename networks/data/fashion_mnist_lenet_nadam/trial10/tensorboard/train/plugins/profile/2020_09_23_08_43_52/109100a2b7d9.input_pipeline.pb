	ݶ�Q�M@ݶ�Q�M@!ݶ�Q�M@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ݶ�Q�M@!���kI@1܁:�@Ai6��`��?IfJ�o	@�?*	��K7�n�@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatD6�.6-�?!\�a�&�U@)c�#�w��?1 �!L�U@:Preprocessing2F
Iterator::Model�٬�\m�?!W�C�Ԕ!@)"nN%@�?1'���Md@:Preprocessing2U
Iterator::Model::ParallelMapV2�׼��Z�?!+O��@)�׼��Z�?1+O��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipAfg�;�?!u�7fe�V@)�c> Й�?1�E'B���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_�;��?!��1�_@)�Ϛi�?1X3�VJ��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�^��x�z?!Z�#���?)�^��x�z?1Z�#���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorpA�,_w?!��&U��?)pA�,_w?1��&U��?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���J̳�?!5 �9�X@) p��s�j?1��4>���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 84.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	!���kI@!���kI@!!���kI@      ��!       "	܁:�@܁:�@!܁:�@*      ��!       2	i6��`��?i6��`��?!i6��`��?:	fJ�o	@�?fJ�o	@�?!fJ�o	@�?B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb 