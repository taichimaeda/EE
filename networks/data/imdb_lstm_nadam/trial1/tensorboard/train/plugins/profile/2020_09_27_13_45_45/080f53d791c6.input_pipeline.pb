	，ｨR3）@，ｨR3）@!，ｨR3）@	�Q｢ﾁﾜ�ﾃ?�Q｢ﾁﾜ�ﾃ?!�Q｢ﾁﾜ�ﾃ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6，ｨR3）@s+�ﾕXa@1ﾗQﾕQ妲@AｬUｻ&､ｹ?Iｯ|也ﾁ�=@Y汚I�2ﾕ?*	βﾊ｡]@2F
Iterator::Model｢ﾓ�n,(ｰ?!\W&ｸ�K@)ｿ�s�ｨ?1l�~ﾙ飛D@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat:ｭﾛ��[�?!Vn兀A�6@)FzQｻ_�?1ﾆ�ﾄh�1@:Preprocessing2U
Iterator::Model::ParallelMapV22身熔�?!ﾁ享zΨ(@)2身熔�?1ﾁ享zΨ(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate.ﾉ&�?!��<T+@)怺T[�?1ｼ$香F@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipUﾁｨ､N@ｫ?!､ｨﾙG
澁@)�o爺~?1k6K.1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceーw�T�|?!?ﾕ揮侯@)ーw�T�|?1?ﾕ揮侯@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor骨�=��w?!@ﾆQﾃｻ@)骨�=��w?1@ﾆQﾃｻ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����｣�?!1HL｣�|0@)ﾈ鷏ﾏI�k?1柚2*�r@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
ﾅData preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
ﾒReading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
ﾅReading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
ｺOther data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)ﾙ
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
bothｫYour program is POTENTIALLY input-bound because 64.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"ﾌ13.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Q｢ﾁﾜ�ﾃ?>Look at Section 3 for the breakdown of input time on the host.Bﾃ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	s+�ﾕXa@s+�ﾕXa@!s+�ﾕXa@      ��!       "	ﾗQﾕQ妲@ﾗQﾕQ妲@!ﾗQﾕQ妲@*      ��!       2	ｬUｻ&､ｹ?ｬUｻ&､ｹ?!ｬUｻ&､ｹ?:	ｯ|也ﾁ�=@ｯ|也ﾁ�=@!ｯ|也ﾁ�=@B      ��!       J	汚I�2ﾕ?汚I�2ﾕ?!汚I�2ﾕ?R      ��!       Z	汚I�2ﾕ?汚I�2ﾕ?!汚I�2ﾕ?JGPUY�Q｢ﾁﾜ�ﾃ?b 