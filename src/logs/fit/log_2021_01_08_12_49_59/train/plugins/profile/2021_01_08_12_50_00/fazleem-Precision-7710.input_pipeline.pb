	��^C@��^C@!��^C@	�9��e�?�9��e�?!�9��e�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��^C@Kt�Y�b�?A�ǁW�B@Y@�t�_��?*	�~j�tR@2F
Iterator::Model?����?!I1�MNE@)@ٔ+�˕?1�_�9.p=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�Ѫ�t��?!�'�L�q:@)���͋�?1�k��4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��Ր�ǒ?!K�t�x]9@)X9��v�?1`X�㦒4@:Preprocessing2U
Iterator::Model::ParallelMapV2����ׁ�?!Ed���X*@)����ׁ�?1Ed���X*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�����>�?!���)��L@)6!�1�p?1��ĴO@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorYO���*p?!���@)YO���*p?1���@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��ǵ�bl?!��q�G+@)��ǵ�bl?1��q�G+@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�9��e�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Kt�Y�b�?Kt�Y�b�?!Kt�Y�b�?      ��!       "      ��!       *      ��!       2	�ǁW�B@�ǁW�B@!�ǁW�B@:      ��!       B      ��!       J	@�t�_��?@�t�_��?!@�t�_��?R      ��!       Z	@�t�_��?@�t�_��?!@�t�_��?JCPU_ONLYY�9��e�?b 