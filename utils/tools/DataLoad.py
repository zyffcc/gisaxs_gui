import tensorflow as tf

class tfrecord_read:
    def __init__(self, filename, input_data_shape, output_data_shape):
        self.filename = filename
        self.input_data_shape = input_data_shape
        self.output_data_shape = output_data_shape

    def _parse_function(self, example_proto):
        feature_description = {
            'input': tf.io.FixedLenFeature(self.input_data_shape, tf.float32),
            'output': tf.io.FixedLenFeature(self.output_data_shape, tf.float32)
        }
        return tf.io.parse_single_example(example_proto, feature_description)
    
    def load_dataset(self, batch_size=32):
        raw_dataset = tf.data.TFRecordDataset(self.filename, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        parsed_dataset = raw_dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parsed_dataset = parsed_dataset.cache()
        parsed_dataset = parsed_dataset.batch(batch_size)
        parsed_dataset = parsed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return parsed_dataset
    
    def load_dataset_no_batch(self):
        raw_dataset = tf.data.TFRecordDataset(self.filename, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        parsed_dataset = raw_dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parsed_dataset = parsed_dataset.cache()
        parsed_dataset = parsed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return parsed_dataset
    
    def split_dataset(self, parsed_dataset, val_per = 0.2):
        dataset_size = sum(1 for _ in parsed_dataset)
        val_size = int(val_per * dataset_size)
        train_size = dataset_size - val_size

        # Splitting the dataset
        train_dataset = parsed_dataset.take(train_size)
        val_dataset = parsed_dataset.skip(train_size).take(val_size)
        return train_dataset, val_dataset
    
    def prepare_for_training(self, dataset, batch_size=128, shuffle_buffer_size=1000):
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def split_input_output(self, data):
        return data['input'], data['output']
    
    def load(self):
        parsed_dataset = self.load_dataset_no_batch()
        train_dataset, val_dataset = self.split_dataset(parsed_dataset)
        train_dataset = self.prepare_for_training(train_dataset)
        val_dataset = self.prepare_for_training(val_dataset)
        train_dataset = train_dataset.map(self.split_input_output)
        val_dataset = val_dataset.map(self.split_input_output)
        return train_dataset, val_dataset
    
if __name__ == '__main__':
    filename = f'/gpfs/dust/maxwell/user/zhaiyufe/TrainSet/Elipsoid/data_all.tfrecord'
    input_data_shape = (128, 128, 1)
    output_data_shape = (20, 20)
    

    tfrecord = tfrecord_read(filename, input_data_shape, output_data_shape)
    # parsed_dataset = tfrecord.load_dataset_no_batch()
    # train_dataset, val_dataset = tfrecord.split_dataset(parsed_dataset)
    # train_dataset = tfrecord.prepare_for_training(train_dataset)
    # val_dataset = tfrecord.prepare_for_training(val_dataset)
    # train_dataset = train_dataset.map(tfrecord.split_input_output)
    # val_dataset = val_dataset.map(tfrecord.split_input_output)
    
    # for data in train_dataset.take(1):
    #     print(data['input'].shape)
    #     print(data['output'].shape)
    
    # for data in val_dataset.take(1):
    #     print(data['input'].shape)
        # print(data['output'].shape)

    train_dataset, val_dataset = tfrecord.load()

    for data in train_dataset.take(1):
        print(data[0].shape)
        print(data[1].shape)
    
    for data in val_dataset.take(1):
        print(data[0].shape)
        print(data[1].shape)
         