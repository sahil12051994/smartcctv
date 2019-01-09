from __future__ import print_function
import base64
import json
import numpy as np
from StringIO import StringIO
from timeit import default_timer as timer
from PIL import Image
import datetime as dt
from random import randint

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class Spark_Object_Detector():

    def __init__(self,
                 interval=10,
                 model_file='',
                 labels_file='',
                 number_classes=90,
                 detect_treshold=.5,
                 topic_to_consume='input_topic',
                 topic_for_produce='result_topic',
                 kafka_endpoint='127.0.0.1:9092'):
        self.topic_to_consume = topic_to_consume
        self.topic_for_produce = topic_for_produce
        self.kafka_endpoint = kafka_endpoint
        self.treshold = detect_treshold
        self.v_sectors = ['top', 'middle', 'bottom']
        self.h_sectors = ['left', 'center', 'right']

        self.producer = KafkaProducer(bootstrap_servers=kafka_endpoint)

        label_map = label_map_util.load_labelmap(labels_file)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=number_classes,
                                                                    use_display_name=True
                                                                    )
        self.category_index = label_map_util.create_category_index(categories)

        sc = SparkContext(appName='PyctureStream')
        self.ssc = StreamingContext(sc, interval)

        log4jLogger = sc._jvm.org.apache.log4j
        log_level = log4jLogger.Level.ERROR
        log4jLogger.LogManager.getLogger('org').setLevel(log_level)
        log4jLogger.LogManager.getLogger('kafka').setLevel(log_level)
        self.logger = log4jLogger.LogManager.getLogger(__name__)

        with tf.gfile.FastGFile(model_file, 'rb') as f:
            model_data = f.read()
        self.model_data_bc = sc.broadcast(model_data)
        self.graph_def = tf.GraphDef()
        self.graph_def.ParseFromString(self.model_data_bc.value)

    def start_processing(self):
        kvs = KafkaUtils.createDirectStream(self.ssc,
                                            [self.topic_to_consume],
                                            {'metadata.broker.list': self.kafka_endpoint}
                                            )
        kvs.foreachRDD(self.handler)
        self.ssc.start()
        self.ssc.awaitTermination()

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def box_to_sector(self, box):
        width = box[3] - box[1]
        h_pos = box[1] + width / 2.0
        height = box[2] - box[0]
        v_pos = box[0] + height / 2.0
        h_sector = min(int(h_pos * 3), 2)
        v_sector = min(int(v_pos * 3), 2)
        return (self.v_sectors[v_sector], self.h_sectors[h_sector])

    def get_annotated_image_as_text(self, image_np, output):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output['detection_boxes'],
            output['detection_classes'],
            output['detection_scores'],
            self.category_index,
            instance_masks=output.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=3)

        img = Image.fromarray(image_np)
        text_stream = StringIO()
        img.save(text_stream, 'JPEG')
        contents = text_stream.getvalue()
        text_stream.close()
        img_as_text = base64.b64encode(contents).decode('utf-8')
        return img_as_text

    def format_object_desc(self, output):
        objs = []
        for i in range(len(output['detection_classes'])):
            score = round(output['detection_scores'][i], 2)
            if score > self.treshold:
                cat_id = output['detection_classes'][i]
                label = self.category_index[cat_id]['name']

                box = output['detection_boxes'][i]

                objs.append({
                    'label': label,
                    'score': score,
                    'sector': self.box_to_sector(box)
                })
        return objs

    def detect_objects(self, event):
        decoded = base64.b64decode(event['image'])
        stream = StringIO(decoded)
        image = Image.open(stream)
        image_np = self.load_image_into_numpy_array(image)
        stream.close()

        tf.import_graph_def(self.graph_def, name='')  # ~ 2.7 sec

        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs
            }
            tensor_dict = {}
            for key in ['num_detections',
                        'detection_boxes',
                        'detection_scores',
                        'detection_classes',
                        'detection_masks'
                        ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = (tf.get_default_graph()
                                          .get_tensor_by_name(tensor_name)
                                        )
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0]
                )
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0]
                )
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32
                )
                detection_boxes = tf.slice(
                    detection_boxes,
                    [0, 0],
                    [real_num_detection, -1]
                )
                detection_masks = tf.slice(
                    detection_masks,
                    [0, 0, 0],
                    [real_num_detection, -1, -1]
                )
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks,
                    detection_boxes,
                    image.shape[0],
                    image.shape[1]
                )
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5),
                    tf.uint8
                )
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed,
                    0
                )

            image_tensor = (tf.get_default_graph()
                            .get_tensor_by_name('image_tensor:0')
                            )

            output = sess.run(
                tensor_dict,
                feed_dict={image_tensor: np.expand_dims(image, 0)}
            )

            output['num_detections'] = int(output['num_detections'][0])
            output['detection_classes'] = output['detection_classes'][0].astype(
                np.uint8)
            output['detection_boxes'] = output['detection_boxes'][0]
            output['detection_scores'] = output['detection_scores'][0]

        tf.reset_default_graph()

        result = {'timestamp': event['timestamp'],
                  'camera_id': event['camera_id'],
                  'objects': self.format_object_desc(output),
                  'image': self.get_annotated_image_as_text(image_np, output)
                  }
        return json.dumps(result)

    def handler(self, timestamp, message):
        records = message.collect()
        to_process = {}
        self.logger.info( '\033[3' + str(randint(1, 7)) + ';1m' +  # Color
            '-' * 25 +
            '[ NEW MESSAGES: ' + str(len(records)) + ' ]'
            + '-' * 25 +
            '\033[0m'
            )
        dt_now = dt.datetime.now()
        for record in records:
            event = json.loads(record[1])
            self.logger.info('Received Message: ' +
                             event['camera_id'] + ' - ' + event['timestamp'])
            dt_event = dt.datetime.strptime(
                event['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
            delta = dt_now - dt_event
            if delta.seconds > 5:
                continue
            to_process[event['camera_id']] = event

        if len(to_process) == 0:
            self.logger.info('Skipping processing...')

        for key, event in to_process.items():
            self.logger.info('Processing Message: ' +
                             event['camera_id'] + ' - ' + event['timestamp'])
            start = timer()
            detection_result = self.detect_objects(event)
            end = timer()
            delta = end - start
            self.logger.info('Done after ' + str(delta) + ' seconds.')
            self.producer.send(self.topic_for_produce, detection_result)
            self.producer.flush()


if __name__ == '__main__':
    sod = Spark_Object_Detector(
        interval=3,
        model_file='/home/cloudera/smartcctv-master/frozen_inference_graph.pb',
        labels_file='/home/cloudera/smartcctv-master/mscoco_label_map.pbtxt',
        number_classes=90,
        detect_treshold=.5,
        topic_to_consume='input_topic',
        topic_for_produce='result_topic',
        kafka_endpoint='127.0.0.1:9092')
    sod.start_processing()
