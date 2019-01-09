from kafka import KafkaProducer
import cv2
import base64
import json
import time
import datetime as dt
import logging
logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Webcam_Producer():

    def __init__(self,
                 interval: int = 5,
                 source=0,
                 camera_id: str = 'camera_generic',
                 topic: str = 'input_topic',
                 server: str = '127.0.0.1:9092'):

        logger.info('-'*50)
        logger.info(f'Initialized camera "{camera_id}" with source {source}.')
        logger.info(f'Send to "{topic}" on "{server}" every {interval} sec.')

        # Class Variables
        self.interval = interval  # Interval for Photos in Seconds
        self.video_source = source
        self.camera_id = camera_id
        self.server = server  # Host + Port of Kafka Endpoint
        self.topic = topic
        self.img_file = './input.jpg'

        # Connection to Kafka Enpoint
        try:
            self.producer = KafkaProducer(bootstrap_servers=self.server,
                                          value_serializer=lambda m: json.dumps(m).encode('utf8'))
        except Exception as e:
            logger.error(e)

        # Start Streaming
        logger.info('-'*50)
        self.stream_video()

    def stream_video(self):

        logger.info(f'Start capturing frames every {self.interval} sec.')
        while True:
            vidcap = cv2.VideoCapture(self.video_source)
            vidcap.set(3,1280)
            vidcap.set(4,720)
            success, image = vidcap.read()
            timestamp = dt.datetime.now().isoformat()
            vidcap.release()
            if success is True:
                jpg = cv2.imencode('.jpg', image)[1]
                jpg_as_text = base64.b64encode(jpg).decode('utf-8')
                result = {
                    'image': jpg_as_text,
                    'timestamp': timestamp,
                    'camera_id': self.camera_id
                }
                self.send_to_kafka(result)
                self.save_image(image)
            else:
                logger.error(f'Could not read image from source {self.video_source}!')
            time.sleep(self.interval)

    def send_to_kafka(self, data):

        self.producer.send(self.topic, data)
        logger.info('Sent image to Kafka endpoint.')

    def save_image(self, image):

        cv2.imwrite(self.img_file, image)
        logger.info(f'Saved image file to {self.img_file}.')

if __name__ == '__main__':

    Webcam_Producer(interval=3,
                    source=0,
                    camera_id='sahil_cam',
                    server='127.0.0.1:9092',
                    topic='input_topic')
