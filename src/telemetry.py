import threading
import queue
import logging
import json
import pika
import time

class TelemetrySender:
    """
    Asynchronous telemetry dispatcher.
    Offloads AMQP network requests to a background thread to prevent blocking the main inference loop.
    """
    def __init__(self, amqp_url="amqp://guest:guest@localhost:5672/"):
        self.amqp_url = amqp_url
        self.q = queue.Queue(maxsize=50) 
        
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        
    def _worker(self):
        while True:
            try:
                connection = pika.BlockingConnection(pika.URLParameters(self.amqp_url))
                channel = connection.channel()
                channel.queue_declare(queue='telemetry', durable=True)
                
                logging.info(f"Connected to RabbitMQ at {self.amqp_url}")
                
                while True:
                    data = self.q.get()
                    channel.basic_publish(
                        exchange='',
                        routing_key='telemetry',
                        body=json.dumps(data),
                        properties=pika.BasicProperties(
                            delivery_mode=pika.DeliveryMode.Persistent
                        )
                    )
            except Exception as e:
                logging.warning(f"Telemetry dispatch failed, attempting reconnect: {e}")
                time.sleep(3) # Wait before attempting to reconnect
                
    def send(self, data):
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(data)

