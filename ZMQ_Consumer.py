import time
import zmq

class ZMQ_Consumer:


    def __init__(self, zmq_connectstring):
        self.context = zmq.Context()
        self.consumer_receiver = self.context.socket(zmq.PULL)
        print(f"connecting to zmq... [{zmq_connectstring}]", end="")
        self.consumer_receiver.connect(zmq_connectstring)
        time.sleep(0.5)
        print("done")

    def consume_multipart(self):
        while True:
            try:
                old_work = None
                while True:
                    new_work = self.consumer_receiver.recv_multipart(copy=False, flags=zmq.NOBLOCK)
                    old_work = new_work
            except zmq.ZMQError:
                if old_work is not None:
                    return str(old_work[0])
                return None

    def consume_json(self):
        while True:
            try:
                old_work = None
                while True:
                    new_work = self.consumer_receiver.recv_json(flags=zmq.NOBLOCK)
                    old_work = new_work
            except zmq.ZMQError:
                if old_work is not None:
                    return old_work
                return None

