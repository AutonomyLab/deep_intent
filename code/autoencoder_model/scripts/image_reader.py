from Queue import Queue
from threading import Thread

class LoadImages(Thread):
   def __init__(self, queue):
       Thread.__init__(self)
       self.queue = queue

   def run(self):
       while True:
           # Get the work from the queue and expand the tuple
           directory, link = self.queue.get()
           download_link(directory, link)
           self.queue.task_done()

def main():
   ts = time()
   client_id = os.getenv('IMGUR_CLIENT_ID')
   if not client_id:
       raise Exception("Couldn't find IMGUR_CLIENT_ID environment variable!")
   download_dir = setup_download_dir()
   links = [l for l in get_links(client_id) if l.endswith('.jpg')]
   # Create a queue to communicate with the worker threads
   queue = Queue()
   # Create 8 worker threads
   for x in range(4):
       worker = LoadImages(queue)
       # Setting daemon to True will let the main thread exit even though the workers are blocking
       worker.daemon = True
       worker.start()
   # Put the tasks into the queue as a tuple
   for link in links:
       logger.info('Queueing {}'.format(link))
       queue.put((download_dir, link))
   # Causes the main thread to wait for the queue to finish processing all the tasks
   queue.join()
   print('Took {}'.format(time() - ts))