import datetime
import time

class Timer:
    last = datetime.datetime.now()
    def measure(self, str):
        now = datetime.datetime.now()
        diff = now - self.last
        print('{: <40} finished, {:5f} ms since last measurement'.
              format(str, diff.total_seconds() * 1000))
        self.last = now

foo = Timer()
time.sleep(2)
foo.measure("sleep")
