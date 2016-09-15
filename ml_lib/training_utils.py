import StringIO
import json
import thread
import threading
import time
import traceback
from bunch import Bunch
import ml_utils

__author__ = 'maciek'
import web

urls = (
  '/run_command', 'run_command',
  '/stop', 'stop'
)

web.config.debug = False


class CommandReceiver(object):
    class NoResult(object):
        pass

    def __init__(self):
        self.to_handle = []
        self.do_stop = 0
        self.mutex = threading.Lock()
        self.command_results = {}

    def add_command(self, command):
        self.mutex.acquire()
        id = ml_utils.id_generator(20)
        self.to_handle.append(Bunch(command=command, id=id))
        self.mutex.release()
        return id

    def get_commands(self):
        self.mutex.acquire()
        commands = self.to_handle
        self.to_handle = []
        self.mutex.release()
        return commands

    def set_command_result(self, id, result):
        self.mutex.acquire()
        self.command_results[id] = result
        self.mutex.release()

    def get_command_result(self, id):
        self.mutex.acquire()
        res = self.command_results.get(id, CommandReceiver.NoResult())
        self.mutex.release()
        return res

    def run(self, port):
        try:
            app = web.application(urls, globals())
            web.httpserver.runbasic(app.wsgifunc(), ("0.0.0.0", port))
        except AttributeError as e:
            print e
            print 'Server Failed!!!!!!!!!!!!!!!!!!!!!!!!!!!'

    def handle_commands(self, ct):
        # This is handled on the main training thread

        #print 'handle_commands', 'do_stop', command_receiver.do_stop

        if self.do_stop:
            raise KeyboardInterrupt()

        commands = self.get_commands()
        for b in commands:
            command = b.command
            id = b.id
            print 'exec', command
            try:
                res = StringIO.StringIO()
                exec(command)
                v = res.getvalue()
                print 'Setting result', id, v
                self.set_command_result(id, v)
            except Exception as e:
                self.set_command_result(id, 'Traceback: ' + traceback.format_exc())

command_receiver = CommandReceiver()

class run_command:
   def POST(self):
       command = web.input()['command']
       id = command_receiver.add_command(command)
       while True:
           print 'Waiting for result'
           res = command_receiver.get_command_result(id)
           if not isinstance(res, CommandReceiver.NoResult):
               print 'Received result', res
               return res

           time.sleep(1)


class stop:
    def POST(self):
        command_receiver.do_stop = 1
        return 'OK'


def run_command_receiver(port):
    thread.start_new_thread(command_receiver.run, (port,))
    time.sleep(1)
    return command_receiver


############################

class Context(object):
    pass


# IMPORTANT! Something strange happened when there was a main() here. Inside main I only had
# access to __main__.command_receiver, which was different that training_utils.command_receiver
