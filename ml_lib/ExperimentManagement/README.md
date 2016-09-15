# ExperimentManager

1. Create a subclass of [`Trainer`](https://github.com/codilime/ml_lib/blob/master/ExperimentManagement/trainer.py) class.
2. Implement go method.
  ```python
  def go(self, exp, args):
        # more or less mandatory
        self.exp = exp
        self.args = args

        self.exp.set_status(Experiment.Status.RUNNING)
        self.create_bokeh_session()
        self.ts = self.create_timeseries_and_figures()
        self.init_command_receiver()

        # user specific
        self.some_local_main()

        return 0
  ```
  exp is instance of [`Experiment`](https://github.com/codilime/ml_lib/blob/master/ExperimentManagement/mongo_resources.py) class. You can set various properties and information about your experiment using this class. Everything is saved in a database. See the definition for more information.
  To use the [`CommandReceiver`](https://github.com/codilime/ml_lib/blob/master/TheanoLib/training_utils.py) (`self.init_command_receiver()`) you should set `MY_PUBLIC_IP` enviromental variable. The best way to set this is append this line:
  ```export MY_PUBLIC_IP=`curl -s http://169.254.169.254/latest/meta-data/public-ipv4` ``` 
  to your `.bashrc` on your AWS machine.
  
3. Implement `create_timeseries_and_figures` method. This method allows you to create figures that will be visible in the Zeus frontend.
4. 
  ```python
  def create_timeseries_and_figures(self, optim_state={}):
        # average over the last FREQ values before plotting
        FREQ1 = 10
        FREQ2 = 1

        channels = ['train_cost',
                    'train_loss',
                    'train_l2_reg_cost',
                    'train_per_example_proc_time_ms',
                    'train_per_example_load_time_ms',
                    'train_per_example_whole_time_ms',
                    'l2_reg_global',
                    'val_loss',
                    'train_epoch_time',
                    'valid_epoch_time',
                    'act_glr'
                    ]

        figures_schema = OrderedDict([
            ('train', [
                ('train_cost', 'cost', FREQ1),
                ('train_loss', 'loss', FREQ1),
                ('train_l2_reg_cost', 'l2_reg_cost', FREQ1)
            ]),

            ('valid', [
                ('val_loss', 'loss', FREQ2)
            ]),

            ('perf', [
                ('train_per_example_proc_time_ms', 'train_per_example_proc_ms', FREQ1),
                ('train_per_example_load_time_ms', 'train_per_example_load_ms', FREQ1),
                ('train_per_example_whole_time_ms', 'train_per_example_whole_ms', FREQ1)
            ]),

            ('perf_2', [
                ('train_epoch_time', 'train_epoch_time', 1),
                ('valid_epoch_time', 'valid_epoch_time', 1)
            ]),

            ('act_glr', [
                ('act_glr', 'act_glr', 1)
            ])

        ])
        return self._create_timeseries_and_figures(channels, figures_schema)
  ```
4. Make sure to call `handle_commands` often (for instance every minibatch).
  ```python
  handle_commands(self.command_receiver, None)
  ```
  If you want to pass command to your program you should also pass some context to this method, see [Maciek's code](https://github.com/codilime/deeplearning/blob/whales/whales/main.py). For example you can send commands to decrease learning rate without stopping your experiment, or change the minibatch size to see if you get better performance.
  
5. To "log" some quantity specified in `create_timeseries_and_figures` method, call .add(value) on an apropriate attribute.
  ```python
  self.ts.train_loss.add(mb_loss)
  ```
  
6. To start everything use the main method and specify the owner.
  ```python
  trainer = ConvNetTrainer()  # subclass of Trainer
  sys.exit(trainer.main(default_owner='Jane Doe'))
    
  ```
There are other control arguments that you can pass to your program. For example to queue your experiment for execution by workers use `--only-queue`. See [`Trainer`](https://github.com/codilime/ml_lib/blob/master/ExperimentManagement/trainer.py) class for more information, or ask Maciek.

7. You can see list of your experiments here: `http://52.8.123.153:5000/experiment/list/OWNER`
8. See [example](https://github.com/codilime/deeplearning/blob/whales/whales/main.py) for nice way you can handle logging in your code (Tee class).
9. See [example](https://github.com/codilime/deeplearning/blob/whales/whales/main.py) for nice way you can save files in your code(ExperimentSaver class).
10. It is a very good practice to read all your files, and save all files to some global location reachable from everywhere(your EBS, hard disk is not). We our NFS server for this. See Example for proper usage:
 path arguments should be URLs, e.g. //nfs/maciek/whales/input.txt //nfs_out/maciek/whales/ and their names should have _url prefix. You should also have `$NFS`, `$NFS_OUT`, environment variables set, which should point to the mount points of the NFS, NFS_OUT directories. Make sure you mount /home/ubuntu/ebs/NFS, /home/ubuntu/ebs/NFS_out, directories. URLs will be translated to local path (using `$NFS`, `$NFS_OUT`), and additional arguments will be added to argparse Namespace namely, for any arguemnt something_url, something_path, will be added.

You can see [Maciek's code](https://github.com/codilime/deeplearning/blob/whales/whales/main.py) for example of usage of all things discussed or ask Maciek if you have any questions.
