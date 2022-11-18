from cavachon.workflow.Workflow import Workflow

import argparse
import os
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

try:
  physical_devices = tf.config.list_physical_devices('GPU')
  for i, d in enumerate(physical_devices):
      tf.config.experimental.set_memory_growth(physical_devices[i], True)
except:
  print('No GPU detected. Use CPU instead')

class CavachonApplication:
  def __init__(self):
    self.args = None

  def run(self):
    for config in self.args.configs:
      workflow = Workflow(config)
      workflow.run()

  def parse_arguments(self):
    parser = argparse.ArgumentParser(
        description=''.join((
            'Cell cluster Analysis with Variational Autoencoder using Conditional Hierarchy ',
            'Of latent representioN')))

    parser.add_argument(
        'configs',
        type=str, 
        default='config.yaml', 
        nargs='+', 
        metavar='config.yaml',
        help="the config files for the experiments.")

    args = parser.parse_args()
    args.configs = [os.path.realpath(x) for x in args.configs]
    self.args = args

def main():
  app = CavachonApplication()
  app.parse_arguments()
  app.run()

if __name__ == "__main__":
  main()