import os
import yaml
import torch
import shutil
import random
import datetime
import argparse
import subprocess
import sys
import numpy as np
from pathlib import Path
from shutil import copyfile

from modules.trainer import Trainer
from utils.mlflow_utils import MLflowTracker, TeeStream

def seed_everything(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
    print(f'Using seed: {seed}')

if __name__ == '__main__':
    seed_everything()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    console_stream = None
    tracker = None

    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        default='config/RangeRet-semantickitti.yaml',
        help='Architecture yaml cfg file. See /config/ for sample. No default!',
    )
    parser.add_argument(
        '--data',
        type=str,
        required=False,
        default='config/labels/semantic-kitti.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.getcwd() + '/log/rangeret' + '/',
        help='Directory to put the log data. Default: ./log/rangeret'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        default=None,
        help='File to the checkpoint model to resume training. If not passed, do from scratch!'
    )
    parser.add_argument(
        '--pretrained-model',
        type=str,
        required=False,
        default=None,
        help='File to get the pretrained RetNet model. If not passed, do from scratch!'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        default=False,
        help='Use mixed precision training. Default: False'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("config", FLAGS.config)
    print("data", FLAGS.data)
    print("log", FLAGS.log)
    print("checkpoint", FLAGS.checkpoint)
    print("pretrained retnet", FLAGS.pretrained_model)
    print("fp16", FLAGS.fp16)
    print("----------\n")
    #print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.config)
        ARCH = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()
    
    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data)
        DATA = yaml.safe_load(open(FLAGS.data, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()
    
    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    console_path = Path(FLAGS.log) / "console.log"
    console_stream = open(console_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(original_stdout, console_stream)
    sys.stderr = TeeStream(original_stderr, console_stream)

    # does model folder exist?
    if FLAGS.checkpoint is not None:
        if os.path.isfile(FLAGS.checkpoint):
            print("pretrained model found! Using model from %s" % (FLAGS.checkpoint))
        else:
            print("model folder doesnt exist! Start with random weights...")
    else:
        print("No pretrained model found.")

    # copy all files to log folder (to remember what we did, and make inference
    # easier). Also, standardize name to be able to open it later
    try:
        print("Copying files to %s for further reference." % FLAGS.log)
        copyfile(FLAGS.config, str(Path(FLAGS.log) / Path(FLAGS.config).name))
        copyfile(FLAGS.data, str(Path(FLAGS.log) / Path(FLAGS.data).name))
        #copyfile(FLAGS.data_cfg, FLAGS.log + "/semantic-kitti.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting...")
        quit()

    tracker = MLflowTracker(ARCH.get("mlflow"), FLAGS.log, run_name_default=Path(FLAGS.log).name)
    tracker.start(
        params={
            "dataset_dir": FLAGS.dataset,
            "config_path": FLAGS.config,
            "data_config_path": FLAGS.data,
            "checkpoint": FLAGS.checkpoint,
            "pretrained_model": FLAGS.pretrained_model,
            "fp16": FLAGS.fp16,
            "model_name": ARCH.get("model", {}).get("name", ARCH.get("model_params", {}).get("model_architecture")),
            "dataset_type": ARCH["dataset"]["pc_dataset_type"],
            "epochs": ARCH["train"]["epochs"],
            "batch_size": ARCH["train"]["batch_size"],
            "learning_rate": ARCH["train"]["learning_rate"],
            "optimizer": ARCH["train"]["optimizer"],
        }
    )

    try:
        trainer = Trainer(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.checkpoint, FLAGS.pretrained_model, FLAGS.fp16, tracker=tracker)
        trainer.train()
    except Exception:
        if tracker is not None:
            tracker.finish(status="FAILED")
        raise
    else:
        if tracker is not None:
            tracker.finish(status="FINISHED")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if console_stream is not None:
            console_stream.close()
