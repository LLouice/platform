'''
SaveModel
'''

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
# import tensorflow as tf
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from result import Err, Ok, Result
from tensorflow.python import debug as tf_debug

import logger
from ada_predict import build_graph
# from utils import set_gpu
from config import build_config_proto
from const import TEST_SIZE, TRAIN_SIZE, VAL_SIZE
from logger import get_logger

logger = get_logger("train")


@dataclass
class TrainConfig:
    logdir: str
    repeat_num: int
    visible_device_list: str = "2"
    log_device_placement: bool = False
    train_size: int = TRAIN_SIZE
    val_size: int = VAL_SIZE
    test_size: int = TEST_SIZE
    batch_size_trn: int = 4
    batch_size_dev: int = 8
    lr: float = 0.001
    wd: float = 0.0003  # 3e-4
    opt: str = "Adam"
    epochs: int = 100
    epoch_start: int = 1
    steps: Optional[int] = None
    eval_interval: int = 5
    ckpt_dir: str = "checkpoints"
    ckpt: Optional[str] = None
    ex: str = "default"
    model_name: str = "AdaE"
    suffix: str = ""
    use_pretrained: bool = False
    use_transe: bool = False
    use_masked_label: bool = True
    q: float = 0.7
    alpha: float = 0.1
    beta: float = 1.0
    A: float = 0.0001
    debug: bool = False

    def destruct(self):
        return self.logdir, \
               self.repeat_num, \
               self.visible_device_list, \
               self.log_device_placement, \
               self.train_size, \
               self.val_size,  \
               self.test_size,  \
               self.batch_size_trn, \
               self.batch_size_dev, \
               self.lr, \
               self.wd, \
               self.opt, \
               self.epochs, \
               self.epoch_start, \
               self.steps, \
               self.eval_interval, \
               self.ckpt_dir, \
               self.ckpt, \
               self.ex, \
               self.model_name, \
               self.suffix, \
               self.use_pretrained, \
               self.use_transe, \
               self.use_masked_label, \
               self.q, \
               self.alpha, \
               self.beta, \
               self.A, \
               self.debug,

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> TrainConfig:
        logdir = f"./logs/{args.ex}"

        if args.ckpt:
            args.ckpt = Ckpt.from_str(args.ckpt).ok()
            epoch_start = args.ckpt.epoch
        else:
            epoch_start = 1

        if not args.steps:
            args.steps = args.train_size // args.batch_size_trn

        repeat_num = args.epochs * args.steps
        use_masked_label = not args.no_masked_label

        return cls(
                logdir, \
                repeat_num, \
                args.visible_device_list, \
                args.log_device_placement, \
                args.train_size, \
                args.val_size, \
                args.test_size, \
                args.batch_size_trn, \
                args.batch_size_dev, \
                args.lr, \
                args.wd, \
                args.opt, \
                args.epochs, \
                epoch_start, \
                args.steps, \
                args.eval_interval, \
                args.ckpt_dir, \
                args.ckpt, \
                args.ex, \
                args.model_name, \
                args.suffix, \
                args.use_pretrained, \
                args.use_transe, \
                use_masked_label, \
                args.q, \
                args.alpha, \
                args.beta, \
                args.A, \
                args.debug,
            )


@dataclass
class Ckpt:
    rank: float
    loss: float
    epoch: int
    step: int

    def __str__(self) -> str:
        return f"rank_{self.rank}_loss_{self.loss}_epoch_{self.epoch}_step_{self.step}"

    @classmethod
    def from_str(cls, s: str) -> Result[Ckpt, Exception]:
        try:
            rank, loss, epoch, step = s[:s.rfind('.')].strip().split("_")[1::2]
            rank = float(rank)
            loss = float(loss)
            epoch = int(epoch)
            step = int(step)
            return Ok(cls(rank, loss, epoch, step))
        except Exception as e:
            return Err(e)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.info("start...")
    args = parse_cli()
    train_config = TrainConfig.from_cli(args)
    logger.info(train_config)
    # tf.disable_eager_execution()
    run(train_config)


def parse_cli():
    parser = argparse.ArgumentParser(description="AdaE-train")
    #Adding optional parameters
    parser.add_argument(
        "-v",
        "--visible_device_list",
        help="visible_device_list",
        default="3",
    )
    parser.add_argument('-V',
                        "--log_device_placement",
                        help="Whether device placements should be logged",
                        action="store_true")

    parser.add_argument("--train_size",
                        help="the subsize of train",
                        default=TRAIN_SIZE,
                        type=int)
    parser.add_argument("--val_size",
                        help="the subsize of val",
                        default=VAL_SIZE,
                        type=int)
    parser.add_argument("--test_size",
                        help="the subsize of test",
                        default=TEST_SIZE,
                        type=int)

    parser.add_argument("--batch_size_trn",
                        help="batch size train",
                        default=4,
                        type=int)
    parser.add_argument("--batch_size_dev",
                        help="batch size dev",
                        default=8,
                        type=int)

    parser.add_argument("--lr",
                        help="learning rate",
                        default=0.001,
                        type=float)
    parser.add_argument("--wd",
                        help="weight decay",
                        default=0.0003,
                        type=float)
    parser.add_argument("--opt",
                        help="select optimizer",
                        choices=("Adam", "Sgd"),
                        default="Adam")
    parser.add_argument("-e",
                        "--epochs",
                        help="training epochs",
                        default=100,
                        type=int)
    parser.add_argument("--steps", help="steps", type=int)
    parser.add_argument("-I",
                        "--eval_interval",
                        help="eval interval",
                        default=5,
                        type=int)
    parser.add_argument(
        "-D",
        "--ckpt_dir",
        help="checkpoint dir",
        default="checkpoints",
    )
    parser.add_argument(
        "-R",
        "--ckpt",
        help="ckeckpoint for resume training",
    )
    parser.add_argument(
        "-E",
        "--ex",
        help="experiment name",
        default="default",
    )
    parser.add_argument(
        "-M",
        "--model_name",
        choices=["AdaE", "AdaE2", "AdaE3", "ConvE"],
        default="AdaE",
    )
    parser.add_argument(
        "--suffix",
        choices=("", "_dis"),
        default="",
    )
    parser.add_argument('--use_pretrained',
                        help="use pretrained w2v embeddings",
                        action="store_true")
    parser.add_argument('--use_transe', help="use transe", action="store_true")
    parser.add_argument('--no_masked_label',
                        help="use masked label",
                        action="store_true")
    parser.add_argument('--q', help="GCE q", default=0.7, type=float)
    # SCE
    parser.add_argument('--alpha', help="SCE alpha", default=0.1, type=float)
    parser.add_argument('--beta', help="SCE beta", default=1.0, type=float)
    parser.add_argument('--A', help="SCE A", default=0.0001, type=float)
    parser.add_argument('--debug', help="debug mode", action="store_true")

    args = parser.parse_args()
    return args


def load_pretrained_embeddings(suffix):
    pretrained_embeddings = np.load(f"assets/embeddings{suffix}.npy")
    return pretrained_embeddings


def run(train_config: TrainConfig) -> Result[None, str]:
    logdir, repeat_num, visible_device_list, log_device_placement, train_size, val_size, test_size, batch_size_trn, batch_size_dev, lr, wd, opt, epochs, epoch_start, steps, eval_interval, ckpt_dir, ckpt, ex, model_name, suffix, use_pretrained, use_transe, use_masked_label, q, alpha, beta, A, debug_mode = train_config.destruct(
    )

    # nogpu
    # config_proto = build_config_proto(visible_device_list,
    #                                   log_device_placement)

    # build network
    build_graph(
        model_name=model_name,
        use_transe2=use_transe,
        use_masked_label=use_masked_label,
        q=q,
        alpha=alpha,
        beta=-beta,
        A=A,
    )

    # session = tf.Session(config=config_proto)
    # nogpu
    session = tf.Session()
    graph = session.graph

    # ops
    op_init = graph.get_operation_by_name("init")

    # ph_pretrained_embeddings = graph.get_tensor_by_name(
    #     "custom/pretrained_embeddings:0")
    # ph_use_transe = graph.get_tensor_by_name("custom/use_transe:0")

    # saver = tf.train.Saver(tf.global_variables())
    # saver = tf.train.Saver()

    # save op
    ph_file_path = graph.get_tensor_by_name("save/Const:0")
    op_save = graph.get_operation_by_name("save/control_dependency")
    _file_path_tensor = "checkpoints/saved.ckpt"

    # if use_transe:
    #     feed_dict[ph_use_transe] = True
    # if use_pretrained:
    #     feed_dict[ph_pretrained_embeddings] = load_pretrained_embeddings(
    #         suffix)

    # init step
    def init():
        session.run([op_init], )

    # init()

    def data():
        init()
        # inspect data
        print(epochs)
        print(steps)
        print(repeat_num)
        for _ in range(epochs):
            for _ in range(steps):
                pass
                # session.run(op_batch_data_trn_print)
                # session.run(op_batch_data_val_print)
                # session.run(op_batch_data_test_print)
                # print("===>", res.shape)
                print("-" * 30)
                # break
            print("=" * 60)
            # break

    def save(ckpt: Ckpt):
        ckpt_path = f"{ckpt_dir}/{ex}/{ckpt}"
        session.run(op_save, feed_dict={ph_file_path: ckpt_path})

    def restore(ckpt):
        # init()
        ckpt_path = f"{ckpt_dir}/{ex}/{ckpt}"
        op_load = graph.get_operation_by_name(("save/restore_all"))
        session.run(op_load, feed_dict={ph_file_path: ckpt_path})

    def start(session, ckpt: Ckpt):
        # init()
        if ckpt:
            logger.info(f"restore form ckpt {ckpt}")
            restore(ckpt)
            logger.info("restore done!")

        # prediction
        ph_e1 = graph.get_tensor_by_name("input/e1:0")
        ph_rel = graph.get_tensor_by_name("input/rel:0")
        prediction = graph.get_tensor_by_name("prediction:0")

        # e1 = np.array([1, 2, 3])
        # rel = np.array([1, 2, 3])
        # prediction = session.run(graph.get_tensor_by_name("prediction:0"),
        #                          feed_dict={
        #                              ph_e1: e1,
        #                              ph_rel: rel
        #                          })
        # print("prediction is: ", prediction)

        # saving models
        tf.saved_model.simple_save(session,
                                   "SavedModel",
                                   inputs={
                                       "e1": ph_e1,
                                       'rel': ph_rel,
                                   },
                                   outputs={"prediciton": prediction})

        # end saving models
        session.close()

    if debug_mode:
        data()
    else:
        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        start(session, ckpt)
        # pass


if __name__ == "__main__":
    main()
