from __future__ import annotations

import argparse
from dataclasses import dataclass
# import tensorflow as tf
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from result import Err, Ok, Result
from tensorflow.python import debug as tf_debug

import logger
from ada import build_graph
# from utils import set_gpu
from config import build_config_proto
from const import TEST_SIZE, TRAIN_SIZE, VAL_SIZE
from logger import get_logger

logger = get_logger("train")


class Dev(Enum):
    Val = 1
    Test = 2


@dataclass
class EvalValue:
    rank: float
    hit1: float
    hit3: float
    hit10: float


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
    opt: str = "Adam"
    epochs: int = 100
    epoch_start: int = 1
    steps: Optional[int] = None
    eval_interval: int = 5
    ckpt_dir: str = "checkpoints"
    ckpt: Optional[str] = None
    ex: str = "default"
    model_name: str = "AdaE"
    use_pretrained: bool = False
    use_transe: bool = False
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
               self.opt, \
               self.epochs, \
               self.epoch_start, \
               self.steps, \
               self.eval_interval, \
               self.ckpt_dir, \
               self.ckpt, \
               self.ex, \
               self.model_name, \
               self.use_pretrained, \
               self.use_transe, \
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
                args.opt, \
                args.epochs, \
                epoch_start, \
                args.steps, \
                args.eval_interval, \
                args.ckpt_dir, \
                args.ckpt, \
                args.ex, \
                args.model_name, \
                args.use_pretrained, \
                args.use_transe, \
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
    logger.info("start...")
    args = parse_cli()
    train_config = TrainConfig.from_cli(args)
    logger.info(train_config)
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
        choices=["AdaE", "ConvE"],
        default="AdaE",
    )
    parser.add_argument('--use_pretrained',
                        help="use pretrained w2v embeddings",
                        action="store_true")
    parser.add_argument('--use_transe', help="use transe", action="store_true")
    parser.add_argument('--debug', help="debug mode", action="store_true")

    args = parser.parse_args()
    return args


def load_pretrained_embeddings():
    pretrained_embeddings = np.load("assets/embeddings.npy")
    return pretrained_embeddings


def run(train_config: TrainConfig) -> Result[None, str]:
    logdir, repeat_num, visible_device_list, log_device_placement, train_size, val_size, test_size, batch_size_trn, batch_size_dev, lr, opt, epochs, epoch_start, steps, eval_interval, ckpt_dir, ckpt, ex, model_name, use_pretrained, use_transe, debug_mode = train_config.destruct(
    )

    config_proto = build_config_proto(visible_device_list,
                                      log_device_placement)

    # build network
    build_graph(model_name=model_name, use_transe2=use_transe)

    session = tf.Session(config=config_proto)
    graph = session.graph

    # summary
    writer = tf.summary.FileWriter(logdir, graph)

    # ops
    op_init = graph.get_operation_by_name("init")

    ph_batch_size_trn = graph.get_tensor_by_name("custom/batch_size_trn:0")
    ph_batch_size_dev = graph.get_tensor_by_name("custom/batch_size_dev:0")
    ph_repeat = graph.get_tensor_by_name("custom/repeat:0")
    ph_lr = graph.get_tensor_by_name("custom/lr:0")

    ph_train_size = graph.get_tensor_by_name("custom/train_size:0")
    ph_val_size = graph.get_tensor_by_name("custom/val_size:0")
    ph_test_size = graph.get_tensor_by_name("custom/test_size:0")

    ph_pretrained_embeddings = graph.get_tensor_by_name(
        "custom/pretrained_embeddings:0")
    ph_use_transe = graph.get_tensor_by_name("custom/use_transe:0")
    # dataset
    op_batch_data_trn_init = graph.get_operation_by_name(
        "data_trn/dataset_trn/MakeIterator")

    op_batch_data_trn = graph.get_operation_by_name(
        "data_trn/dataset_trn/get_next")

    # op_batch_data_trn_print = graph.get_operation_by_name(
    #     "print_data_trn/print")
    # op_batch_data_val_print = graph.get_operation_by_name(
    #     "print_data_val/print")
    # op_batch_data_test_print = graph.get_operation_by_name(
    #     "print_data_test/print")

    ph_trn_record_path = graph.get_tensor_by_name(
        "data_trn/dataset_trn/Const:0")
    # based on current dir
    trn_record_path = "assets/symptom_trn.tfrecord"

    op_batch_data_val_init = graph.get_operation_by_name(
        "data_val/dataset_val/MakeIterator")
    _op_batch_data_val = graph.get_operation_by_name(
        "data_val/dataset_val/get_next")
    ph_val_record_path = graph.get_tensor_by_name(
        "data_val/dataset_val/Const:0")
    # based on current dir
    val_record_path = "assets/symptom_val.tfrecord"

    op_batch_data_test_init = graph.get_operation_by_name(
        "data_test/dataset_test/MakeIterator")
    _op_batch_data_test = graph.get_operation_by_name(
        "data_test/dataset_test/get_next")
    ph_test_record_path = graph.get_tensor_by_name(
        "data_test/dataset_test/Const:0")
    # based on current dir
    test_record_path = "assets/symptom_test.tfrecord"

    # train op
    op_optimize = graph.get_operation_by_name("optimizer/optimize")
    if opt == "Sgd":
        op_optimize = graph.get_operation_by_name("optimizer/optimize_sgd")
    elif opt != "Adam":
        raise Exception(f"the Optimizer {opt} is not support!")
    else:
        pass

    op_global_step = graph.get_operation_by_name("optimizer/global_step")
    t_loss = graph.get_tensor_by_name("loss/loss:0")
    t_loss_model = graph.get_tensor_by_name("loss/loss_model:0")
    t_loss_margin = graph.get_tensor_by_name("loss/loss_margin:0")

    # eval op
    op_rank_val = graph.get_operation_by_name("eval_val/rank_val")
    op_hit1_val = graph.get_operation_by_name("eval_val/hits1_val")
    op_hit3_val = graph.get_operation_by_name("eval_val/hits3_val")
    op_hit10_val = graph.get_operation_by_name("eval_val/hits10_val")
    op_eval_val = graph.get_operation_by_name("eval_val/eval_op_val")

    op_rank_test = graph.get_operation_by_name("eval_test/rank_test")
    op_hit1_test = graph.get_operation_by_name("eval_test/hits1_test")
    op_hit3_test = graph.get_operation_by_name("eval_test/hits3_test")
    op_hit10_test = graph.get_operation_by_name("eval_test/hits10_test")
    op_eval_test = graph.get_operation_by_name("eval_test/eval_op_test")

    # save op
    ph_file_path = graph.get_tensor_by_name("save/Const:0")
    op_save = graph.get_operation_by_name("save/control_dependency")
    _file_path_tensor = "checkpoints/saved.ckpt"

    # summary
    op_summary = graph.get_operation_by_name("summaries/summary_op/summary_op")
    op_val_summary = graph.get_operation_by_name(
        "summaries/summary_val_op/summary_val_op")
    op_grads_summary = graph.get_operation_by_name(
        "gradients/summary_grads_op/summary_grads_op")
    op_weights_summary = graph.get_operation_by_name(
        "weights/summary_weights_op/summary_weights_op")

    ph_rank_val = graph.get_tensor_by_name("summaries/rank_val:0")
    ph_hit1_val = graph.get_tensor_by_name("summaries/hit1_val:0")
    ph_hit3_val = graph.get_tensor_by_name("summaries/hit3_val:0")
    ph_hit10_val = graph.get_tensor_by_name("summaries/hit10_val:0")

    # feed config data
    feed_dict = {
        ph_batch_size_trn: np.int64(batch_size_trn),
        ph_batch_size_dev: np.int64(batch_size_dev),
        ph_repeat: np.int64(repeat_num),
        ph_lr: lr,
        ph_train_size: np.int64(train_size),
        ph_val_size: np.int64(val_size),
        ph_test_size: np.int64(test_size),
        ph_trn_record_path: trn_record_path,
        ph_val_record_path: val_record_path,
        ph_test_record_path: test_record_path,
        # ph_pretrained_embeddings: load_pretrained_embeddings(),
    }
    if use_transe:
        feed_dict[ph_use_transe] = True
    if use_pretrained:
        feed_dict[ph_pretrained_embeddings] = load_pretrained_embeddings()

    # init step
    def init():
        session.run([
            op_init, op_batch_data_trn_init, op_batch_data_val_init,
            op_batch_data_test_init
        ],
                    feed_dict=feed_dict)

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
        init()
        ckpt_path = f"{ckpt_dir}/{ex}/{ckpt}"
        op_load = graph.get_operation_by_name("save/restore_all")
        session.run(op_load, feed_dict={ph_file_path: ckpt_path})

    def eval(dev: Dev, global_step: int) -> EvalValue:
        feed_dict = {
            ph_batch_size_dev: np.int64(batch_size_dev),
            ph_val_size: np.int64(val_size),
            ph_test_size: np.int64(test_size),
            ph_val_record_path: val_record_path,
            ph_test_record_path: test_record_path,
        }
        steps = None
        fetches = None
        if dev == Dev.Val:
            logger.info("re-init dev dataset")
            session.run(op_batch_data_val_init, feed_dict=feed_dict)
            steps = val_size // batch_size_dev
            fetches = [
                op_rank_val.outputs[0],
                op_hit1_val.outputs[0],
                op_hit3_val.outputs[0],
                op_hit10_val.outputs[0],
            ]
        else:
            logger.info("re-init dev dataset")
            session.run(op_batch_data_test_init, feed_dict=feed_dict)
            steps = test_size // batch_size_dev
            fetches = [
                op_rank_test.outputs[0],
                op_hit1_test.outputs[0],
                op_hit3_test.outputs[0],
                op_hit10_test.outputs[0],
            ]

        ranks = []
        hit1s = []
        hit3s = []
        hit10s = []

        for i in range(steps):
            # session.run(op_batch_data_val_init, feed_dict=feed_dict)
            logger.info(f"eval [{i}]")
            rank, hit1, hit3, hit10 = session.run(fetches, feed_dict=feed_dict)
            ranks.append(rank)
            hit1s.append(hit1)
            hit3s.append(hit3)
            hit10s.append(hit10)

        rank = sum(ranks) / len(ranks)
        hit1 = sum(hit1s) / len(hit1s)
        hit3 = sum(hit3s) / len(hit3s)
        hit10 = sum(hit10s) / len(hit10s)

        summaries = session.run(op_val_summary.outputs[0],
                                feed_dict={
                                    ph_rank_val: rank,
                                    ph_hit1_val: hit1,
                                    ph_hit3_val: hit3,
                                    ph_hit10_val: hit10,
                                })
        writer.add_summary(summaries, global_step)

        value = EvalValue(rank, hit1, hit3, hit10)
        logger.info(value)
        return value

    # train
    def train(ckpt: Ckpt):
        init()
        if ckpt:
            logger.info(f"restore form ckpt {ckpt}")
            restore(ckpt)

        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        global_step = 0
        for e in range(epoch_start, epochs + 1):
            total_loss = 0.
            total_loss_model = 0.
            total_loss_margin = 0.
            for i in range(steps):
                if i == 0:
                    loss, loss_model, loss_margin, summaries, summaries_grads, summaries_weights, global_step, _ = session.run(
                        [
                            t_loss, t_loss_model, t_loss_margin,
                            op_summary.outputs[0], op_grads_summary.outputs[0],
                            op_weights_summary.outputs[0],
                            op_global_step.outputs[0], op_optimize
                        ], )
                    writer.add_summary(summaries_grads, global_step)
                    writer.add_summary(summaries_weights, global_step)
                else:
                    loss, loss_model, loss_margin, summaries, global_step, _ = session.run(
                        [
                            t_loss, t_loss_model, t_loss_margin,
                            op_summary.outputs[0], op_global_step.outputs[0],
                            op_optimize
                        ], )

                # feed_dict={ph_batch_size_dev: np.int64(batch_size_dev)})
                total_loss += loss
                total_loss_model += loss_model
                total_loss_margin += loss_margin

                writer.add_summary(summaries, global_step)

            loss = total_loss / steps
            loss_model = total_loss_model / steps
            loss_margin = total_loss_margin / steps
            logger.info(
                f"[{e}] loss: {loss}\tloss_model: {loss_model}\tloss_margin: {loss_margin}"
            )

            if e % eval_interval == 0:
                eval_value = eval(Dev.Val, global_step)
                ckpt = Ckpt(eval_value.rank, loss, e, global_step)
                save(ckpt)

        session.close()
        writer.close()

    if debug_mode:
        data()
    else:
        train(ckpt)


if __name__ == "__main__":
    main()
