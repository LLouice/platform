# this file explore to load tfrecord

import tensorflow as tf
from tensorflow.python import debug as tfdbg
from utils import set_gpu, write_graph, scatter_update_tensor

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N = 28754


def build_train_dataset(repeat_num, batch_size=4):
    with tf.name_scope('dataset_trn'):
        dataset_trn = tf.data.TFRecordDataset("symptom_trn.tfrecord",
                                              num_parallel_reads=8)

        # Create a description of the features.
        feature_description_trn = {
            'input':
            tf.io.FixedLenSequenceFeature([],
                                          tf.int64,
                                          default_value=0,
                                          allow_missing=True),
            'label':
            tf.io.FixedLenSequenceFeature([],
                                          tf.int64,
                                          default_value=0,
                                          allow_missing=True),
            'num_ent':
            tf.io.FixedLenSequenceFeature([],
                                          tf.int64,
                                          default_value=0,
                                          allow_missing=True),
        }

        def _parse_function_trn(example_proto):
            return tf.io.parse_single_example(example_proto,
                                              feature_description_trn)

        def _transform0(d):
            with tf.name_scope("transform"):
                zeros = tf.Variable(lambda: tf.zeros(N), name="zeros")
                label = tf.convert_to_tensor(d["label"], dtype=tf.int64)
                # eps = tf.Variable(lambda: tf.constant(eps), name="eps")
                label = tf.scatter_nd_update(
                    zeros, label, tf.ones_like(label, dtype=tf.float32))
                # soft_label = tf.divide(1., 1. - eps) * label + tf.divide(1.0, N)
                d["label"] = label
                return d

        def _transform1(d):
            with tf.name_scope("transform"):
                zeros = tf.zeros(N, name="zeros")
                label = tf.convert_to_tensor(d["label"], dtype=tf.int64)
                # eps = tf.Variable(lambda: tf.constant(eps), name="eps")
                # [m,] => [m, 1]
                d["num_label"] = tf.shape(label)
                label = scatter_update_tensor(
                    zeros, tf.cast(tf.expand_dims(label, axis=1), tf.int32),
                    tf.ones_like(label, dtype=tf.float32))
                # soft_label = tf.divide(1., 1. - eps) * label + tf.divide(1.0, N)
                d["label"] = label
                return d

        def _transform(d):
            with tf.name_scope("transform"):
                zeros = tf.zeros(N, name="zeros")
                label = tf.convert_to_tensor(d["label"], dtype=tf.int64)
                # eps = tf.Variable(lambda: tf.constant(eps), name="eps")
                # [m,] => [m, 1]
                d["num_label"] = tf.shape(label)
                label = scatter_update_tensor(
                    zeros, tf.cast(tf.expand_dims(label, axis=1), tf.int32),
                    tf.ones_like(label, dtype=tf.float32))
                # soft_label = tf.divide(1., 1. - eps) * label + tf.divide(1.0, N)
                d["label"] = label
                return d

        dataset_trn = dataset_trn.map(_parse_function_trn).map(_transform)

        iterator_trn = dataset_trn.shuffle(30000).repeat(repeat_num).batch(
            batch_size).prefetch(batch_size).make_initializable_iterator()
        # iterator = dataset.shuffle(30000).repeat(10).batch(4).prefetch(4).make_one_shot_iterator()
        # iterator = dataset.make_one_shot_iterator()
        # iterator = dataset.make_initializable_iterator()
        batch_data_trn = iterator_trn.get_next("get_next")
    return iterator_trn, batch_data_trn


def build_dev_dataset(dev="test", batch_size=4):
    with tf.name_scope(f"dataset_{dev}"):
        dataset_dev = tf.data.TFRecordDataset(f"symptom_{dev}.tfrecord",
                                              num_parallel_reads=8)
        feature_description_dev = {
            'input':
            tf.io.FixedLenSequenceFeature([],
                                          tf.int64,
                                          default_value=0,
                                          allow_missing=True),
            # 'num_ent':
            # tf.io.FixedLenSequenceFeature([],
            #                               tf.int64,
            #                               default_value=0,
            #                               allow_missing=True),
            'num_ent':
            tf.io.FixedLenFeature(
                [1],
                tf.int64,
                default_value=0,
            ),
            'label':
            tf.io.FixedLenSequenceFeature([],
                                          tf.int64,
                                          default_value=0,
                                          allow_missing=True),
            'aux_label':
            tf.io.FixedLenSequenceFeature([],
                                          tf.int64,
                                          default_value=0,
                                          allow_missing=True),
        }

        def _parse_function_dev(example_proto):
            return tf.io.parse_single_example(example_proto,
                                              feature_description_dev)

        def _transform0(d):
            with tf.name_scope("transform"):
                zeros = tf.Variable(lambda: tf.zeros(N), name="zeros")
                aux_label = tf.convert_to_tensor(d["aux_label"],
                                                 dtype=tf.int64)
                aux_label = tf.scatter_nd_update(
                    zeros, aux_label, tf.ones_like(aux_label,
                                                   dtype=tf.float32))
                d["aux_label"] = aux_label
                return d

        def _transform1(d):
            with tf.name_scope("transform"):
                zeros = tf.zeros(N, name="zeros")

                aux_label = tf.convert_to_tensor(d["aux_label"],
                                                 dtype=tf.int64)
                d["num_aux_label"] = tf.shape(aux_label)
                # [m,] => [m, 1]
                aux_label = scatter_update_tensor(
                    zeros, tf.cast(tf.expand_dims(aux_label, axis=1),
                                   tf.int32),
                    tf.ones_like(aux_label, dtype=tf.float32))
                # soft_label = tf.divide(1., 1. - eps) * label + tf.divide(1.0, N)
                d["aux_label"] = aux_label
                d["label"] = tf.RaggedTensor.from_tensor(
                    tf.expand_dims(d["label"], axis=0))
                return d

        def _transform(d):
            with tf.name_scope("transform"):
                zeros = tf.zeros(N, name="zeros")

                aux_label = tf.convert_to_tensor(d["aux_label"],
                                                 dtype=tf.int64)
                # [m,] => [m, 1]
                aux_label = scatter_update_tensor(
                    zeros, tf.cast(tf.expand_dims(aux_label, axis=1),
                                   tf.int32),
                    tf.ones_like(aux_label, dtype=tf.float32))
                # soft_label = tf.divide(1., 1. - eps) * label + tf.divide(1.0, N)
                d["aux_label"] = aux_label
                d["label"] = tf.RaggedTensor.from_tensor(
                    tf.expand_dims(d["label"], axis=0))
                return d

        dataset_dev = dataset_dev.map(_parse_function_dev).map(_transform)

        iterator_dev = dataset_dev.batch(batch_size).prefetch(
            batch_size).make_initializable_iterator()
        batch_data_dev = iterator_dev.get_next("get_next")
    return iterator_dev, batch_data_dev


def eval_inner(logits, labels, auxes, batch_size=3):
    with tf.variable_scope("eval"):
        preds = tf.nn.sigmoid(logits, name="preds")

        i = tf.constant(0, dtype=tf.int64)
        arr_idx = tf.constant(0, dtype=tf.int64)
        ranks = tf.TensorArray(tf.int32,
                               size=1,
                               element_shape=(),
                               dynamic_size=True,
                               name="ranks")

        scores = preds * tf.subtract(1.0, tf.cast(auxes, tf.float32))

        # preds is tensor(readout)
        def cond(i, arr_idx, preds, scores, labels, ranks):
            return tf.less(i, batch_size)

        def loop_body(i, arr_idx, preds, scores, labels, ranks):
            # the i-th row
            # build once
            pred = preds[i]
            score = scores[i]
            label = labels[i][0]  #[0] for ragged extra axis

            print_op = tf.print("===\n", i, label)

            with tf.control_dependencies([print_op]):
                row_len = tf.cast(tf.shape(label)[0], tf.int64)

            def cond_inner(j, i, row_len, arr_idx, pred, score, label, ranks):
                return tf.less(j, row_len)

            def loop_body_inner(j, i, row_len, arr_idx, pred, score, label,
                                ranks):
                # build once
                l = label[j]
                # pick out cur position
                p = pred[l]
                op3 = tf.print("p shape: ", tf.shape(p))
                op4 = tf.print("p : ", p, "\n", "label: ", label, "l: ", l)
                # race?
                print_op = tf.print("score1", score)
                with tf.control_dependencies([print_op, op3, op4]):
                    indices = tf.cast(tf.broadcast_to(l, [1, 1]), tf.int32)
                    score_l = scatter_update_tensor(score, indices, [p])

                print_op = tf.print("score2", score_l)

                with tf.control_dependencies([print_op]):
                    sorted_idx = tf.argsort(score_l, direction="DESCENDING")
                    print_op = tf.print("sorted_idx", sorted_idx)

                with tf.control_dependencies([print_op]):
                    rank = tf.add(tf.argsort(sorted_idx)[l], 1)
                    print_rank = tf.print("==== rank ===: ", rank)

                with tf.control_dependencies([print_rank]):
                    ranks = ranks.write(tf.cast(arr_idx, tf.int32), rank)

                return (tf.add(j, 1), i, row_len, tf.add(arr_idx, 1), pred,
                        score, label, ranks)

            j = tf.constant(0, dtype=tf.int64)

            # 2)
            j, i, row_len, arr_idx, pred, score, label, ranks_inner = tf.while_loop(
                cond_inner,
                loop_body_inner,
                (j, i, row_len, arr_idx, pred, score, label, ranks),
                parallel_iterations=10)

            return (tf.add(i, 1), arr_idx, preds, scores, labels, ranks_inner)

        # 1)
        i, arr_idx, preds, scores, labels, ranks_out = tf.while_loop(
            cond,
            loop_body, [i, arr_idx, preds, scores, labels, ranks],
            parallel_iterations=10)

        # ranks -> hits 1/3/10
        ranks_tensor = ranks_out.stack()

        rank = tf.reduce_mean(tf.cast(ranks_tensor, tf.float32), name="rank")
        hits1 = tf.reduce_mean(tf.where(
            tf.less(ranks_tensor, 2),
            tf.ones_like(ranks_tensor, dtype=tf.float32),
            tf.zeros_like(ranks_tensor, dtype=tf.float32)),
                               name="hits1")
        hits3 = tf.reduce_mean(tf.where(
            tf.less(ranks_tensor, 4),
            tf.ones_like(ranks_tensor, dtype=tf.float32),
            tf.zeros_like(ranks_tensor, dtype=tf.float32)),
                               name="hits3")
        hits10 = tf.reduce_mean(tf.where(
            tf.less(ranks_tensor, 11),
            tf.ones_like(ranks_tensor, dtype=tf.float32),
            tf.zeros_like(ranks_tensor, dtype=tf.float32)),
                                name="hits10")

    return preds, scores, ranks_tensor, rank, hits1, hits3, hits10


def eval_op():
    set_gpu(3)
    # logits = tf.placeholder(tf.float32, shape=(None, 5))
    # labels = tf.ragged.placeholder(tf.float32, 5)
    # auxes = tf.placeholder(tf.float32, shape=(None, None))
    # preds = tf.nn.sigmoid(logits, name="pred")

    logits = tf.get_variable("logits",
                             initializer=tf.constant([
                                 [1., 2., 3., 4., 5.],
                                 [1., 2., 3., 4., 5.],
                                 [1., 2., 3., 4., 5.],
                             ]),
                             dtype=tf.float32)
    labels = tf.ragged.constant([[[1, 2]], [[3]], [[0, 2]]], dtype=tf.int64)
    # auxes = tf.ragged.constant([[1, 2, 3, 4], [0, 3], [0, 1, 2, 3]],
    #                            dtype=tf.int64)
    auxes = tf.constant([
        [
            0,
            1,
            1,
            1,
            1,
        ],
        [
            1,
            0,
            0,
            1,
            0,
        ],
        [
            1,
            1,
            1,
            1,
            0,
        ],
    ],
                        dtype=tf.int64)

    # label_auxes = tf.ragged.stack([labels, auxes], axis=1)
    preds, scores, ranks_tensor, rank, hits1, hits3, hits10 = eval_inner(
        logits, labels, auxes, batch_size=3)

    write_graph("eval")

    with tf.Session() as sess:
        print("===")
        # sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_nan", tfdbg.has_inf_or_nan)
        sess.run(tf.global_variables_initializer())
        # tf.global_variables_initializer().run()
        preds, scores, ranks, rank, hits1, hits3, hits10 = sess.run(
            [preds, scores, ranks_tensor, rank, hits1, hits3, hits10])
        print(preds, ranks, rank, hits1, hits3, hits10)
        print(scores)


def demo():
    set_gpu(3)
    iterator_trn, batch_data_trn = build_train_dataset(3)
    iterator_val, batch_data_val = build_dev_dataset("val", 4)
    iterator_test, batch_data_test = build_dev_dataset("test")

    write_graph("data")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_trn.initializer)
        sess.run(iterator_val.initializer)
        sess.run(iterator_test.initializer)

        now = time.time()
        print(batch_data_val, type(batch_data_val))

        for i in range(3):
            pass
            # batch_data = sess.run(batch_data_trn)
            # print(batch_data)
            # print(batch_data["label"].sum(axis=1))

            # print("---")
            # batch_data = sess.run(batch_data_val)
            # print(batch_data)
            # print(batch_data["aux_label"].sum(axis=1))
            # aux_label = batch_data["aux_label"]
            # label = batch_data[
            #     "label"]  # RaggedTensorValue, can't not index, to_list then index
            # label = label.to_list()
            # print("label[0]", label[0][0], "\n")
            # print("label[1]", label[1][0], "\n")
            # print("label[2]", label[2][0], "\n")
            # print("label[3]", label[3][0], "\n")

            print("---")
            label = batch_data_val["label"]
            aux_label = batch_data_val["aux_label"]

            print(sess.run(label))
            print(sess.run(label[0]))
            print(sess.run(label[0][0]))

            num_ent = batch_data_val["num_ent"][0][0]
            print("===> num_ent", num_ent)
            logits = tf.random_normal((4, N))
            preds, scores, ranks_tensor, rank, hits1, hits3, hits10 = eval_inner(
                logits, label, aux_label, batch_size=4)

            preds, scores, ranks, rank, hits1, hits3, hits10 = sess.run(
                [preds, scores, ranks_tensor, rank, hits1, hits3, hits10])
            print(preds, ranks, rank, hits1, hits3, hits10)
            print(scores)
            print(
                "\n",
                '*' * 20,
            )

    # print("---")
    # print(sess.run(batch_data_test))
        print("cost: ", time.time() - now)


def main():
    # eval_op()
    demo()


if __name__ == "__main__":
    main()
