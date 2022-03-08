# full-AdaE2-no_dropout-reg-SCELoss_stable-a10_b1-wd1e2-bnfixed-bsz256 experiment justfile:88
# rank_5198.26474609375_loss_10.498673604516423_epoch_100_step_17000.data-00000-of-00001

LOCAL_IP := "10.38.10.136"
LOCAL_DIR := "/run/media/llouice/LLouice/dev/Rust/my/doing/Do/platform/AdaptiveE/py"
batch_size_trn := "256"

alias ps := push-save
alias dbg := default-dbg

default: fetch
    python save_model.py --wd 0.01 --alpha 10.0 --beta 1.0 --no_masked_label --ex full-AdaE2-no_dropout-reg-SCELoss_stable-a10_b1-wd1e2-bnfixed-bsz{{ batch_size_trn }} --model_name AdaE2 --visible_device_list 0 --batch_size_trn {{ batch_size_trn }} --batch_size_dev 4000 -e 125 -I 50 --lr 0.001 --use_pretrained \
    --ckpt rank_5198.26474609375_loss_10.498673604516423_epoch_100_step_17000.data-00000-of-00001

default-dbg: fetch
    python -m pdb save_model.py --wd 0.01 --alpha 10.0 --beta 1.0 --no_masked_label --ex full-AdaE2-no_dropout-reg-SCELoss_stable-a10_b1-wd1e2-bnfixed-bsz{{ batch_size_trn }} --model_name AdaE2 --visible_device_list 3 --batch_size_trn {{ batch_size_trn }} --batch_size_dev 4000 -e 125 -I 50 --lr 0.001 --use_pretrained \
    --ckpt rank_5198.26474609375_loss_10.498673604516423_epoch_100_step_17000.data-00000-of-00001


# experiment
ex:
    scp -r llouice@{{LOCAL_IP}}:{{LOCAL_DIR}}/save_model_simple.py save_model_simple.py
    python save_model_simple.py
# end experiment



# file transfer
fetch:
    scp -r llouice@{{LOCAL_IP}}:{{LOCAL_DIR}}/ada.py ada.py
    scp -r llouice@{{LOCAL_IP}}:{{LOCAL_DIR}}/save_model.py save_model.py
    scp -r llouice@{{LOCAL_IP}}:{{LOCAL_DIR}}/save.justfile save.justfile




push-save:
    just -f save.justfile push ada.py
    just -f save.justfile push save_model.py
    just -f save.justfile push save.justfile

push src target=".":
    scp -P 17877 -r {{src}} llouice@10.112.186.186:/home/llouice/dev/Rust/my/ada_python/{{target}}

# end file transfer
