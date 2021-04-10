# SPM
BASEDIR=/mnt/wanyao/ghproj_d/fairseq/mbart
SPM=/usr/local/bin/spm_encode
MODEL=${BASEDIR}/mbart.cc25/sentence.bpe.model
DATA=${BASEDIR}/preprocessed
TRAIN=train
VALID=valid
TEST=test
SRC=en_XX
TGT=ja_XX
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} &


# Data Preprocess
python -m fairseq_cli.preprocess --source-lang en_XX --target-lang ja_XX --trainpref /mnt/wanyao/ghproj_d/fairseq/mbart/preprocessed/train.spm --validpref /mnt/wanyao/ghproj_d/fairseq/mbart/preprocessed/valid.spm --testpref /mnt/wanyao/ghproj_d/fairseq/mbart/preprocessed/test.spm --destdir /mnt/wanyao/ghproj_d/fairseq/mbart/postprocessed/en-ja --thresholdtgt 0  --thresholdsrc 0  --srcdict /mnt/wanyao/ghproj_d/fairseq/mbart/mbart.cc25/dict.txt --tgtdict /mnt/wanyao/ghproj_d/fairseq/mbart/mbart.cc25/dict.txt --workers 70


# Fine-tuning
python -m train /mnt/wanyao/ghproj_d/fairseq/mbart/postprocessed/en-ja  --encoder-normalize-before --decoder-normalize-before  --arch mbart_large --task translation_from_pretrained_bart  --source-lang en_XX --target-lang ja_XX --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --max-update 40000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 --max-tokens 768 --update-freq 2 --save-interval 1 --save-interval-updates 8000 --keep-interval-updates 10 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2 --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler  --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN --layernorm-embedding  --ddp-backend no_c10d --save-dir /mnt/wanyao/ghproj_d/fairseq/mbart/checkpoint --distributed-world-size 1

--restore-file /mnt/wanyao/ghproj_d/fairseq/mbart/mbart.cc25/model.pt