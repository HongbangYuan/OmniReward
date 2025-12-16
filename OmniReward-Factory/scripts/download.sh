huggingface-cli download jinzhuoran/OmniRewardData \
  --repo-type dataset \
  --local-dir ./data/ \
  --local-dir-use-symlinks False \
  --include evalmuse.json hpdv2.json omni_t2i_hpd.json omni_t2i_oip.json omni_t2t_skywork.json omni_t2t_ultrafeedback.json omni_ti2t_rlaifv.json omni_ti2t_vlfeedback.json omnialign.json rlaifv.json skywork_reward.json videodpo.json visionreward.json

huggingface-cli download jinzhuoran/OmniRewardData \
  --repo-type dataset \
  --local-dir ./data/ \
  --local-dir-use-symlinks False \
  --include train_data/evalmuse.tar train_data/hpdv2.tar train_data/omni_t2i.tar train_data/omni_t2i_hpd.tar train_data/omni_ti2t.tar train_data/omnialign.tar train_data/rlaifv.tar train_data/videodpo_image.tar train_data/visionrewardv.tar

huggingface-cli download HongbangYuan/OmniRewardBench \
  --repo-type dataset \
  --local-dir ./data/ \
  --local-dir-use-symlinks False \
  --include omni_t2t.json omni_t2i.json omni_t2v.json omni_t2a.json omni_ti2t.json omni_tv2t.json omni_ta2t.json omni_t23d.json omni_ti2i.json

huggingface-cli download HongbangYuan/OmniRewardBench \
  --repo-type dataset \
  --local-dir ./data/ \
  --local-dir-use-symlinks False \
  --include media_data.zip

cd ./data/
unzip media_data.zip
cd ./train_data
for f in *.tar; do
  echo "Extracting $f ..."
  tar -xf "$f"
done