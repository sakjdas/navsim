TEAM_NAME="SmallFish"
AUTHORS="Hao"
EMAIL="hao.guo@tum.de"
INSTITUTION="TUM"
COUNTRY="Germany"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
agent=urban_driver_agent \
agent.checkpoint_path=/home/hguo/e2eAD/endtoenddriving/train/Hao_Guo/best_model/model_FC_preprocessing.ckpt \
split=competition_test \
experiment_name=submission_urban_driver_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
