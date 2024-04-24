python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=urban_driver_agent \
experiment_name=training_urban_driver \
scene_filter=all_scenes \
cache_path=$NAVSIM_EXP_ROOT/mini_cache \
trainer.params.max_epochs=500 \
split=mini \
