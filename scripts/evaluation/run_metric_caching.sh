SPLIT=trainval

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_trainval \
scene_filter.frame_interval=1 \
