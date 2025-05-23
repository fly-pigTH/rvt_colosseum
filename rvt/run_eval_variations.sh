export DISPLAY=:0.0

export eval_dir=eval_variations_3525mask2100_all_final
# export DEMO_PATH=/home/d632/ColosseumChallenge/train_dataset/setup_chess/
export eval_episodes=25


# ✅ 这里是你要处理的 DEMO_PATH 子路径（多个任务）
demo_paths=(
    # "/home/d632/ColosseumChallenge/test_dataset/basketball_in_hoop"
    #"/home/d632/ColosseumChallenge/test_dataset/close_box"
    #"/home/d632/ColosseumChallenge/test_dataset/close_laptop_lid"
    #"/home/d632/ColosseumChallenge/test_dataset/empty_dishwasher"
    # "/home/d632/ColosseumChallenge/test_dataset/get_ice_from_fridge"
    # "/home/d632/ColosseumChallenge/test_dataset/hockey"
    # "/home/d632/ColosseumChallenge/test_dataset/insert_onto_square_peg"
    # "/home/d632/ColosseumChallenge/test_dataset/meat_on_grill"
    # "/home/d632/ColosseumChallenge/test_dataset/move_hanger"
    # "/home/d632/ColosseumChallenge/test_dataset/open_drawer"
    # "/home/d632/ColosseumChallenge/test_dataset/place_wine_at_rack_location"
    # "/home/d632/ColosseumChallenge/test_dataset/put_money_in_safe"
    # "/home/d632/ColosseumChallenge/test_dataset/reach_and_drag"
    # "/home/d632/ColosseumChallenge/test_dataset/scoop_with_spatula"
    # "/home/d632/ColosseumChallenge/test_dataset/setup_chess"
     "/home/d632/ColosseumChallenge/test_dataset/slide_block_to_target"
    # "/home/d632/ColosseumChallenge/test_dataset/stack_cups"
    # "/home/d632/ColosseumChallenge/test_dataset/straighten_rope"
    # "/home/d632/ColosseumChallenge/test_dataset/turn_oven_on"
    # "/home/d632/ColosseumChallenge/test_dataset/wipe_desk"
)

## change data folder in train.py
## change tasks in RVT/rvt/utils/rvt_utils.py

# TODO: 修改为单GPU版本，删除GPU轮换
device=0
counter=0
# counter_inisde=0

for DEMO_PATH in "${demo_paths[@]}"; do
    # for task_list in $(ls data/test_variations_final/); do
    for task_list in $(ls "$DEMO_PATH"); do
        echo "Running evaluation on task: $task_list"
        CUDA_VISIBLE_DEVICES=$device xvfb-run -a python eval.py \
            --model-folder runs/rvt_bs_4_NW_16/  \
            --eval-datafolder $DEMO_PATH \
            --tasks $task_list \
            --eval-episodes $eval_episodes \
            --log-name test/$eval_dir \
            --device 0 \
            --headless \
            --model-name model_35_25_mask_new_2100.pth \
            # --save-video # & 去掉串行化
            # >> runs/rvt_bs_12_NW_12/eval/logs/rvt_${task_list}_final.txt &
        counter=$((counter+1))
    done
done