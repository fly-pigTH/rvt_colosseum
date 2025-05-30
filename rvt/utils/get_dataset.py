# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import sys
import shutil
import torch
import clip
from multiprocessing import Manager, Process
import yaml




from rvt.libs.peract.helpers.utils import extract_obs
from rvt.utils.rvt_utils import ForkedPdb
from rvt.utils.dataset import create_replay, fill_replay
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer

def normalize_task_weights(raw_scores: dict, total_weight: float = 20.0) -> dict:
    total_score = sum(raw_scores.values())
    if total_score == 0:
        raise ValueError("总倾向分数为0，无法归一化")
    return {
        task: (score / total_score) * total_weight
        for task, score in raw_scores.items()
    }

def get_dataset(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="transition_uniform",
):

    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
    )
    if not only_train:
        test_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
        )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    # tasks_rank = tasks[:10] if device==0 else tasks[10:]
    # print("device and task rank: ", device, tasks_rank)


    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        # processes = [
        # Process(
        #     target=fill_replay,
        #     args=(
        #         train_replay_buffer,
        #         task_parallel,
        #         train_replay_storage_folder,
        #         0,
        #         NUM_TRAIN,
        #         True,
        #         DEMO_AUGMENTATION_EVERY_N,
        #         CAMERAS,
        #         SCENE_BOUNDS,
        #         VOXEL_SIZES,
        #         ROTATION_RESOLUTION,
        #         False,
        #         data_path_train,
        #         EPISODE_FOLDER,
        #         VARIATION_DESCRIPTIONS_PKL,
        #         clip_model,
        #         device,
        #     ),
        # )
        # for task_parallel in tasks]
        # [t.start() for t in processes]
        # [t.join() for t in processes]


        # adaptive num_train

        with open("/home/d632/ColosseumChallenge/test/rvt_colosseum/rvt/configs/task_scores.yaml", "r") as f:
            raw_scores = yaml.safe_load(f)
        
        normalized_weights = normalize_task_weights(raw_scores, total_weight=20.0)



        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN*int(normalized_weights[task]),
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
        ) 

        if not only_train:
            # print("----- Test Buffer -----")
            fill_replay(
                replay=test_replay_buffer,
                task=task,
                task_replay_storage_folder=test_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_VAL,
                demo_augmentation=True,
                demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                device=device,
            )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="enumerate",
            num_workers=num_workers,
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset
