import numpy as np
import os.path as osp
import sys
import glob
from tqdm import tqdm
import json
from spot.data.loading.scene import SceneIO, SceneParser
from torch_geometric.data.dataset import Dataset as GeometricDataset
from torch.utils.data import IterableDataset
import torch
import math
import os
import multiprocessing

SHARED_MANAGER = multiprocessing.Manager()
SHARED_SCENE_IDX_COUNTER = SHARED_MANAGER.dict()
SHARED_OMITTED_SCENE_IDX_COUNTER = SHARED_MANAGER.dict()
SHARED_LOCK = SHARED_MANAGER.RLock()

# =========================================================================================
# =============================== Static Loaders ==========================================
# =========================================================================================

def load_dataset(data_root, data_source, object_class, dataloading_cfg, dataaug_cfg, split='train', load_baseline=False, dataset_type="default", batch_size=1, shuffle=True, drop_last=True):
    # assert split is appropriate
    if split not in ['train', 'val']:
        print('Split %s is not a valid option. Choose train or val.' % (split))
        exit()

    if object_class not in ['vehicle', 'car', 'pedestrian', 'bicycle', 'motorcycle', 'trailer', 'truck', 'bus']:
        print("Currently only support vehicles, cars, pedestrians, bicycles, motorcycles, trailers, and trucks.")
        sys.exit(1)

    # Prepare config inputs
    dataloading_cfg = dataloading_cfg.copy()
    training_aug_cfg = dataaug_cfg.copy()
    if split == "val":
        training_aug_cfg['train-augment'] = False
        training_aug_cfg['train-augment-bboxes'] = False

    # overwrite if in tracking mode
    if dataloading_cfg['dataset-properties']['tracking-mode']:
        tracking_seq_len = 12 if data_source == "nuscenes" else 1
        dataloading_cfg['sequence-properties']['sequence-length'] = tracking_seq_len
        dataloading_cfg['sequence-properties']['seqs-span-many-keyframes'] = False
        dataloading_cfg['sequence-properties']['allow-false-positive-frames'] = True
        dataloading_cfg['sequence-properties']['min-keyframes'] = 1

    # Load data
    scenes_objects, scenes_ignored_objects = load_data(data_root, data_source, split, object_class, load_baseline, dataloading_cfg, training_aug_cfg)

    # Build datasets with loaded data
    if dataset_type == "default":
        dataset = ObjectsIterableDatasetSampling(scenes_objects, split, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        dataset_ignoredobjs = None
    else:  # dataset_type == "track-iter"
        dataset = ObjectsIterableDatasetTracking(scenes_objects, split, omitted=False)
        dataset_ignoredobjs = ObjectsIterableDatasetTracking(scenes_ignored_objects, split, omitted=True)
    return dataset, dataset_ignoredobjs


def load_data(data_root, data_source, split, object_class, only_baseline, dataloading_cfg, training_aug_cfg):
    i = 0
    scenes_objects, scenes_ignored_objects = [], []
    for scene_file in tqdm(sorted(glob.glob(osp.join(data_root, split, "*.json")))):
        # if i > 5:
        #     break
        with open(scene_file, "rb") as f:
            scene_data = json.load(f)
        scene_io = SceneIO(scene_id=None)
        scene_io.load_from_dict(scene_data)
        scene_io.save_dir = os.path.join(data_root, split, scene_io.scene_id)
        scene_parser = SceneParser(scene_io, object_class, data_source, dataloading_cfg, only_baseline)
        scene_objects, scene_ignored_objects = scene_parser.parse(training_aug_cfg)
        scenes_objects.append(scene_objects)
        scenes_ignored_objects.append(scene_ignored_objects)
        i += 1

    num_seqs = sum([scene_objects.num_sequences for scene_objects in scenes_objects])
    if num_seqs == 0 and not only_baseline:
        print("Current configuration causes no samples!")
        print(data_root)
        print(dataloading_cfg)
        sys.exit(1)

    return scenes_objects, scenes_ignored_objects


# =========================================================================================
# =============================== Datasets Objects ========================================
# =========================================================================================

class ObjectsIterableDatasetBase(IterableDataset):

    def __init__(self, scenes, split, debug=False):

        '''
        - split             : "train", "test", or "val"
        - num_pts           : number of points to sample for input/output of TNOCS at each time step
        - always_use_first_step : if True, the step t=0 will always be included in the returned sequence
                            rather than by model.
        - data_source : either 'object_seed' or 'random_seed'. Determines the data format.
        '''
        super(ObjectsIterableDatasetBase, self).__init__()
        self.scenes = scenes
        self.split = split
        self.debug = debug
        self.seqs_per_scene = [scene_objects.num_sequences for scene_objects in self.scenes]
        self.num_seqs = sum(self.seqs_per_scene)

    def len(self):
        return self.num_seqs

    def __len__(self):
        return self.len()

    def get(self, idx):
        scene_idx, seq_idx = self._globalidx2sceneseqidxs(idx)

        # make random noise patterns deterministic for validation (important when we're using noisy GT crops)
        if self.split == "val": np.random.seed(int(idx))
        data = self.scenes[scene_idx].get(seq_idx)
        if self.split == "val": np.random.seed() # reset to random seed

        return data

    def __iter__(self):
        load_idxs, end_of_dataset = self._get_iteration_indices()
        if end_of_dataset:
            return iter([])
        worker_info = torch.utils.data.get_worker_info()

        # split across workers
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(load_idxs)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((len(load_idxs)) / float(worker_info.num_workers)))
            worker_id = worker_info.id

            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(load_idxs))

        # load sequence data
        this_worker_idxs = [load_idxs[i] for i in range(len(load_idxs)) if i in range(iter_start, iter_end)]
        data = []
        for idx in this_worker_idxs:
            data.append(self.get(idx))

        # increment the iteration for this worker's copy
        self.next()

        return iter(data)

    def _get_iteration_indices(self):
        raise NotImplementedError("Must be overwritten by a child class!")

    def _globalidx2sceneseqidxs(self, idx):
        scene_idx, seq_idx = 0, idx
        for i in range(len(self.seqs_per_scene)):
            if seq_idx >= self.seqs_per_scene[i]:
                seq_idx -= self.seqs_per_scene[i]
                scene_idx += 1
            else:
                break
        return scene_idx, seq_idx

    def next(self):
        raise NotImplementedError("Must be implemented by child class!")


class ObjectsIterableDatasetSampling(ObjectsIterableDatasetBase):

    def __init__(self, scenes, split, batch_size, shuffle=True, drop_last=True):
        super(ObjectsIterableDatasetSampling, self).__init__(scenes, split)
        self.cur_idx = 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if shuffle:
            self.batch_idxs = torch.randperm(self.num_seqs)
        else:
            self.batch_idxs = torch.arange(self.num_seqs)

        if self.drop_last:
            self.batch_idxs = self.batch_idxs[:(self.num_seqs // self.batch_size) * self.batch_size]

    def _get_iteration_indices(self):
        if self.cur_idx*self.batch_size > self.num_seqs:
            return None, True
        load_idxs = self.batch_idxs[self.cur_idx*self.batch_size: (self.cur_idx+1)*self.batch_size]
        return load_idxs, False

    def next(self):
        self.cur_idx += 1
        if self.cur_idx * self.batch_size > self.num_seqs:
            return "end-data"
        return "success"

    def reshuffle(self):
        self.batch_idxs = torch.randperm(self.num_seqs)
        if self.drop_last:
            self.batch_idxs = self.batch_idxs[:(self.num_seqs // self.batch_size) * self.batch_size]

    def reset(self):
        self.cur_idx = 0
        if self.shuffle:
            self.reshuffle()

class ObjectsIterableDatasetTracking(ObjectsIterableDatasetBase):

    def __init__(self, scenes, split, omitted=False):
        super(ObjectsIterableDatasetTracking, self).__init__(scenes, split)
        self.cur_scene_idx = 0
        self.cur_sample_idx = 0
        self.omitted = omitted

    def _get_iteration_indices(self):
        # update global state to inform program what Scene Idx and Sampled Idx we're on
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # multi-process data loading
            worker_id = worker_info.id
            SHARED_LOCK.acquire()
            if self.omitted:
                SHARED_OMITTED_SCENE_IDX_COUNTER[worker_id] = (self.cur_scene_idx, self.cur_sample_idx)
            else:
                SHARED_SCENE_IDX_COUNTER[worker_id] = (self.cur_scene_idx, self.cur_sample_idx)
            SHARED_LOCK.release()

        # get indices to load
        if self.cur_scene_idx > len(self.scenes):
            raise RuntimeError("Trying to access data with invalid scene idx!")
        load_idxs = self.get_sample_objectidxs(self.cur_scene_idx, self.cur_sample_idx)
        return load_idxs, False

    def next(self):
        self.cur_sample_idx += 1
        out_flag = "success"
        if self.cur_sample_idx >= self.num_samples(scene_idx=self.cur_scene_idx):
            self.cur_sample_idx = 0
            self.cur_scene_idx += 1
            out_flag = "scene-end"

        if self.cur_scene_idx >= len(self.scenes):
            out_flag = "dataset-end"

        return out_flag

    def reset(self):
        self.cur_scene_idx = 0
        self.cur_sample_idx = 0

    def set_to_scene(self, scene_idx):
        self.cur_sample_idx = scene_idx
        self.cur_sample_idx = 0

    def num_scenes(self):
        return len(self.scenes)

    def num_samples(self, scene_idx):
        return len(self.scenes[scene_idx].keyframe_ids)

    def get_sample_objectidxs(self, scene_idx, sample_idx):
        obj_idxs_in_scene = self.scenes[scene_idx].keyframeidx2seqidxs(sample_idx)
        scene_offset = sum(self.seqs_per_scene[:scene_idx])
        return [idx + scene_offset for idx in obj_idxs_in_scene]

    def get_sample_token(self, scene_idx, sample_idx):
        return self.scenes[scene_idx].keyframe_ids[sample_idx]

    def get_sample_timestep(self, scene_idx, sample_idx):
        return self.scenes[scene_idx].keyframe_timesteps[sample_idx]

    def get_scene_name(self, scene_idx):
        return self.scenes[scene_idx].scene_id


if __name__ == "__main__":

    dataloading_cfgf = '/home/ubuntu/Casper/configs/01-04-22-backbone-sweep/dataloading-001.json'
    dataaug_cfgf = '/home/ubuntu/Casper/configs/default/data-augmentation.json'
    data_root = "/mnt/fsx/scratch/colton.stearns/pi/nuscenes-processed-cp-withfp-fixed"
    with open(dataloading_cfgf, "r") as f:
        dataloading_cfg = json.load(f)
    with open(dataaug_cfgf, "r") as f:
        dataaug_cfg = json.load(f)
    split = 'train'
    num_pts = 256
    nuscenes_dataset, dataset_ignoredobjs = load_nusc_dataset(data_root=data_root, object_class='car', dataloading_cfg=dataloading_cfg,
                                            dataaug_cfg=dataaug_cfg, split='train', load_baseline=False, debug=False)
    print(len(nuscenes_dataset))
    print(len(nuscenes_dataset.scenes[0].fp_sequences))
    print(len(nuscenes_dataset.scenes[0].sequences))

    scene_idx, sample_idx = 1, 8
    filtering_obj_idxs = nuscenes_dataset.get_sample_objectidxs(scene_idx, sample_idx)
    print(filtering_obj_idxs)
    object_idxs = dataset_ignoredobjs.get_sample_objectidxs(scene_idx, sample_idx)
    print(object_idxs)
    objs = nuscenes_dataset[object_idxs]
    print(len(objs))
    for car in objs:
        print("==============")
        print(car.boxes.box)
        print("--")
        print(car.point2frameidx)
        print("--")
        print(car.labels.haslabel_mask)
        print("--")
        print(car.reference_frame.keyframe_mask)
