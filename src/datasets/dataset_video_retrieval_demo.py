import torch
from torch.utils.data import Dataset
import random
import os
import json
from torch.utils.data.dataloader import default_collate
from src.utils.logger import LOGGER
from src.utils.basic_utils import flat_list_of_lists
from src.datasets.data_utils import mask_batch_text_tokens, img_collate
from src.datasets.dataloader import init_transform_dict, init_transform_dict_simple
import decord # video loader 
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge("torch")
import math
import torch.nn.functional as F
import numpy as np
import cv2
import lmdb
import glob
import src.utils.stop_words as stop_words
from PIL import Image
from src.datasets.sample_frames import SampleFrames

class HDVILAVideoRetrievalDataset(Dataset):
    """
    datalist
    """

    def __init__(self, cfg, vis_dir, anno_path, vis_format='video', mode="train"):
        assert vis_format in ["video", "frame"]
        self.cfg = cfg
        self.vis_dir = vis_dir
        self.anno_path = anno_path
        self.mode = mode
        self.vis_format = vis_format
        self.n_clips = cfg.train_n_clips if mode == "train" else cfg.test_n_clips
        self.num_frm = cfg.train_num_frms if mode == "train" else cfg.test_num_frms
        self.sample_rate = cfg.sample_rate
        if hasattr(cfg, "text_pos_num"):
            self.pos_num = cfg.pos_num
        else:
            self.pos_num = 1
        self.transform = init_transform_dict_simple(video_res=cfg.video_res,
                                             input_res=cfg.input_res)[mode]
        self.frame_sampler = SampleFrames(clip_len=self.num_frm, 
                                          frame_interval=self.sample_rate, 
                                          num_clips=self.n_clips, 
                                          temporal_jitter=True)
        self.init_dataset_process()


    def init_dataset_process(self):
        json_type = os.path.splitext(self.anno_path)[-1]
        assert json_type in ['.json', '.jsonl']

        if json_type == '.jsonl':
            data = []
            with open(self.anno_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = json.load(open(self.anno_path))
        self.datalist = data
        if self.cfg.is_demo:
            self.dir_list = os.listdir(self.vis_dir)

    def id2path(self, id):
        clip_name = id
        if self.vis_format == 'video':
            name = os.path.join(self.vis_dir, clip_name.split('/')[-1]+".mp4")
            if "lsmdc" in self.vis_dir:
                name = os.path.join(self.vis_dir, clip_name + ".avi")
        else:
            name = os.path.join(self.vis_dir, clip_name)
        return name

    def __len__(self):
        if self.cfg.is_demo:
            # data_len = len(self.dir_list)
            return len(self.dir_list)
        else:
            return len(self.datalist)

    def get_sample_idx(self, total_frame_num):
        """
        sample rate > 0: use SampleFrames, loop default
        sample rate = 0: uniform sampling, temporal jittering
        """
        if self.sample_rate > 0:
            results = {"total_frames": total_frame_num,
                    "start_index": 0}
            results = self.frame_sampler(results)
            return results["frame_inds"]
        elif self.sample_rate == 0:
            if hasattr(self.cfg, "sample_jitter") and self.cfg.sample_jitter and self.mode == "train":
                interval = int(total_frame_num / (self.n_clips*self.num_frm - 1))
                start = np.random.randint(0, interval+1)
                end = np.random.randint(total_frame_num-1-interval, total_frame_num)
                return np.linspace(start, end, self.n_clips*self.num_frm).astype(int)
            else:
                return np.linspace(0, total_frame_num-1, self.n_clips*self.num_frm).astype(int)

    def load_video(self, vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx) # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array

    def load_frames(self, vis_path, total_frame_num):
        frame_idx = self.get_sample_idx(total_frame_num)

        img_array = []
        for i in frame_idx:
            img = Image.open(os.path.join(vis_path, \
                    vis_path.split('/')[-1] + '_{0:03d}.jpg'.format(i))).convert("RGB")
            img_array.append(np.array(img))
        img_array = torch.from_numpy(np.array(img_array))  # (n_clips*num_frm, H, W, 3)

        img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array = self.transform(img_array)

        return img_array

    def __getitem__(self, index):
        if self.cfg.dummy_data:
            return dict(
            video = torch.randn(self.n_clips*self.num_frm, 3, self.cfg.input_res[0], self.cfg.input_res[1]),  # [clips, num_frm, C, H_crop, W_crop]
            texts = ["This is a dummy sentence, which contains nothing meaningful."]
        )

        if self.cfg.is_demo:
            # Get the list of all files and directories 
            video = self.dir_list[index]
            video_id, _ = os.path.splitext(video)
            vis_id = video_id
            texts = [self.cfg.query]  # for testing
            emotions = self.cfg.emotion
            
        else:
            if not ("video_id" in self.datalist[index].keys()):
                video = self.datalist[index]["video"]
                video_id, _ = os.path.splitext(video)
                vis_id = video_id
                texts = self.datalist[index]['caption']
            else:
                vis_id = self.datalist[index]['video_id']
                texts = self.datalist[index]['caption']
            
            if isinstance(texts, list):
                texts = random.sample(self.datalist[index]['caption'], self.pos_num)
                if 'didemo' in self.anno_path:
                    texts = [' '.join(self.datalist[index]['caption'])]
            else:
                texts = [texts]
            
        vis_path = self.id2path(vis_id)
        video = self.load_video(vis_path) if self.vis_format=='video' else self.load_frames(vis_path, self.datalist[index]['num_frame'])     
        if self.cfg.is_embed:
            return dict(
                video = video,  # [clips*num_frm, C, H_crop, W_crop]
                texts = texts,
                emotions = emotions,
                vis_id = vis_id
            )
        else:
            return dict(
                video = video,  # [clips*num_frm, C, H_crop, W_crop]
                texts = texts,
                vis_id = vis_id
            )


class VideoRetrievalCollator(object):
    def __init__(self, tokenizer, max_length=40, is_train=True, is_embed=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.is_embed = is_embed

    def collate_batch(self, batch):
        if isinstance(batch[0]["video"], torch.Tensor):
            v_collate = default_collate
        else:
            v_collate = img_collate
        video = v_collate([d["video"] for d in batch])

        text_examples = flat_list_of_lists([d["texts"] for d in batch])
        if self.is_embed:
            
            emotions = torch.LongTensor([(d["emotions"]) for d in batch])
        
        # for vis_id collation
        vid_collate = default_collate
        vis_id = vid_collate([d["vis_id"] for d in batch])

        text_str_list = [d for d in text_examples]  # (B, )

        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        if self.is_embed:
                
            collated_batch = dict(
                video=video,   # [B, clips, num_frm, C, H_crop, W_crop]
                text_input_ids=text_input_ids,
                text_input_mask=text_input_mask,
                emotions=emotions,
                vis_id=vis_id
            )
        else:
            collated_batch = dict(
                video=video,   # [B, clips, num_frm, C, H_crop, W_crop]
                text_input_ids=text_input_ids,
                text_input_mask=text_input_mask,
                vis_id=vis_id
            )

        return collated_batch
