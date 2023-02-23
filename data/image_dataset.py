from data.base_dataset import BaseDataset
from PIL import Image
import blobfile as bf
import os
import numpy as np

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(BaseDataset):
    def __init__(self, opt, manager):
        super().__init__(opt, manager)
        self.image_dir = os.path.join(
            opt.data_dir, opt.dataset_dir
        )
        self.image_paths = _list_image_files_recursively(self.image_dir)

        classes = None
        if opt.model.class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in self.image_paths]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        shard = manager.get_rank()
        num_shards = manager.get_n_gpus()

        self.resolution = opt.model.image_size
        self.local_images = self.image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict