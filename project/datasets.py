import torchvision

def remove_dumped_ids(item):
    key, value = item
    return 'dumped' not in value['file_name']


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = 'annotations_train.json' if train else 'annotations_test.json'
        # ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)

        self.feature_extractor = feature_extractor
        self.coco.imgs = dict(filter(remove_dumped_ids, self.coco.imgs.items()))
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

    def collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        # encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        encoding = self.feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {'pixel_values': encoding['pixel_values'], 'pixel_mask': encoding['pixel_mask'], 'labels': labels}
        return batch

