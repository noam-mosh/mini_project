import torchvision

def remove_dumped_ids(item):
    key, value = item
    return 'dumped' not in value['file_name']


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, feature_extractor, train=True):
        
        super(CocoDetection, self).__init__(root=root, annFile=annFile)

        self.feature_extractor = feature_extractor
        self.coco.imgs = dict(filter(remove_dumped_ids, self.coco.imgs.items()))
        img_ids = [i for i in self.coco.getImgIds() if i not in self.coco.catToImgs[self.coco.getCatIds('other')[0]]]
        self.coco.imgs = {k: self.coco.imgs[k] for k in img_ids}

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

