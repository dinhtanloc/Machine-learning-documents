import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# use tfds.load() or image_dataset_from_directory() to load images
ds, meta = tfds.load('citrus_leaves', with_info=True, split='train', shuffle_files=True)
ds = ds.batch(32)

# Take one batch from dataset and display the images
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5,5))

for sample in ds.take(1):
    images, labels = sample["image"], sample["label"]
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(images[i*3+j].numpy().astype("uint8"))
            ax[i][j].set_title(meta.features['label'].int2str(labels[i*3+j]))
plt.show()
