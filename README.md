# U-Net RoadSegmentation

Dataset: [SROADEX: Dataset for binary recognition and semantic segmentation of road surface areas from high resolution Aerial Orthoimages Covering Approximately 8,650 km2 of the Spanish Territory Tagged with Road Information](https://zenodo.org/records/5905850)

## Training Details

- Model: U-Net(Custom)
  I added BatchNormalization2d layers to the original U-Net model to speed up the training process. I also added same padding to the convolutional layers to keep the spatial dimensions the same.
  Images of 256x256 and normalized to [0, 1] range.

### Checkpoint

The checkpoint was trained for `EPOCHS=10` and `BATCH_SIZE=40` with `Adam` optimizer, `LR=3e-4` and `BinaryCrossentropy` loss function.
The checkpoint was only trained on 30% of the training data as indicated by `train_final, _ = random_split(train_dataset, [0.3, 0.7], generator=generator)`

[Checkpoint](https://icedrive.net/s/wAygwTzRWhbVTjzG6BjC5SuahijV)


### Hardware Used

- GPU: Nvidia RTX 3090
- RAM: 32GB
- CPU: i7-13700K
