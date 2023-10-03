# Long-Term Photometric Consistent Novel View Synthesis with Diffusion Models
[Project Page](https://yorkucvil.github.io/Photoconsistent-NVS)

## Python environment
Please install the included environment in the root of this repo:
```
conda env create -f environment.yaml
```

## Directory structure
```
├── environment.yaml
├── dataset-data
│   ├── data
│   │   ├── test
│   │   │   ├── videos                       // videos for this split
│   │   │   └──  poses.npy                   // converted camera poses
│   │   ├── train
│   │   │   ├── videos                       // videos for this split
│   │   │   └──  poses.npy                   // converted camera poses
│   │   └── RealEstate10K-original           // original data from RealEstate10K dataset
│   │   │   ├── test                         // txt files for test camera poses
│   │   │   └── train                        // txt files for test camera poses
│   └──  extract-poses.py                    // Camera pose conversion script
├── instance-data                            // contains data from training and sampling
│   ├── checkpoints                          // Model checkpoints
│   ├── logs                                 // Tensorboard logs
│   └── taming-32-4-realestate-256.ckpt      // First stage VQGAN weights
└── src
    ├── datasets         // Data input pipeline
    ├── launch-scripts   // shell scripts for launching  slurm jobs
    ├── models
    ├── scripts          // python scripts for training and sampling
    └── utils
```

## Data preparation
[RealEstate10K](https://google.github.io/realestate10k) is a dataset consisting of real estate videos scraped from YouTube. Camera poses are recovered using SLAM.
Videos in the dataset are provided as YouTube URLs, and need to be downloaded manually using tools such as [yt-dlp](https://github.com/yt-dlp/yt-dlp).
The included data pipeline directly reads frames from the videos downloaded at 360p.
The camera poses provided by the dataset are provided using the camera extrinsics. We preprocess the camera poses into world transformations of a canonical camera, specifically the same camera and coordinate system as Blender.
Navigate to the `dataset-data` directory and place the downloaded Realestate files under `dataset-data/data/RealEstate10K-original`.
Please also populate the `dataset-data/data/test/videos` and `dataset-data/data/train/videos` directories with the downloaded videos.
To convert the poses run:
```
python extract-poses.py test
python extract-poses.py train
```

## Training
Training uses PyTorch DDP. An example slurm script is provided under `src/launch-scripts/train-deploy.sh`.

## Pretrained weights
RealEstate10K VQGAN weights: [Google Drive](https://drive.google.com/file/d/1SF8BWzKk9nggX1l0BiDJnYuxhX-g9dvd/view?usp=sharing)
RealEstate10K diffusion model weights: [Google Drive](https://drive.google.com/file/d/1bv_NF0UWH2DVYa93NUIxRvjUXipJSBQg/view?usp=sharing)
Please place the first stage VQGAN weights under `instance-data/taming-32-4-realestate-256.ckpt` and the diffusion model weights under `instance-data/checkpoints/2000-00000000.pth`.

## Sampling
Sampling requires a specific directory structure per sequence to specify the desired camera pose and the given source image.
The directory will also contain the generated samples, and any intermediate files generated for evaluation.
An example is provided under `instance-data/samples`:
```
└── instance-data
    └── samples
        └── 584f2fc6d686aebc         // directory for one sample
            ├── init-ims             // contains given source image
            │   └── 0000.png
            ├── samples              // contains sampled images, not created by sampling script
            └── sampling-spec.json   // specifies trajectory of poses
```
We also provide examples of our custom trajectories under `custom-trajectories`, the focal length should be adjusted differently for each sequence.
Sampling is performed by navigating to the `src` directory and running:
```
python scripts/sample-trajectory.py -o samples/584f2fc6d686aebc
```

## Thresholded Symmetric Epipolar Distance (TSED)
Coming soon!
