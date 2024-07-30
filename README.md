# Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives

This repository comprises leaderboards, dataset and paper lists of Video-Language Understanding. This provides supplementary information for our survey paper [Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives](https://arxiv.org/abs/2406.05615) published at ACL 2024 (Findings). If you found any error, please don't hesitate to open an issue or pull request.

If you are interested in our survey, please cite as
```
@article{nguyen2024video,
  title={Video-Language Understanding: A Survey from Model Architecture, Model Training, and Data Perspectives},
  author={Nguyen, Thong and Bin, Yi and Xiao, Junbin and Qu, Leigang and Li, Yicong and Wu, Jay Zhangjie and Nguyen, Cong-Duy and Ng, See-Kiong and Tuan, Luu Anh},
  journal={arXiv preprint arXiv:2406.05615},
  year={2024}
}
```

## Resources

- [Leaderboards](#leaderboards)
    - [Text-video retrieval](#text-video-retrieval)
    - [Video captioning](#video-captioning)
    - [Video question answering](#video-question-answering)
- [Datasets](#datasets)
- [Paper list](#paper-list)
    - [Surveys](#surveys)
    - [Model architecture perspective](#model-architecture-perspective)
        - [Pre-transformer](#pre-transformer)
        - [Shared Transformer](#shared-transformer)
        - [Stack Transformer](#stack-transformer)
        - [Dual Transformer](#dual-transformer)
        - [LLM-augmented](#llm-augmented)
    - [Model training perspective](#model-training-perspective)
        - [Pre-training](#pre-training)
        - [Fine-tuning](#fine-tuning)
    - [Data perspective](#data-perspective)
        - [Manual collection](#manual-collection)
        - [Data augmentation](#data-augmentation)
        - [Manual annotation](#manual-annotation)
        - [Automatic annotation](#automatic-generation)

****
### Leaderboards
#### Text-video retrieval  

|  **Methods**  | **Model architecture** |     **Video**     |    **Text**   | **R@1** | **R@5** | **R@10** |
|:-------------:|:----------------------:|:-----------------:|:-------------:|:-------:|:-------:|:--------:|
|    [[VSE-LSTM-Neurips2014]](https://arxiv.org/abs/1411.2539)   |         Pre-TF         | ConvNet/OxfordNet |   GloVe-LSTM  |   3.8   |   12.7  |   17.1   |
| [[C+LSTM+SA-FC7-arXiv2016]](https://arxiv.org/abs/1609.08124) |         Pre-TF         |        VGG        |   GloVe-LSTM  |   4.2   |   12.9  |   19.9   |
|    [[EITanque-arXiv2016]](https://arxiv.org/abs/1612.06950)   |         Pre-TF         |        VGG        | word2vec-LSTM |   4.7   |   16.6  |   24.1   |
|  [[SA-G+SA-FC7-arxiv2016]](https://arxiv.org/abs/1609.08124)  |         Pre-TF         |        VGG        |     GloVe     |   3.1   |   9.0   |   13.4   |
|     [[CT-SAN-CVPR2017]](https://arxiv.org/abs/1610.02947)    |         Pre-TF         |         RN        | word2vec-LSTM |   4.4   |   16.6  |   22.3   |
|    [[JSFusion-ECCV2018]](https://arxiv.org/abs/1808.02559)   |         Pre-TF         |         RN        |   GloVe-LSTM  |   10.2  |   31.2  |   43.2   |
|   [[All-in-one-arXiv2022]](https://arxiv.org/abs/2203.07303)  |        Shared TF       |        Linear        |       BT      |   37.9  |   68.1  |   77.1   |
|   [[VLM-ACL2021]](https://arxiv.org/abs/2105.09996)  |        Shared TF       |        S3D        |       BT      |   28.1  |   55.5  |   67.4   |
|   [[DeCEMBERT-NAACL2021]](https://aclanthology.org/2021.naacl-main.193/)  |        Shared TF       |        RN        |       BT      |   17.5  |   44.3  |   58.6   |
|   [[ActBERT-arXiv2020]](https://arxiv.org/abs/2011.07231)  |        Stacked TF       |        Faster-RCNN        |       BT      |   16.3  |   42.8  |   56.9   |
|   [[VIOLET-CVPR2023]](https://arxiv.org/abs/2209.01540)  |        Stacked TF       |        VS-TF        |       BT      |   37.2  |   64.8  |   75.8   |
|     [[VindLU-CVPR2023]](https://arxiv.org/abs/2212.05051)    |        Stacked TF       |        ViT        |       BT      |   48.8  |   72.4  |   82.2   |
|      [[HERO-EMNLP2020]](https://arxiv.org/abs/2005.00200)     |        Stacked TF        |    RN+SlowFast    |       BT      |   16.8  |   43.4  |   57.7   |
|     [[MV-GPT-arXiv2022]](https://arxiv.org/abs/2201.08264)    |        Stacked TF        |       ViViT       |       BT      |   37.3  |   65.5  |   75.1   |
|    [[CLIP2TV-ICLR2023]](https://arxiv.org/abs/2209.06430)   |         Dual TF        |        ViT        |   CLIP-text   |   32.4  |   58.2  |   68.6   |
|    [[CLIP-ViP-ICLR2023]](https://arxiv.org/abs/2209.06430)   |         Dual TF        |        ViT        |   CLIP-text   |   49.6  |   74.5  |   84.8   |
|   [[CLIP4Clip-arXiv2021]](https://arxiv.org/abs/2104.08860)   |         Dual TF        |        ViT        |   CLIP-text   |   44.5  |   71.4  |   81.6   |

#### Video captioning

| **Methods** | **Architecture** |   **Video**   | **BLEU-4** | **METEOR** | **CIDEr** |
|:-----------:|:----------------:|:-------------:|:----------:|:----------:|:----------:|
| [[TA-ICCV2015]](https://arxiv.org/abs/1502.08029)          |      Pre-TF      | Video: 3D-CNN |    36.5    |    25.7    | / |
| [[h-RNN-CVPR2016]](https://arxiv.org/abs/1510.07712)       |      Pre-TF      |   Video: VGG  |    36.8    |    25.9    | / |
| [[MFATT-arXiv2016]](https://arxiv.org/abs/1612.00234)       |      Pre-TF      | Video: RN+C3D |    39.1    |    26.7    | / |
| [[CAT-TM-arXiv2016]](https://arxiv.org/abs/1612.00234)      |      Pre-TF      | Video: RN+C3D |    36.6    |    25.6    | / |
| [[NFS-TM-arXiv2016]](https://arxiv.org/abs/1612.00234)      |      Pre-TF      | Video: RN+C3D |    37.0    |    25.9    | / |
| [[Fuse-TM-arXiv2016]](https://arxiv.org/abs/1612.00234)     |      Pre-TF      | Video: RN+C3D |    37.5    |    25.9    | / |
| [[MARN-CVPR2019]](https://arxiv.org/abs/1905.03966)     |      Pre-TF      | Video: RN |    /    |    /    | 46.8 |
| [[Res-ATT-WWW2019]](https://link.springer.com/article/10.1007%2Fs11280-018-0531-z)     |      Pre-TF      | Video: RN |    37.0    |    26.9    | 40.7 |
| [[DenseLSTM-ACMM2019]](https://dl.acm.org/doi/abs/10.1145/3343031.3350932)     |      Pre-TF      | Video: VGG |    38.1    |    27.2    | 42.8 |
| [[VIOLET-CVPR2023]](https://arxiv.org/abs/2209.01540)        |     Stacked TF     |     VS-TF     |    /   |    /  | 58.0 |
| [[LAVENDER-arXiv2023]](https://arxiv.org/abs/2305.13167)        |     Stacked TF     |     VS-TF     |     /   |    /  | 57.4 |
| [[VLAB-arXiv2023]](https://arxiv.org/abs/2305.13167)        |     Stacked TF     |     EVA-G     |    54.6    |    33.4    | 74.9 |
| [[UniVL-arXiv2020]](https://arxiv.org/abs/2002.06353)       |     Stacked TF     |      S3D      |    41.8    |    28.9    | 50.0 |
| [[MV-GPT-arXiv2022]](https://arxiv.org/abs/2201.08264)      |     Stacked TF     |     ViViT     |    48.9    |    38.7    | 60.0 |
| [[CLIP-DCD-PRCV2022]](https://arxiv.org/abs/2111.15162)    |     Stacked TF     |      ViT      |    48.2    |    30.9    | 64.8 |
| [[DeCEMBERT-NAACL2021]](https://aclanthology.org/2021.naacl-main.193/)   |     Stacked TF     |       RN      |    45.2    |    29.7    | 52.3 |
| [[mPLUG-2-ICML2023]](https://arxiv.org/abs/2302.00402)     |     Stacked TF     |      ViT      |    57.8    |    34.9    | 80.3 |

#### Video question answering

| **Methods** | **Architecture** |   **Video**  |  **Text**  | **MSRVTT** | **MSVD** |
|:-----------:|:----------------:|:------------:|:----------:|:----------:|:--------:|
| [[E-MN-ACMM2017]](https://dl.acm.org/doi/abs/10.1145/3123266.3123427)       |      Pre-TF      |   VGG + C3D   | GloVe-LSTM |    30.4    |     26.7    |
| [[QueST-AAAI2020]](https://ojs.aaai.org/index.php/AAAI/article/view/6766)       |      Pre-TF      |   RN + C3D   | GloVe-LSTM |    40.0    |     /    |
| [[HME-CVPR2019]](https://arxiv.org/abs/1904.04357)         |      Pre-TF      | RN/VGG + C3D |  GloVe-GRU |    34.6    |   36.1   |
| [[HGA-AAAI2020]](https://ojs.aaai.org/index.php/AAAI/article/view/6767)         |      Pre-TF      | RN/VGG + C3D |  GloVe-GRU |    33.0    |   33.7   |
| [[ST-VQA-IJCV2019]](https://link-springer-com.libproxy1.nus.edu.sg/article/10.1007/s11263-019-01189-x)      |      Pre-TF      |    RN+C3D    | GloVe-LSTM |    35.5    |   34.7   |
| [[PGAT-ACMMM2021]](https://dl.acm.org/doi/10.1145/3474085.3475193)        |      Pre-TF      |  Faster-RCNN | GloVe-LSTM |    38.1    |   39.0   |
| [[HCRN-CVPR2020]](https://arxiv.org/abs/2002.10698)        |      Pre-TF      |      RN      | GloVe-LSTM |    38.6    |   41.2   |
| [[All-in-one-arXiv2022]](https://arxiv.org/abs/2203.07303)  |     Shared TF    |      Linear     |     BT     |    44.3    |   47.9   |
| [[LAVENDER-arXiv2022]](https://arxiv.org/abs/2206.07160)    |     Stacked TF    |     VS-TF    |     BT     |    45.0    |   56.6   |
| [[DeCEMBERT-NAACL2021]](https://aclanthology.org/2021.naacl-main.193/)  |        Stacked TF       |        RN       |       BT      |   37.4 |   / |  
| [[VindLU-CVPR2023]](https://arxiv.org/abs/2212.05051)    |        Stacked TF       |  ViT |  BT  |  44.6  |  /  |
| [[VIOLET-CVPR2023]](https://arxiv.org/abs/2209.01540)      |     Stacked TF    |     VS-TF    |     BT     |    44.5    |   54.7   |
| [[ClipBERT-CVPR2021]](https://arxiv.org/abs/2102.06183)    |     Stacked TF    |   CLIP-text  |     BT     |    37.4    |     /    |
| [[VGT-ECCV2022]](https://arxiv.org/abs/2207.05342)         |      Dual TF     |  Faster-RCNN |     BT     |    39.7    |     /    |
| [[CoVGT-TPAMI2023]](https://arxiv.org/abs/2302.13668)       |      Dual TF     |  Faster-RCNN |     BT     |    40.0    |   /   |
| [[Video-ChatGPT-arXiv2023]](https://arxiv.org/abs/2306.05424)   |   LLM-Augmented  |     ViT    |   Vicuna   |    49.3    |   64.9   |
| [[LLaMA-Vid-arXiv2023]](https://arxiv.org/abs/2311.17043)   |   LLM-Augmented  |     EVA-G    |   Vicuna   |    58.9    |   70.0   |
****

### Datasets

|     **Dataset**    |          **Links**         | **Video source** | **Annotation** |     **Tasks**    | **#Videos/#Scenes** |
|:------------------:|:--------------------------:|:----------------:|:--------------:|:----------------:|:-------------------:|
| `MSVD`               | [[Paper]](https://aclanthology.org/P11-1020/), [[Dataset]](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) |     YouTube videos     |     Manual     | TVR, VC, VideoQA |         1.9K        |
| `MSRVTT`             | [[Paper]](https://ieeexplore.ieee.org/document/7780940), [[Dataset]](https://github.com/WingsBrokenAngel/MSR-VTT-DataCleaning) |     Web videos     |     Manual     | TVR, VC, VideoQA |         7.2K        |
| `ActivityNet`        | [[Paper]](https://ieeexplore.ieee.org/document/7298698), [[Dataset]](http://activity-net.org/) |     YouTube videos     |     Manual     | AL, TVR, VC, VMR |         5.8K        |
| `FIBER`              | [[Paper]](https://arxiv.org/abs/2104.04182), [[Dataset]](https://github.com/MichiganNLP/video-fill-in-the-blank) |     [[VaTeX]](https://arxiv.org/abs/1904.03493)     |     Manual     |    VC, VideoQA   |         28K         |
| `WildQA`             | [[Paper]](https://arxiv.org/abs/2209.06650), [[Dataset]](https://github.com/MichiganNLP/In-the-wild-QA) |     YouTube videos     |     Manual     |      VideoQA     |         0.4K        |
| `NExT-QA`            | [[Paper]](https://arxiv.org/abs/2105.08276), [[Dataset]](https://github.com/doc-doc/NExT-QA) | [[VidOR]](https://dl.acm.org/doi/10.1145/3323873.3325056) |     Manual     |      VideoQA     |         5.4K        |
| `CausalVid-QA`       | [[Paper]](https://arxiv.org/abs/2205.14895), [[Dataset]](https://github.com/bcmi/Causal-VidQA) | [[Kinetics-700]](https://arxiv.org/abs/1907.06987) |     Manual     |      VideoQA     |         26K         |
| `HowTo100M`          | [[Paper]](https://arxiv.org/abs/1906.03327), [[Dataset]](https://www.di.ens.fr/willow/research/howto100m/) |     YouTube videos     |      Auto      |        PT        |         1.2M        |
| `HD-VILA-100M`       | [[Paper]](https://arxiv.org/abs/2111.10337), [[Dataset]](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/README.md) |     YouTube videos     |      Auto      |        PT        |         3.3M        |
| `YT-Temporal-180M`   | [[Paper]](https://arxiv.org/abs/2106.02636), [[Dataset]](https://huggingface.co/datasets/HuggingFaceM4/yttemporal180m) |     YouTube videos     |      Auto      |        PT        |          6M         |
| `TGIF-QA`            | [[Paper]](https://arxiv.org/abs/1704.04497), [[Dataset]](https://github.com/YunseokJANG/tgif-qa) | Animated TGIFs |     Manual     |      VideoQA     |         71K         |
| `TGIF-QA-R`          | [[Paper]](https://dl.acm.org/doi/10.1145/3474085.3475193), [[Dataset]](https://github.com/PengLiang-cn/PGAT) | [[TGIF-QA]](https://arxiv.org/abs/1704.04497) |  Manual, Auto  |      VideoQA     |         71K         |
| `DiDeMo`             | [[Paper]](https://arxiv.org/abs/1708.01641), [[Dataset]](https://github.com/lisaanne/localizingmoments) | [[YFCC100M]](https://arxiv.org/abs/1503.01817) |     Manual     |        TVR       |         11K         |
| `YouCook2`           | [[Paper]](https://arxiv.org/abs/1805.02834), [[Dataset]](http://youcook2.eecs.umich.edu/) |     YouTube videos     |     Manual     |      TVR, VC     |          2K         |
| `HMDB-51`            | [[Paper]](https://ieeexplore.ieee.org/document/6126543), [[Dataset]](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) |     Web videos     |     Manual     |      TVR, AR     |         6.8K        |
| `Kinetics-400`       | [[Paper]](https://arxiv.org/abs/1705.06950), [[Dataset]](https://github.com/cvdfoundation/kinetics-dataset) |     YouTube videos     |     Manual     |        AR        |         306K        |
| `Kinetics-600`       | [[Paper]](https://arxiv.org/abs/1808.01340), [[Dataset]](https://github.com/cvdfoundation/kinetics-dataset) |     [[Kinetics-400]](https://arxiv.org/abs/1705.06950)     |     Manual     |      AR, VG      |         480K        |
| `Kinetics-700`       | [[Paper]](https://arxiv.org/abs/1907.06987), [[Dataset]](https://github.com/cvdfoundation/kinetics-dataset) |     [[Kinetrics-600]](https://arxiv.org/abs/1808.01340)     |     Manual     |        AR        |         650K        |
| `VaTeX`              | [[Paper]](https://arxiv.org/abs/1904.03493), [[Dataset]](https://eric-xw.github.io/vatex-website/about.html) | [[Kinetrics-600]](https://arxiv.org/abs/1808.01340) |     Manual     |      TVR, VC     |         41K         |
| `TVR`                | [[Paper]](https://arxiv.org/abs/2001.09099), [[Dataset]](https://github.com/jayleicn/TVRetrieval) | [[TVQA]](https://arxiv.org/abs/1809.01696) |     Manual     |        VMR       |         22K         |
| `How2R`              | [[Paper]](https://arxiv.org/abs/2005.00200), [[Dataset]](https://github.com/linjieli222/HERO) | [[HowTo100M]](https://arxiv.org/abs/1906.03327) |     Manual     |        VMR       |         22K         |
| `How2QA`             | [[Paper]](https://arxiv.org/abs/2005.00200), [[Dataset]](https://github.com/linjieli222/HERO) | [[HowTo100M]](https://arxiv.org/abs/1906.03327) |     Manual     |      VideoQA     |         22K         |
| `YouTube Highlights` | [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_51), [[Dataset]](https://github.com/TencentARC/UMT) |     YouTube videos     |     Manual     |        VMR       |         0.6K        |
| `TACoS`              | [[Paper]](https://aclanthology.org/Q13-1003/), [[Dataset]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus) | [[MPII Composites]](https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/151_P2A-01.pdf) |     Manual     |        VMR       |         0.1K        |
| `QVHighlights`       | [[Paper]](https://arxiv.org/abs/2107.09609), [[Dataset]](https://github.com/jayleicn/moment_detr) |     YouTube vlogs     |     Manual     |        VMR       |         10K         |
| `TVSum`              | [[Paper]](https://openaccess.thecvf.com/content_cvpr_2015/html/Song_TVSum_Summarizing_Web_2015_CVPR_paper.html), [[Dataset]](https://people.csail.mit.edu/yalesong/tvsum/) |     YouTube videos     |     Manual     |        VMR       |          50         |
| `ViTT`               | [[Paper]](https://arxiv.org/abs/2011.11760), [[Dataset]](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT) | [[YouTube-8M]](https://arxiv.org/abs/1609.08675) |     Manual     |        VMR       |         5.8K        |
| `VidChapters-7M`     | [[Paper]](https://arxiv.org/abs/2309.13952), [[Dataset]](https://github.com/antoyang/VidChapters) |     [[YT-Temporal-180M]](https://arxiv.org/abs/2106.02636)     |      Auto      |      VC, VMR     |         817K        |
| `VideoCC3M`          | [[Paper]](https://arxiv.org/abs/2204.00679), [[Dataset]](https://github.com/google-research-datasets/videoCC-data) |     Web videos     |      Auto      |        PT        |         6.3M        |
| `WebVid-10M`         | [[Paper]](https://arxiv.org/abs/2104.00650), [[Dataset]](https://github.com/m-bain/webvid) |     Web videos     |      Auto      |        PT        |        10.7M        |
| `COIN`               | [[Paper]](https://arxiv.org/abs/1903.02874), [[Dataset]](https://coin-dataset.github.io/) |     YouTube videos     |     Manual     |        AS        |         12K         |
| `CrossTask`          | [[Paper]](https://arxiv.org/abs/1903.08225), [[Dataset]](https://github.com/DmZhukov/CrossTask) |     YouTube videos     |     Manual     |        AR        |         4.7K        |
| `Alivol-10M`         | [[Paper]](https://arxiv.org/abs/2104.09411) |     E-commerce videos     |      Auto      |        PT        |         10M         |
| `LSMDC`              | [[Paper]](https://arxiv.org/abs/1501.02530), [[Dataset]](https://sites.google.com/site/describingmovies) |     British movies     |     Manual     |        TVR       |          72         |
| `EK-100`             | [[Paper]](https://arxiv.org/abs/2006.13256), [[Dataset]](https://epic-kitchens.github.io/2023) |      Manual      |     Manual     |      AR, AL      |          7K         |
| `SSV1`               | [[Paper]](https://arxiv.org/abs/1706.04261), [[Dataset]](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv1/README.md) |      Manual      |     Manual     |        AR        |         108K        |
| `SSV2`               | [[Paper]](https://arxiv.org/abs/1706.04261), [[Dataset]](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv2/README.md) |      Manual      |     Manual     |        AR        |         221K        |
| `Moments in Time`    | [[Paper]](https://arxiv.org/abs/1801.03150), [[Dataset]](http://moments.csail.mit.edu/) |     Web videos     |     Manual     |        AR        |          1M         |
| `InternVid`          | [[Paper]](https://arxiv.org/abs/2307.06942), [[Dataset]](https://huggingface.co/datasets/OpenGVLab/InternVid) |     YouTube videos     |      Auto      |        PT        |         7.1M        |
| `How2`               | [[Paper]](https://arxiv.org/abs/1811.00347), [[Dataset]](https://github.com/srvk/how2-dataset) |     YouTube videos     |      Auto      |        VC        |        13.2K        |
| `WTS70M`             | [[Paper]](https://arxiv.org/abs/2007.14937) |     YouTube videos     |      Auto      |        PT        |         70M         |
| `Charades`           | [[Paper]](https://arxiv.org/abs/1705.02101), [[Dataset]](https://prior.allenai.org/projects/charades) |      Manual      |     Manual     | AR, VMR, VideoQA |         10K         |

### Paper list
#### Survey

1. Survey: Transformer based video-language pre-training `arXiv 2021` [[Paper]](https://arxiv.org/abs/2109.09920) 
2. Self-supervised learning for videos: A survey `ACM Computing Survey 2022` [[Paper]](https://arxiv.org/abs/2207.00419) [[Code]](https://github.com/Maddy12/SSL4VideoSurvey)
3. Video question answering: Datasets, algorithms and challenges `EMNLP 2022` [[Paper]](https://arxiv.org/abs/2203.01225) [[Code]](https://github.com/vru-next/videoqa) 
4. Deep learning for video-text retrieval: a review `IJMIR 2023` [[Paper]](https://arxiv.org/abs/2302.12552) 
5. A review of deep learning for video captioning `arXiv 2023` [[Paper]](https://arxiv.org/abs/2304.11431) 
6. Video question answering: a survey of models and datasets `Mobile Networks and Applications` 2021 [[Paper]](https://dl.acm.org/doi/abs/10.1007/s11036-020-01730-0) 

#### Model architecture perspective
##### Pre-transformer 

1. Video question answering via attribute-augmented attention network learning `SIGIR 2017` [[Paper]](https://arxiv.org/abs/1707.06355) [[Code]]() 
2. Convolutional Two-Stream Network Fusion for Video Action Recognition `CVPR 2016` [[Paper]](https://arxiv.org/abs/1604.06573) [[Code]](https://github.com/feichtenhofer/twostreamfusion) 
3. Tensor-train recurrent neural networks for video classifcation `arXiv 2017` [[Paper]](https://arxiv.org/abs/1707.01786) [[Code]](https://github.com/Tuyki/TT_RNN) 
4. Two-stream rnn/cnn for action recognition in 3d videos `IROS 2017` [[Paper]](https://arxiv.org/abs/1703.09783) [[Code]]() 
5. Convnet architecture search for spatiotemporal feature learning `arXiv 2017` [[Paper]](https://arxiv.org/abs/1708.05038) [[Code]](https://github.com/farazahmeds/Classification-of-brain-tumor-using-Spatiotemporal-models)
6. A joint sequence fusion model for video question answering and retrieval `ECCV 2018` [[Paper]](https://arxiv.org/abs/1808.02559)  
7. Learning language-visual embedding for movie understanding with natural-language `arXiv 2016` [[Paper]](https://arxiv.org/abs/1609.08124) 
8. Unifying visual-semantic embeddings with multimodal neural language models `NeurIPS 2014` [[Paper]](https://arxiv.org/abs/1411.2539) 
9. Temporal tessellation for video annotation and summarization `arXiv 2016` [[Paper]](https://arxiv.org/abs/1612.06950) [[Code]](https://github.com/dot27/temporal-tessellation)
10. End-to-end concept word detection for video captioning, retrieval, and question answering `CVPR 2017` [[Paper]](https://arxiv.org/abs/1610.02947) 
11. Video captioning with multi-faceted attention `arXiv 2016` [[Paper]](https://arxiv.org/abs/1612.00234) 
12. Describing videos by exploiting temporal structure `ICCV 2015` [[Paper]](https://arxiv.org/abs/1502.08029) [[Code]](https://github.com/yaoli/arctic-capgen-vid)
13. Video paragraph captioning using hierarchical recurrent neural networks `CVPR 2016` [[Paper]](https://arxiv.org/abs/1510.07712)  
14. Localizing moments in video with natural language `ICCV 2017` [[Paper]](https://arxiv.org/abs/1708.01641) 
15. Video question answering via attribute-augmented attention network learning `SIGIR 2017` [[Paper]](https://arxiv.org/abs/1707.06355) 
16. Hierarchical boundary-aware neural encoder for video captioning `CVPR 2017` [[Paper]](https://arxiv.org/abs/1611.09312) 
17. Tall: Temporal activity localization via language query `ICCV 2017` [[Paper]](https://arxiv.org/abs/1705.02101) [[Code]](https://github.com/jiyanggao/TALL)
18. Leveraging video descriptions to learn video question answering `AAAI 2017` [[Paper]](https://arxiv.org/abs/1611.04021) 

##### Shared Transformer
1. VATT: Transformers for multimodal selfsupervised learning from raw video, audio and text `NeurIPS 2021` [[Paper]](https://arxiv.org/abs/2104.11178) [[Code]](https://github.com/google-research/google-research/tree/master/vatt)
2. Lavender: Unifying video-language understanding as masked language modeling `arXiv 2022` [[Paper]](https://arxiv.org/abs/2206.07160) [[Code]](https://github.com/microsoft/lavender)
3. All in one: Exploring unifed video-language pretraining `CVPR 2023` [[Paper]](https://arxiv.org/abs/2203.07303) [[Code]](https://github.com/showlab/all-in-one)
4. An empirical study of end-to-end video-language transformers with masked visual modeling `CVPR 2023` [[Paper]](https://arxiv.org/abs/2209.01540) [[Code]](https://github.com/tsujuifu/pytorch_empirical-mvm)
5. VIOLET : End-to-End Video-Language Transformers with Masked Visual-token Modeling `arXiv 2021` [[Paper]](https://arxiv.org/abs/2111.12681) [[Code]](https://github.com/tsujuifu/pytorch_violet)
6. Vindlu: A recipe for effective video-and-language pretraining `CVPR 2023` [[Paper]](https://arxiv.org/abs/2212.05051) [[Code]](https://github.com/klauscc/vindlu)
7. Less is more: Clipbert for video-and-language learning via sparse sampling `CVPR 2021` [[Paper]](https://arxiv.org/abs/2102.06183) [[Code]](https://github.com/jayleicn/ClipBERT)

##### Stack Transformer
1. HERO: Hierarchical encoder for video+ language omni-representation pretraining `EMNLP 2020` [[Paper]](https://arxiv.org/abs/2005.00200) [[Code]](https://github.com/linjieli222/HERO)
2. End-to-end generative pretraining for multimodal video captioning `arXiv 2022` [[Paper]](https://arxiv.org/abs/2201.08264)
3. VLAB: Enhancing video language pre-training by feature adapting and blending `arXiv 2023` [[Paper]](https://arxiv.org/abs/2305.13167)
4. UniVL: A unifed video and language pre-training model for multimodal understanding and generation `arXiv 2020` [[Paper]](https://arxiv.org/abs/2002.06353) [[Code]](https://github.com/microsoft/UniVL)
5. CLIP meets video captioning: Concept-aware representation learning does matter `PRCV 2022` [[Paper]](https://arxiv.org/abs/2111.15162) [[Code]](https://github.com/yangbang18/CLIP-Captioner)
6. mPLUG-2: A modularized multimodal foundation model across text, image and video `ICML 2023` [[Paper]](https://arxiv.org/abs/2302.00402) [[Code]](https://github.com/alibaba/AliceMind)

##### Dual Transformer
1. CLIP-ViP: Adapting pre-trained image-text model to videolanguage representation alignment `ICLR 2023` [[Paper]](https://arxiv.org/abs/2209.06430) [[Code]](https://github.com/microsoft/xpretrain)
2. CLIP4Clip: An empirical study of clip for end to end video clip retrieval and captioning `arXiv 2021` [[Paper]](https://arxiv.org/abs/2104.08860) [[Code]](https://github.com/ArrowLuo/CLIP4Clip)
3. Video graph transformer for video question answering `ECCV 2022` [[Paper]](https://arxiv.org/abs/2207.05342) [[Code]](https://github.com/sail-sg/vgt)
4. Contrastive video question answering via video graph transformer `TPAMI 2023` [[Paper]](https://arxiv.org/abs/2302.13668) [[Code]](https://github.com/doc-doc/covgt)
5. Frozen in time: A joint video and image encoder for end-to-end retrieval `ICCV 2021` [[Paper]](https://arxiv.org/abs/2104.00650) [[Code]](https://github.com/m-bain/frozen-in-time)
6. A CLIP-Hitchhiker’s guide to long video retrieval `arXiv 2022` [[Paper]](https://arxiv.org/abs/2205.08508) [[Code]](https://github.com/m-bain/clip-hitchhiker)
7. ECLIPSE: Efficient long-range video retrieval using sight and sound `ECCV 2022` [[Paper]](https://arxiv.org/abs/2204.02874) [[Code]](https://github.com/GenjiB/ECLIPSE)
8. VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding `EMNLP 2021` [[Paper]](https://arxiv.org/abs/2109.14084) [[Code]](https://github.com/facebookresearch/fairseq/blob/main/examples/MMPT/README.md)

##### LLM-augmented
1. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding `EMNLP 2023` [[Paper]](https://arxiv.org/abs/2306.02858) [[Code]](https://github.com/damo-nlp-sg/video-llama)
2. VideoChat: Chat-centric video understanding `arXiv 2023` [[Paper]](https://arxiv.org/abs/2305.06355) [[Code]](https://github.com/opengvlab/ask-anything)
3. VideoLLM: Modeling video sequence with large language models `arXiv 2023` [[Paper]](https://arxiv.org/abs/2305.13292) [[Code]](https://github.com/cg1177/videollm)
4. LlaMA-VID: An image is worth 2 tokens in large language models `arXiv 2023` [[Paper]](https://arxiv.org/abs/2311.17043) [[Code]](https://github.com/dvlab-research/llama-vid)
5. Retrieving-to-Answer: Zero-Shot Video Question Answering with Frozen Large Language Models `arXiv 2023` [[Paper]](https://arxiv.org/abs/2306.11732)

#### Model training perspective
##### Pre-training
1. CLIP2TV: Align, match and distill for video-text retrieval `arXiv 2021` [[Paper]](https://arxiv.org/abs/2111.05610)
2. Understanding chinese video and language via contrastive multimodal pre-training `arXiv 2021` [[Paper]](https://arxiv.org/abs/2104.09411)
3. DeCEMBERT: Learning from noisy instructional videos via dense captions and entropy minimization `NAACL 2021` [[Paper]](https://aclanthology.org/2021.naacl-main.193/) [[Code]](https://github.com/zinengtang/decembert)
4. VideoBERT: A joint model for video and language representation learning `ICCV 2019` [[Paper]](https://arxiv.org/abs/1904.01766) [[Code]](https://github.com/ammesatyajit/VideoBERT)
5. Learning video representations using contrastive bidirectional transformer `arXiv 2019` [[Paper]](https://arxiv.org/abs/1906.05743)
6. MERLOT: Multimodal neural script knowledge models `NeurIPS 2021` [[Paper]](https://arxiv.org/abs/2106.02636) [[Code]](https://github.com/rowanz/merlot)
7. Revealing single frame bias for video-and-language learning `arXiv 2022` [[Paper]](https://arxiv.org/abs/2206.03428) [[Code]](https://github.com/jayleicn/singularity)
8. ActBERT: Learning Global-Local Video-Text Representations `CVPR 2020` [[Paper]](https://arxiv.org/abs/2011.07231) [[Code]](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/en/model_zoo/multimodal/actbert.md)

##### Fine-tuning
1. Multilevel language and vision integration for text-to-clip retrieval `AAAI 2019` [[Paper]](https://arxiv.org/abs/1804.05113) [[Code]](https://github.com/VisionLearningGroup/Text-to-Clip_Retrieval)
2. ST-Adapter: Parameter-efficient image-to-video transfer learning `NeurIPS 2022` [[Paper]](https://arxiv.org/abs/2206.13559) [[Code]](https://github.com/linziyi96/st-adapter)
3. Zero-shot video question answering via frozen bidirectional language models `NeurIPS 2022` [[Paper]](https://arxiv.org/abs/2206.08155) [[Code]](https://github.com/antoyang/FrozenBiLM)
4. Attentive Moment Retrieval in Videos `SIGIR 2018` [[Paper]](https://dl.acm.org/doi/10.1145/3209978.3210003)
5. To Find Where You Talk: Temporal Sentence Localization in Video with Attention Based Location Regression `arXiv 2018` [[Paper]](https://arxiv.org/abs/1804.07014) 
6. Cross-Modal Adapter for Text-Video Retrieval `arXiv 2022` [[Paper]](https://arxiv.org/abs/2211.09623) [[Code]](https://github.com/leaplabthu/cross-modal-adapter)
7. AIM: Adapting Image Models for Efficient Video Action Recognition `ICLR 2023` [[Paper]](https://arxiv.org/abs/2302.03024) [[Code]](https://github.com/taoyang1122/adapt-image-models)
8. Prompting Visual-Language Models for Efficient Video Understanding `ECCV 2022` [[Paper]](https://arxiv.org/abs/2112.04478) [[Code]](https://github.com/ju-chen/Efficient-Prompt)
9. Multi-modal Circulant Fusion for Video-to-Language and Backward `IJCAI 2018` [[Paper]](https://dl.acm.org/doi/10.5555/3304415.3304561) 
10. Long-term temporal convolutions for action recognition `arXiv 2016` [[Paper]](https://arxiv.org/abs/1604.04494) [[Code]](https://github.com/gulvarol/ltc)

#### Data perspective
##### Manual collection
1. Advancing high-resolution video-language representation with large-scale video transcriptions `CVPR 2022` [[Paper]](https://arxiv.org/abs/2111.10337) [[Code]](https://github.com/microsoft/xpretrain)
2. Howto100M: Learning a text-video embedding by watching hundred million narrated video clips `ICCV 2019` [[Paper]](https://arxiv.org/abs/1906.03327) [[Code]](https://github.com/antoine77340/MIL-NCE_HowTo100M)
3. FIBER: Fill-in-the-blanks as a challenging video understanding evaluation framework `ACL 2022` [[Paper]](https://arxiv.org/abs/2104.04182) [[Code]](https://github.com/MichiganNLP/video-fill-in-the-blank)
4. NExT-QA: Next phase of questionanswering to explaining temporal actions `CVPR 2021` [[Paper]](https://arxiv.org/abs/2105.08276) [[Code]](https://github.com/doc-doc/NExT-QA)
5. The "Something Something" Video Database for Learning and Evaluating Visual Common Sense `arXiv 2017` [[Paper]](https://arxiv.org/abs/1706.04261) [[Code]](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv1/README.md)
6. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100 `IJCV 2020` [[Paper]](https://arxiv.org/abs/2006.13256) [[Code]](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
7. From representation to reasoning: Towards both evidence and commonsense reasoning for video question answering `CVPR 2022` [[Paper]](https://arxiv.org/abs/2205.14895) [[Code]](https://github.com/bcmi/causal-vidqa)
8. Grounding Action Descriptions in Videos `TACL 2013` [[Paper]](https://aclanthology.org/Q13-1003/) 
9. Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments `CVPR 2018` [[Paper]](https://arxiv.org/abs/1711.07280) [[Code]](https://github.com/peteanderson80/Matterport3DSimulator)
10. Multimodal Pretraining for Dense Video Captioning `AACL-IJCNLP 2020` [[Paper]](https://arxiv.org/abs/2011.11760) [[Code]](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT)
11. QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries `NeurIPS 2021` [[Paper]](https://arxiv.org/abs/2107.09609) [[Code]](https://github.com/jayleicn/moment_detr)
12. VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research `ICCV 2019` [[Paper]](https://arxiv.org/abs/1904.03493) [[Code]](https://github.com/eric-xw/Video-guided-Machine-Translation)

##### Data augmentation
1. SVFormer: Semisupervised video transformer for action recognition `CVPR 2023` [[Paper]](https://arxiv.org/abs/2211.13222) [[Code]](https://github.com/chenhsing/svformer)
2. Semi-supervised video paragraph grounding with contrastive encoder `CVPR 2022` [[Paper]](https://ieeexplore.ieee.org/document/9879558) 
3. Learning temporal action proposals with fewer labels `arXiv 2019` [[Paper]](https://arxiv.org/abs/1910.01286) 
4. Self-supervised learning for semi-supervised temporal action proposal `CVPR 2021` [[Paper]](https://arxiv.org/abs/2104.03214) [[Code]](https://github.com/wangxiang1230/SSTAP)
5. Semi-Supervised Action Recognition with Temporal Contrastive Learning `CVPR 2021` [[Paper]](https://arxiv.org/abs/2102.02751) [[Code]](https://github.com/CVIR/TCL)
6. Learning Action Proposals With Fewer Labels `arXiv 2019` [[Paper]](https://arxiv.org/abs/1910.01286) 

##### Manual annotation
1. Collecting Highly Parallel Data for Paraphrase Evaluation `ACL 2011` [[Paper]](https://aclanthology.org/P11-1020/) [[Code]](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
2. MSR-VTT: A Large Video Description Dataset for Bridging Video and Language `CVPR 2016` [[Paper]](https://ieeexplore.ieee.org/document/7780940) [[Code]](https://github.com/WingsBrokenAngel/MSR-VTT-DataCleaning)
3. TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering `CVPR 2017` [[Paper]](https://arxiv.org/abs/1704.04497) [[Code]](https://github.com/YunseokJANG/tgif-qa)
4. Weakly-Supervised Video Object Grounding from Text by Loss Weighting and Object Interaction `arXiv 2018`  [[Paper]](https://arxiv.org/abs/1805.02834) [[Code]](http://youcook2.eecs.umich.edu/)
5. HMDB: A large video database for human motion recognition `ICCV 2011` [[Paper]](https://ieeexplore.ieee.org/document/6126543) [[Code]](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
6. The Kinetics Human Action Video Dataset `arXiv 2017` [[Paper]](https://arxiv.org/abs/1705.06950) [[Code]](https://github.com/cvdfoundation/kinetics-dataset)
7. TVSum: Summarizing Web Videos Using Titles `CVPR 2015` [[Paper]](https://ieeexplore.ieee.org/document/7299154) [[Code]](https://people.csail.mit.edu/yalesong/tvsum/)
8. TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval `ECCV 2020` [[Paper]](https://arxiv.org/abs/2001.09099) [[Code]](https://github.com/jayleicn/TVRetrieval)
9. COIN: A Large-scale Dataset for Comprehensive Instructional Video Analysis `CVPR 2019` [[Paper]](https://arxiv.org/abs/1903.02874) [[Code]](https://coin-dataset.github.io/)
10. Cross-task weakly supervised learning from instructional videos `CVPR 2019` [[Paper]](https://arxiv.org/abs/1903.08225) [[Code]](https://github.com/DmZhukov/CrossTask)
11. Moments in Time Dataset: one million videos for event understanding `CVPR 2019` [[Paper]](https://arxiv.org/abs/1801.03150) [[Code]](https://github.com/zhoubolei/moments_models/tree/v2)

##### Automatic generation
1. Progressive Graph Attention Network for Video Question Answering `ACMMM 2021` [[Paper]](https://dl.acm.org/doi/10.1145/3474085.3475193) [[Code]](https://github.com/PengLiang-cn/PGAT)
2. The StreetLearn Environment and Dataset `arXiv 2019` [[Paper]](https://arxiv.org/abs/1903.01292) [[Code]](https://github.com/deepmind/streetlearn)
3. VidChapters-7M: Video Chapters at Scale `NeurIPS 2023` [[Paper]](https://arxiv.org/abs/2309.13952) [[Code]](https://github.com/antoyang/VidChapters)
4. Learning Audio-Video Modalities from Image Captions `arXiv 2022` [[Paper]](https://arxiv.org/abs/2204.00679)
5. InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation `arXiv 2022` [[Paper]](https://arxiv.org/abs/2307.06942) [[Code]](https://github.com/opengvlab/internvideo)
6. How2: A Large-scale Dataset for Multimodal Language Understanding `NeurIPS 2018` [[Paper]](https://arxiv.org/abs/1811.00347) [[Code]](https://github.com/srvk/how2-dataset)
7. Learning Video Representations from Textual Web Supervision `arXiv 2020` [[Paper]](https://arxiv.org/abs/2007.14937)
