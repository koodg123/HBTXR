---
source_type: codebase
source_name: M3ViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/M3ViT/analysis.md -->

# M3ViT Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT`
- repo_remote: `origin	https://github.com/VITA-Group/M3ViT (fetch)`
- category: `multi-task MoE-ViT model/accelerator`
- HGTXR relevance: `high`
- matched_paper: `M3ViT`
- paper_title: M3ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/M3ViT.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `133` |
| 주요 언어 | `YAML:67, Python:53, JSON:3, .png:3, Markdown:2, .m:2, .txt:1, no_ext:1, .npy:1` |
| LOC 추정 | `Python:12949, YAML:3708, Markdown:163, .txt:86, JSON:24` |

### Directory Map
- `configs/` (4 entries)
- `configs/cityscapes/` (3 entries)
- `configs/nyud/` (4 entries)
- `configs/pascal/` (4 entries)
- `data/` (6 entries)
- `data/db_info/` (4 entries)
- `evaluation/` (10 entries)
- `evaluation/seism/` (3 entries)
- `losses/` (2 entries)
- `models/` (24 entries)
- `models/gate_funs/` (2 entries)
- `models/model_info/` (3 entries)
- `resources/` (3 entries)
- `train/` (1 entries)
- `utils/` (9 entries)

### Metadata / Config Refs
- `README.md`
- `evaluation/seism/README.md`
- `data/db_info/nyu_classes.json`
- `data/db_info/context_classes.json`
- `data/db_info/pascal_part.json`

### Core Source Refs
- `utils/common_config.py`: 838 lines
- `models/resnet.py`: 785 lines
- `models/vision_transformer_moe.py`: 601 lines
- `models/seg_hrnet.py`: 527 lines
- `data/pascal_context.py`: 505 lines
- `models/vit.py`: 503 lines
- `train_fastmoe.py`: 498 lines
- `evaluation/evaluate_utils.py`: 387 lines
- `utils/helpers.py`: 329 lines
- `data/custom_transforms.py`: 318 lines

### Hardware-Oriented Source Refs
- no obvious HLS/RTL source found in first scan

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/README.md`
- # [NeurIPS 2022] “M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design” [[Paper]](https://openreview.net/forum?id=cFOhdl1cyU-)
- <p align="center">
- </p>
- ### Abstract
- ### Task-dependent MoE ViT Design
- <p align="center">
- </p>
- ### Circuit-level Implementation
- <p align="center">
- </p>
- ## Installation
- Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/), the most important packages can be installed as:
- ```
- conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
- conda install imageio scikit-image     # Image operations
- conda install -c conda-forge opencv           # OpenCV
- conda install pyyaml easydict                   # Configurations
- conda install termcolor                         # Colorful print statements
- pip install easydict
- pip install mmcv

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: ViT backbone에 task-specific MoE layer를 넣어 task별 sparse expert pathway만 활성화한다.
- 알고리즘 축: task gate가 expert path를 선택하고 hardware reordering이 task switching overhead를 줄인다.
- 하드웨어 축: ZCU104 FPGA에서 sparse expert execution과 memory-efficient task switching을 co-design한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/decoder_head.py:177:                `mmseg/datasets/pipelines/formatting.py:Collect`.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/decoder_head.py:198:                `mmseg/datasets/pipelines/formatting.py:Collect`.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/train/train_utils.py:163:            #     images, F.softmax(logits_aug(prior_out) if p['data_aug']`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/train/train_utils.py:166:                images, F.softmax(logits_aug(prior_out) if p['data_aug']`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/train/train_utils.py:170:            #     images, F.softmax(logits_aug(prior_out) if p['data_aug'] else`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/train/train_utils.py:174:                images, F.softmax(logits_aug(prior_out) if p['data_aug'] else`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:159:    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:177:class Attention(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:195:        # print('for attention',x.shape)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:201:        attn = attn.softmax(dim=-1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:227:                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:230:        self.attn = Attention(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:350:                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:425:            elif isinstance(m, nn.LayerNorm):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/mti_net.py:87:        mask = F.softmax(shared.view(B, C//self.N, self.N, H, W), dim = 2) # Per task attention mask`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/padnet.py:59:        We apply an attention mask to features from other tasks and`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/padnet.py:66:        self.self_attention = {}`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/padnet.py:70:            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/padnet.py:71:        self.self_attention = nn.ModuleDict(self.self_attention)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/padnet.py:75:        adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/padnet.py:133:    def __init__(self, p, backbone, embed_dim=1024, img_size=768, patch_size = 16, norm_layer=partial(nn.LayerNorm, eps=1e-6),align_corners=False,norm_cfg = None):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit_up_head.py:78:                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/requirements.txt:7:https://repo.anaconda.com/pkgs/main/linux-ppc64le/ca-certificates-2020.12.8-h6ffa863_0.conda`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/requirements.txt:38:https://repo.anaconda.com/pkgs/main/linux-ppc64le/certifi-2020.12.5-py38h6ffa863_0.conda`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/requirements.txt:39:https://repo.anaconda.com/pkgs/main/linux-ppc64le/chardet-4.0.0-py38h6ffa863_1003.conda`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/requirements.txt:52:https://repo.anaconda.com/pkgs/main/linux-ppc64le/pysocks-1.7.1-py38h6ffa863_0.conda`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/requirements.txt:62:https://repo.anaconda.com/pkgs/main/linux-ppc64le/cffi-1.14.2-py38he30daa8_0.conda`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/evaluation/evaluate_utils.py:316:                    imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/evaluation/eval_semseg.py:22:                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vision_transformer_moe.py:56:        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/models/vit.py:55:        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/data/db_info/pascal_part.json:17:	"16": {"plant": 2 , "pot": 1},`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/data/pascal_context.py:60:                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/data/db_info/context_classes.json:1:{"accordion": 1, "aeroplane": 2, "air conditioner": 3, "antenna": 4, "artillery": 5, "ashtray": 6, "atrium": 7, "baby carriage": 8, "bag": 9, "ball": 10, "balloon": 11, "bamboo weaving": 12, "barrel": 13, "baseball bat": 14, "basket": 15, "basketball backboard": 16, "bathtub": 17, "bed": 18, "bedclothes": 19, "beer": 20, "bell": 21, "bench": 22, "bicycle": 23, "binoculars": 24, "bird": 25, "bird cage": 26, "bird feeder": 27, "bird nest": 28, "blackboard": 29, "board": 30, "boat": 31, "bone": 32, "book": 33, "bottle": 34, "bottle opener": 35, "bowl": 36, "box": 37, "bracelet": 38, "brick": 39, "bridge": 40, "broom": 41, "brush": 42, "bucket": 43, "building": 44, "bus": 45, "cabinet": 46, "cabinet door": 47, "cage": 48, "cake": 49, "calculator": 50, "calendar": 51, "camel": 52, "camera": 53, "camera lens": 54, "can": 55, "candle": 56, "candle holder": 57, "cap": 58, "car": 59, "card": 60, "cart": 61, "case": 62, "casette recorder": 63, "cash register": 64, "cat": 65, "cd": 66, "cd player": 67, "ceiling": 68, "cell phone": 69, "cello": 70, "chain": 71, "chair": 72, "chessboard": 73, "chicken": 74, "chopstick": 75, "clip": 76, "clippers": 77, "clock": 78, "closet": 79, "cloth": 80, "clothes tree": 81, "coffee": 82, "coffee machine": 83, "comb": 84, "computer": 85, "concrete": 86, "cone": 87, "container": 88, "control booth": 89, "controller": 90, "cooker": 91, "copying machine": 92, "coral": 93, "cork": 94, "corkscrew": 95, "counter": 96, "court": 97, "cow": 98, "crabstick": 99, "crane": 100, "crate": 101, "cross": 102, "crutch": 103, "cup": 104, "curtain": 105, "cushion": 106, "cutting board": 107, "dais": 108, "disc": 109, "disc case": 110, "dishwasher": 111, "dock": 112, "dog": 113, "dolphin": 114, "door": 115, "drainer": 116, "dray": 117, "drink dispenser": 118, "drinking machine": 119, "drop": 120, "drug": 121, "drum": 122, "drum kit": 123, "duck": 124, "dumbbell": 125, "earphone": 126, "earrings": 127, "egg": 128, "electric fan": 129, "electric iron": 130, "electric pot": 131, "electric saw": 132, "electronic keyboard": 133, "engine": 134, "envelope": 135, "equipment": 136, "escalator": 137, "exhibition booth": 138, "extinguisher": 139, "eyeglass": 140, "fan": 141, "faucet": 142, "fax machine": 143, "fence": 144, "ferris wheel": 145, "fire extinguisher": 146, "fire hydrant": 147, "fire place": 148, "fish": 149, "fish tank": 150, "fishbowl": 151, "fishing net": 152, "fishing pole": 153, "flag": 154, "flagstaff": 155, "flame": 156, "flashlight": 157, "floor": 158, "flower": 159, "fly": 160, "foam": 161, "food": 162, "footbridge": 163, "forceps": 164, "fork": 165, "forklift": 166, "fountain": 167, "fox": 168, "frame": 169, "fridge": 170, "frog": 171, "fruit": 172, "funnel": 173, "furnace": 174, "game controller": 175, "game machine": 176, "gas cylinder": 177, "gas hood": 178, "gas stove": 179, "gift box": 180, "glass": 181, "glass marble": 182, "globe": 183, "glove": 184, "goal": 185, "grandstand": 186, "grass": 187, "gravestone": 188, "ground": 189, "guardrail": 190, "guitar": 191, "gun": 192, "hammer": 193, "hand cart": 194, "handle": 195, "handrail": 196, "hanger": 197, "hard disk drive": 198, "hat": 199, "hay": 200, "headphone": 201, "heater": 202, "helicopter": 203, "helmet": 204, "holder": 205, "hook": 206, "horse": 207, "horse-drawn carriage": 208, "hot-air balloon": 209, "hydrovalve": 210, "ice": 211, "inflator pump": 212, "ipod": 213, "iron": 214, "ironing board": 215, "jar": 216, "kart": 217, "kettle": 218, "key": 219, "keyboard": 220, "kitchen range": 221, "kite": 222, "knife": 223, "knife block": 224, "ladder": 225, "ladder truck": 226, "ladle": 227, "laptop": 228, "leaves": 229, "lid": 230, "life buoy": 231, "light": 232, "light bulb": 233, "lighter": 234, "line": 235, "lion": 236, "lobster": 237, "lock": 238, "machine": 239, "mailbox": 240, "mannequin": 241, "map": 242, "mask": 243, "mat": 244, "match book": 245, "mattress": 246, "menu": 247, "metal": 248, "meter box": 249, "microphone": 250, "microwave": 251, "mirror": 252, "missile": 253, "model": 254, "money": 255, "monkey": 256, "mop": 257, "motorbike": 258, "mountain": 259, "mouse": 260, "mouse pad": 261, "musical instrument": 262, "napkin": 263, "net": 264, "newspaper": 265, "oar": 266, "ornament": 267, "outlet": 268, "oven": 269, "oxygen bottle": 270, "pack": 271, "pan": 272, "paper": 273, "paper box": 274, "paper cutter": 275, "parachute": 276, "parasol": 277, "parterre": 278, "patio": 279, "pelage": 280, "pen": 281, "pen container": 282, "pencil": 283, "person": 284, "photo": 285, "piano": 286, "picture": 287, "pig": 288, "pillar": 289, "pillow": 290, "pipe": 291, "pitcher": 292, "plant": 293, "plastic": 294, "plate": 295, "platform": 296, "player": 297, "playground": 298, "pliers": 299, "plume": 300, "poker": 301, "poker chip": 302, "pole": 303, "pool table": 304, "postcard": 305, "poster": 306, "pot": 307, "pottedplant": 308, "printer": 309, "projector": 310, "pumpkin": 311, "rabbit": 312, "racket": 313, "radiator": 314, "radio": 315, "rail": 316, "rake": 317, "ramp": 318, "range hood": 319, "receiver": 320, "recorder": 321, "recreational machines": 322, "remote control": 323, "road": 324, "robot": 325, "rock": 326, "rocket": 327, "rocking horse": 328, "rope": 329, "rug": 330, "ruler": 331, "runway": 332, "saddle": 333, "sand": 334, "saw": 335, "scale": 336, "scanner": 337, "scissors": 338, "scoop": 339, "screen": 340, "screwdriver": 341, "sculpture": 342, "scythe": 343, "sewer": 344, "sewing machine": 345, "shed": 346, "sheep": 347, "shell": 348, "shelves": 349, "shoe": 350, "shopping cart": 351, "shovel": 352, "sidecar": 353, "sidewalk": 354, "sign": 355, "signal light": 356, "sink": 357, "skateboard": 358, "ski": 359, "sky": 360, "sled": 361, "slippers": 362, "smoke": 363, "snail": 364, "snake": 365, "snow": 366, "snowmobiles": 367, "sofa": 368, "spanner": 369, "spatula": 370, "speaker": 371, "speed bump": 372, "spice container": 373, "spoon": 374, "sprayer": 375, "squirrel": 376, "stage": 377, "stair": 378, "stapler": 379, "stick": 380, "sticky note": 381, "stone": 382, "stool": 383, "stove": 384, "straw": 385, "stretcher": 386, "sun": 387, "sunglass": 388, "sunshade": 389, "surveillance camera": 390, "swan": 391, "sweeper": 392, "swim ring": 393, "swimming pool": 394, "swing": 395, "switch": 396, "table": 397, "tableware": 398, "tank": 399, "tap": 400, "tape": 401, "tarp": 402, "telephone": 403, "telephone booth": 404, "tent": 405, "tire": 406, "toaster": 407, "toilet": 408, "tong": 409, "tool": 410, "toothbrush": 411, "towel": 412, "toy": 413, "toy car": 414, "track": 415, "train": 416, "trampoline": 417, "trash bin": 418, "tray": 419, "tree": 420, "tricycle": 421, "tripod": 422, "trophy": 423, "truck": 424, "tube": 425, "turtle": 426, "tvmonitor": 427, "tweezers": 428, "typewriter": 429, "umbrella": 430, "unknown": 431, "vacuum cleaner": 432, "vending machine": 433, "video camera": 434, "video game console": 435, "video player": 436, "video tape": 437, "violin": 438, "wakeboard": 439, "wall": 440, "wallet": 441, "wardrobe": 442, "washing machine": 443, "watch": 444, "water": 445, "water dispenser": 446, "water pipe": 447, "water skate board": 448, "watermelon": 449, "whale": 450, "wharf": 451, "wheel": 452, "wheelchair": 453, "window": 454, "window blinds": 455, "wineglass": 456, "wire": 457, "wood": 458, "wool": 459}`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/M3ViT/data/custom_transforms.py:295:                sample[elem] = self.to_tensor(tmp.astype(np.uint8)) # Between 0 .. 255 so cast as uint8 to ensure compatible w/ imagenet weight`

## 5. HGTXR 적용 해석
- eye tracking의 search/track/head 분기와 직접 맞는다. SW 학습/정확도 실험 우선, HW는 static expert only.

### 적용 가능 모듈
- Search/Track mode-specific expert or head routing

## 6. 실험 옵션으로 변환할 때의 규칙
- 먼저 HGTXR-SW 또는 C++ reference에서 accuracy/bit-exact behavior를 검증한다.
- HLS에 반영할 때는 C3b signoff path를 덮어쓰지 말고 새로운 suffix variant로 추가한다.
- 완료된 `csynth.xml`, routed timing/power, PYNQ smoke JSON 없이는 resource matrix의 완료 variant로 승격하지 않는다.
- HGTXR 논문 범위를 벗어나는 구조 변경은 `paper_scope_review_required`로 둔다.

## 7. 리스크와 비적용 조건
- 동적 MoE routing, softmax-free attention, ternary/LUT-heavy compute는 정확도 또는 resource 정책과 충돌할 수 있다.
- 현재 사용자의 resource 방향은 DSP/URAM 활용 증가와 LUT 과사용 억제이므로 LUT-LLM/LUT-GEMM류는 기본값이 아니라 negative-control이다.
- codebase가 LLM 중심이면 HGTXR eye-tracking 적용은 kernel/system-flow 수준으로 제한한다.

## 8. 다음 분석/구현 액션
- 핵심 source file을 1개 선택해 line-by-line 분석을 수행한다.
- 해당 방법을 HGTXR `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml`의 planned experiment와 연결한다.
- SW accuracy metric과 HW resource metric을 같은 manifest에 기록한다.
