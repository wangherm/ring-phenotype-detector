# Ring phenotype detector — P0/P1

颜色无关的细胞内部荧光环候选检测。当前阶段仅实现 **M0：morphology 分割 → 逐荧光通道几何候选 → 可视化 → 人工审核**。没有监督分类器、没有训练权重，也没有经人工真值验证的 precision/recall 或阳性率。

## 安装与真实运行

建议 Python 3.11/3.12，CPU 即可；P0/P1 不需要 GPU。Windows、Colab、AutoDL 都使用同一套 Python CLI。

```bash
git clone https://github.com/wangherm/ring-phenotype-detector.git
cd ring-phenotype-detector
python -m venv .venv
# Linux/AutoDL:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q

python -m ring_detector.cli audit --input /path/to/TEST.zip --output outputs/audit
python -m ring_detector.cli segment --manifest outputs/audit/manifest.csv --config configs/default.yaml --output outputs/segments
python -m ring_detector.cli propose --manifest outputs/audit/manifest.csv --segments outputs/segments --config configs/default.yaml --output outputs/proposals
python -m ring_detector.cli review-export --proposals outputs/proposals --output outputs/review
```

也可以使用 `predict --baseline-only` 代替 `propose`，这是同一套未训练的候选算法，不会加载或伪造 checkpoint。

```bash
python -m ring_detector.cli predict --baseline-only --manifest outputs/audit/manifest.csv --segments outputs/segments --config configs/default.yaml --output outputs/proposals
python -m ring_detector.cli review-check --annotations outputs/review/annotations.csv --cells outputs/proposals/cells.csv
```

## Colab / AutoDL

[打开 Colab notebook](https://colab.research.google.com/github/wangherm/ring-phenotype-detector/blob/main/notebooks/P0_P1_Colab.ipynb)。先运行环境检查与合成形状测试，再上传 `TEST.zip` 或填写 Google Drive 中的路径；notebook 不包含训练代码。

AutoDL 上把原始 zip 上传到 `/root/autodl-tmp/TEST.zip`，在安装环境后运行：

```bash
bash scripts/run_baseline.sh /root/autodl-tmp/TEST.zip /root/autodl-tmp/ring-work/run
```

脚本沿用 `Virtual_Cell` 的可追踪日志思路，结果与仓库代码分开。需要后台执行时可使用服务器已有的 `screen`/`tmux`。本仓库尚未实际在 Colab/AutoDL 远程机器上执行；已在本地真实 TEST 数据上执行。

## 输入规则与坐标

- 角色：`c1=fluorescence`, `c2=morphology`, `c1-2=display merge`。按通道标记两侧完整 stem 配对，合并图不是独立样本。
- `fluorescence_channels: [auto]` 只在荧光角色内选择动态范围及局部对比最大的通道，记录选择证据。可设 `[R]`、`[G]`、`[B]`、`[gray]` 或索引列表 `[0, 1]`。
- 不同荧光通道分别检测；`trigger_channels` 记录触发候选的通道，不混加信号。多通道预览显示第一个选定通道，所有通道原值保存在 NPZ。
- 保留 uint8/uint16；支持 `YX`, `YXS/YXC`, `SYX/CYX`。复杂 Z/T、多 TIFF series 清楚报错，禁止隐式投影。数据文件需先导出明确平面。
- 框采用 `x1,y1` 包含、`x2,y2` 不包含；mask、候选椭圆中心与原始画布坐标一致。裁剪额外留 12px，保存 crop 原点以便映射。
- 本次为 8-bit 显示图，强度仅作相对分析。自动选择到了 R 并不代表代码依赖红色。R/G/B/gray 伪彩色一致性测试不证明真实跨染料泛化。

## 输出与人工审核

- `audit/manifest.csv`、`audit.json`：配对、尺寸、dtype、完整原始 stem、日期/处理元数据、SHA256、merge 几何一致性检查；未知处理保留 unknown。
- `segments/masks/*.tif`：每 FOV uint16 标签图；`cells.csv`：框、质心、面积与 segmentation QC；`segmentation_summary.csv`：每视野区域数。
- `proposals/overlays/*.png`：青色 mask 边界=自动 QC 通过，橙色=需核对，紫色小椭圆=ring 候选。白字 cell ID。颜色只用于显示。
- `proposals/cells.csv`、`rings.csv`、`fov_qc.csv`：逐细胞候选与特征、逐视野 QC。`ring_score` 是启发式检索分数，`p_positive` 留空；所有可判读区域 `label_pred=uncertain`，边缘截断/大块合并暂为 invalid。**无候选不代表 negative，候选不代表 positive。**
- `crops/*.png`：左 morphology、中荧光强度、右 mask 与候选；`crop_masks/*.png` 和 `raw_crops/*.npz` 保留 mask、荧光各通道原值。
- `review/review.html`：离线打开即可标 `positive/negative/uncertain/invalid` 并下载 CSV。初始标签全空，原图 FOV/cell/框映射完整。保留整套 review 文件夹，避免裁剪路径断开。
- `review/annotations.csv`：每 FOV 平衡导出最多 6 个样本（候选、亮斑/无环、弱信号、QC），而非只挑高分；`all_cells_for_review.csv` 可用于全部细胞复核。

审核判断：明确内部亮边暗心的近闭环=positive；可判读但无环=negative；离焦/弱信号/可疑结构=uncertain；错分割=invalid。用 contact sheets 快速定位，再打开原分辨率 crop/overlay 判断；缩略图不能代替细节检查。

## 已知限制与下一阶段

传统 Canny 闭合边缘 + interior-seeded watershed 会漏掉离焦细胞，也可能合并接触细胞、过分割纹理细胞和把背景间隙当对象。QC pass 不等于人工确认分割正确。候选算法偏向可拟合椭圆、局部连续且亮边暗心的结构；弱环、严重开环、不规则环可能漏检。颗粒围成的圈或大暗腔周边信号可能是假阳性，需科研人员审核。

先复查多个日期、处理条件的分割与候选，并建立手工计数/定位的小型分割真值；然后标约 50–100 个明确阳性和相当阴性（可行性目标，不是硬门槛），复标部分样本。未来 P2 才按 FOV/日期分组划分并训练 DenseNet201；uncertain/invalid 不进入正负训练。P3 的阳性比例与条件推断必须等经过验证的标签/模型。本次不实现空的 train/model 模块。

真实运行报告 `P0_P1_RESULTS.md` 与人工审核包单独在本地交付。含实验日期、条件和逐视野数字的报告、原始数据及大批运行产物不提交到公开 GitHub。
