# Ring phenotype detector — P0/P1 v0.2

颜色无关的细胞内部荧光环候选检测。v0.2 使用 **明场预训练酵母分割 + 荧光全视野独立候选检测 → 坐标关联 → 人工审核**。Cellpose 权重只负责明场分割；没有训练或微调环表型分类模型，也没有经人工真值验证的 precision/recall 或阳性率。

## 安装与真实运行

建议 Python 3.11/3.12，CPU 可运行；GPU 可选。Windows、Colab、AutoDL 都使用同一套 Python CLI。首次运行会从 Cellpose 官方下载约 26 MB 的 `yeast_BF_cp3` 权重到 `outputs/model_cache`，之后可离线复用。默认以半分辨率运行分割，再将标签用最近邻映射回原始坐标；边界精度约受 2px 采样限制，可设 `inference_scale: 1.0` 做原分辨率推理。

```bash
git clone https://github.com/wangherm/ring-phenotype-detector.git
cd ring-phenotype-detector
python -m venv .venv
# Linux/AutoDL:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cellpose.txt
python -m pytest -q

python -m ring_detector.cli audit --input /path/to/TEST.zip --output outputs/audit
python -m ring_detector.cli segment --manifest outputs/audit/manifest.csv --config configs/default.yaml --output outputs/segments
python -m ring_detector.cli propose --manifest outputs/audit/manifest.csv --segments outputs/segments --config configs/default.yaml --output outputs/proposals
python -m ring_detector.cli review-export --proposals outputs/proposals --output outputs/review
```

也可以使用 `predict --baseline-only` 代替 `propose`，这是同一套未训练的候选算法，不会加载或伪造 checkpoint。

只运行传统明场分割时，安装 `requirements.txt` 并将命令配置改为 `configs/classical.yaml`。新版不静默回退到传统算法；缺少 Cellpose 时会清楚报错。Cellpose 3 需要 NumPy <2.1，因此完整安装使用 NumPy 2.0 与兼容的 tifffile 版本。

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
- `morphology_overlays/*.png`：只看明场细胞定位；`fluorescence_overlays/*.png`：只看荧光环候选。匹配到细胞内部的候选为紫色，边界/归属不清的为黄色。
- `unassigned_rings.csv` 和 `unassigned_crops/`：独立检测到了但未可靠归属的候选，不丢弃、不强行配给邻近细胞。只在严格内部覆盖与边界间距满足条件时进行匹配。
- `proposals/cells.csv`、`rings.csv`、`fov_qc.csv`：逐细胞候选与特征、逐视野 QC。`ring_score` 是启发式检索分数，`p_positive` 留空；所有可判读区域 `label_pred=uncertain`，边缘截断/大块合并暂为 invalid。**无候选不代表 negative，候选不代表 positive。**
- `crops/*.png`：左 morphology、中荧光强度、右 mask 与候选；`crop_masks/*.png` 和 `raw_crops/*.npz` 保留 mask、荧光各通道原值。
- `review/review.html`：离线打开即可标 `positive/negative/uncertain/invalid` 并下载 CSV。初始标签全空，原图 FOV/cell/框映射完整。保留整套 review 文件夹，避免裁剪路径断开。
- `review/annotations.csv`：每 FOV 平衡导出最多 6 个样本（候选、亮斑/无环、弱信号、QC），而非只挑高分；`all_cells_for_review.csv` 可用于全部细胞复核。
- `cell_uid` 绑定原始明场和分割结果。换算法/换 mask 后禁止将旧 cell_id 标签直接套入；`review-check` 会拒绝身份不匹配的旧标签，需重新审核。
- 逐视野独立输出区域数、去重环候选数、已关联环数和未关联环数。不同荧光通道不混加；共定位且尺度一致的跨通道候选共享 ring_group_id，避免把同一位置重复计数。环数/细胞数不是有环细胞比例。

审核判断：明确内部亮边暗心的近闭环=positive；可判读但无环=negative；离焦/弱信号/可疑结构=uncertain；错分割=invalid。用 contact sheets 快速定位，再打开原分辨率 crop/overlay 判断；缩略图不能代替细节检查。

## 已知限制与下一阶段

预训练酵母明场分割仍会漏掉离焦细胞、分错出芽/分裂细胞，或把细胞间隙当对象。旧 Canny/分水岭基线保留供比较。QC pass 不等于人工确认分割正确。候选算法增加近闭合环的最大缺口限制和跨图块去重；仍偏向可拟合椭圆，弱环、严重开环、不规则环可能漏检。针对低动态范围图块，要求环边缘-中心差同时超过原始显示量化级别与局部噪声门槛，避免归一化放大噪声；孤立小环使用显著信号的尺度估计，避免被整块分位数淹没。默认最小半径为 5px，其他像素尺度应调整 min_radius/max_radius/diameter。原始强度门槛须随真实成像噪声校准，不能宣称跨染料已验证。颗粒围成的圈或大暗腔周边信号可能是假阳性，需科研人员审核。预训练模型说明见 [Cellpose 官方文档](https://cellpose.readthedocs.io/en/v3.1.1.1/models.html)；使用信息见 THIRD_PARTY_NOTICES.md。

先复查多个日期、处理条件的分割与候选，并建立手工计数/定位的小型分割真值；然后标约 50–100 个明确阳性和相当阴性（可行性目标，不是硬门槛），复标部分样本。未来 P2 才按 FOV/日期分组划分并训练 DenseNet201；uncertain/invalid 不进入正负训练。P3 的阳性比例与条件推断必须等经过验证的标签/模型。本次不实现空的 train/model 模块。

真实运行报告 `P0_P1_RESULTS.md` 与人工审核包单独在本地交付。含实验日期、条件和逐视野数字的报告、原始数据及大批运行产物不提交到公开 GitHub。
