# xuhui AT 数值生成 SOP（基于专家 Time@AT 的 10s 时间加权取值）

版本：v1.0  
适用数据：xuhui（呼吸级时序 + 专家 Time_at_AT）  
产出标签：`VO2_kg_at_AT_gold`（主），`VO2_kg_at_AT_sop`（未校准留档）  
label_source：`expert_time_sop`  
label_method：`time_weighted_10s_winsor005_995`  
sop_version：`v1.0`

---

## 1. 目的
以最小专家成本，将“仅有专家 AT 时间（Time_at_AT）的 xuhui 数据”转换为受背书的 VO2/kg@AT 数值标签，用于模型训练与补充评估；保证方法可审计、可复现、与专家值等效或非劣。

## 2. 输入与先决条件
- 必需字段（按 Examination_ID 分组的时序表）
  - `Time`（可解析为秒，支持 mm:ss/HH:MM:SS/秒数）
  - `VO2_kg`（mL/kg/min）。若缺失而有 `VO2`（mL/min）+ 体重（kg），则 `VO2_kg = VO2 / weight`；无法计算则记缺失
  - `Examination_ID`
  - 建议：`RER`、`VE`、`VCO2`、`HR`、`Power_Load`（用于质控与审计图）
- 必需字段（目标表）
  - `Time_at_AT`（专家给定 AT 时间；单位秒或可解析）
- 单位：`VO2_kg` = mL/kg/min；`Time` = 秒

## 3. 核心定义（10s 时间加权取值规约）
- 窗口：以 `Time_at_AT` 为中心的 10 秒窗口 `[t_AT − 5, t_AT + 5]`
- 权重：对不等间隔的呼吸级数据，按“该呼吸在窗口内的时间覆盖 dt_i（秒）”作为权重做时间加权均值：
  - 使用相邻时间点中点估计每口气的左右边界，与窗口相交得到 `dt_i`
  - `VO2_kg_at_AT_sop = sum(dt_i * VO2_kg_i) / sum(dt_i)`
- 稳健：对窗口内 `VO2_kg` 做轻度 winsorize（0.5%–99.5%）后再加权，抑制极端值影响
- 覆盖度：`coverage_ratio = sum(dt_i) / 10.0`，反映窗口实际覆盖
- 匹配容差：允许 ±5s 内的代表时间偏移，记录 `time_delta_s`（理论 ≤ 5s）
- 注：比值型信号（如 VE/VO2）应使用“和/和”再比；`VO2_kg` 为标量，直接时间加权均值

## 4. 操作步骤
1) 输入标准化：
   - 解析 `Time` 至秒；按 `Examination_ID, Time` 排序去重
   - 计算/校验 `VO2_kg`；缺失则记 `missing_vo2kg`
2) 窗口与权重：
   - 以 `Time_at_AT` 建立 `[t_AT−5, t_AT+5]` 窗口，计算每口气与窗口相交的 `dt_i`
   - 计算 `coverage_ratio`；<0.5 记为低覆盖警告
3) 时间加权求值：
   - 对窗口内 `VO2_kg` 做 winsorize（0.5%/99.5%）→ 以 `dt_i` 归一化权重加权均值 → `VO2_kg_at_AT_sop`
   - 记录 `time_delta_s`
4) 质量与告警：
   - `coverage_ratio < 0.8` → `warning=low_coverage`
   - 窗口内有效呼吸数 < 3 → `warning=few_samples`
   - `Time_at_AT` 无法解析/越界/重复 → `warning=invalid_time`
   - 阈值边界：`|v−11| ≤ 0.3` → `borderline_11=1`；`|v−14| ≤ 0.3` → `borderline_14=1`
5) 校准（如已建立；见第 6 节）
   - 若存在校准参数 `(a,b)`：`VO2_kg_at_AT_gold = a + b * VO2_kg_at_AT_sop`
   - 同时保留未校准值 `VO2_kg_at_AT_sop` 与 `calib_version`
6) 产出与存档：
   - 每例输出：`Examination_ID, Time_at_AT, VO2_kg_at_AT_sop, VO2_kg_at_AT_gold, coverage_ratio, time_delta_s, borderline_11, borderline_14, warnings, label_source, label_method, sop_version, calib_version, code_commit_hash`
   - 可选审计图：原始 `VO2_kg` 曲线叠加 AT 窗口与取值点、阈值参照线

## 5. 审计与追溯
- 运行日志：样本数、平均/分布（coverage_ratio、time_delta_s）、警告计数
- 审计 CSV：逐例记录差异与告警，便于抽查与回溯
- 版本化：保存 `sop_version`、`calib_version`、`code_commit_hash`、`run_id`

## 6. 锚定与校准（一次性或按版本）
- 目的：以小样本专家金标准锚定 SOP 输出，必要时建立线性校准函数
- 锚定集抽样（一次性）：
  - N≈60–120；分层覆盖性别/年龄/协议时长/难度/RER≥1.05/靠近阈值（±0.5）
  - ≥2 名高年资专家在 10s 视图独立判 `VO2_kg@AT`，分歧由第三人仲裁为 `gold_anchor`
- 校准与等效验证：
  - 用 SOP 计算锚定集 `VO2_kg_at_AT_sop` 与 `gold_anchor` 对比
  - 若存在系统偏差，用 Deming/Passing–Bablok 回归拟合：`VO2_gold = a + b * VO2_sop`，锁定 `(a,b)` 及置信区间
  - 复验等效性（见“验收阈值”）；记录 `calib_version`

## 7. 验收阈值（建议）
- 等效性（TOST）：δ=0.5 mL/kg/min（如更严格用 0.3），p<0.05
- 偏倚：`|Bias| ≤ 0.10` mL/kg/min；Bland–Altman 95% 一致界限窄且无趋势性偏倚
- 阈值一致：11/14 mL/kg/min 一致率 ≥ 0.95；误差 P95 ≤ 0.5 mL/kg/min

## 8. 质量控制与复核
- 自动复核池（推荐规则）：
  - `coverage_ratio < 0.8` 或 `few_samples` 或 `time_delta_s > 3s`
  - `borderline_11/14=1`（阈值±0.3 内）
  - 任意系统异常 `warnings`
- 复核流程：每期抽查 ≥60 例或高风险全查；两名专家独立复核，必要时仲裁
- 周期监控：月度偏倚、MAE、阈值翻转率、coverage_ratio 分布；超阈触发回看/再校准

## 9. 与模型训练/评估的关系
- 训练：`VO2_kg_at_AT_gold` 可与 shanxi/zhongshan 专家真值一起作为强监督标签
- 评估：
  - 主测试集仍以独立专家/设备真值（shanxi/zhongshan）为主结论
  - xuhui 作为“经锚定等效验证的方法学金标准”单列展示，明确来源与限制
  - punan 仅以 `Time@AT` 为主；如做 VO2，仅作代理（proxy）

## 10. 默认参数（可调并版本化）
- 窗口宽度 `W=10s`；容差 ±5s
- winsorize 分位：0.5% 和 99.5%
- 低覆盖阈值：0.8（软警告）；0.3（硬失败输出空值）
- 阈值边界带：±0.3 mL/kg/min（用于 `borderline_11/14`）
- 锚定集等效阈值：TOST δ=0.5（或 0.3），`|Bias|≤0.1`，一致率≥0.95，P95≤0.5

## 11. 例外与降级
- 无法得到 `VO2_kg`：输出空值并标 `missing_vo2kg`
- `Time_at_AT` 解析失败或超范围：`invalid_time`，输出空值
- 覆盖严重不足（`coverage_ratio < 0.3`）：输出空值，`low_coverage_hard`
- AT 时间重复：按首次有效值处理并标 `duplicated_time`

## 12. 落地产物与路径（建议）
- 输出 CSV：`cpet_former/experiments/xuhui_at_gold.csv`
- 审计图：`cpet_former/experiments/plots/xuhui_at_gold/<Examination_ID>.png`
- 报告：`cpet_former/experiments/reports/xuhui_sop_validation.md`

---

备注：若将该 SOP 由 `vox_cpet/vox_cpet/dataset/cpet_processor.py` 调用，请确保使用与本 SOP 一致的取值规则（10s 窗、时间加权、winsorize、半窗容差），并在输出中追加 `sop_version/calib_version/code_commit_hash` 字段以便审计与追溯。
