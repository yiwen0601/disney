# 上海迪士尼乐园运营客流预测与商业洞察分析

一个面向上海迪士尼乐园运营分析与出行判断场景的本地数据产品。  
用户选择一个日期区间后，系统会结合节假日、学校假期、季节、天气与历史规律，对该时间段的客流趋势进行预测，并输出可直接用于 business 展示的运营建议、游客建议和摘要结论。

## 项目简介

这个项目希望解决的不是“预测一个数字”这么简单，而是把客流预测转化为更容易被理解和使用的商业洞察。

在页面中，用户可以：

- 选择开始日期和结束日期
- 查看该时间段的预计总客流与日均客流
- 识别最值得关注的高峰日与相对轻松的窗口
- 获取面向运营方的资源调度建议
- 获取面向游客的行程建议
- 在配置 AI 接口后生成管理摘要

## 适用场景

- 运营管理展示
- 节假日客流趋势分析
- 园区资源排班与补货讨论
- 游客出行前的日期比较与行程判断
- 数据分析课程中的“小型数据产品”展示

## 数据说明

### 数据来源

由于上海迪士尼官方不公开逐日客流数据，本项目并不使用官方逐日客流原始表，而是基于公开资料构建“规则生成 + 历史规律模拟”的训练数据。  
其中，历史训练数据仍然是基于公开事实约束构建的模拟数据；页面在进行近未来日期分析时，会优先接入高德天气 API 的上海预报数据，以增强天气相关建议的真实感。核心参考包括：

- [上海迪士尼度假区官方网站](https://www.shanghaidisneyresort.com/zh-cn/)：用于参考园区运营信息、门票与主题活动节奏
- [上海迪士尼度假区主题活动示例：新春主题装饰](https://www.shanghaidisneyresort.com/zh-cn/experience/event/cny-decoration-event)：用于参考季节性活动对客流波动的潜在影响
- [国务院办公厅关于 2025 年部分节假日安排的通知](https://www.gov.cn/zhengce/zhengceku/202411/content_6986383.htm?sourcefrom=aladdin)：用于参考法定节假日与调休安排
- [上海市教育委员会关于印发《上海市中小学 2024 学年度校历》的通知](https://edu.sh.gov.cn/xxgk2_zdgz_jcjy_05/20240329/34b9d042f7664c1d81bb0e703af6539e.html)：用于参考寒假与暑假窗口
- [高德天气查询 API](https://lbs.amap.com/api/webservice/guide/api/weatherinfo/)：用于页面分析时获取上海的实时/近未来天气预报
- [AECOM Theme Index Report 2023](https://aecom.com/theme-index/)：用于参考主题乐园行业公开客流基线与恢复趋势

换句话说，当前数据集是“基于公开事实约束的模拟数据”，而不是官方直接发布的上海迪士尼每日客流数据。

### 数据文件

- `data/raw/shanghai_disney_attendance.csv`：基础客流数据，共 3653 行、22 列
- `data/processed/shanghai_disney_featured.csv`：特征工程后的训练数据，共 3653 行、35 列
- `data/processed/train_data.csv`、`data/processed/test_data.csv`：训练集与测试集切分结果
- 数据时间范围：`2016-01-01` 至 `2025-12-31`

### 关键字段

- `date`：日期
- `attendance`：当日客流量
- `is_holiday`：是否节假日
- `is_school_break`：是否学校假期
- `temperature`：温度
- `is_rainy`：是否降雨
- `has_special_event`：是否有特殊活动
- `attendance_lag1`、`attendance_lag7`、`attendance_rolling_30`：滞后与滚动窗口特征
- `season_encoded`、`month_sin`、`weekday_cos` 等：周期性与编码特征

## 方法流程

### 数据处理

- 使用 Python 对日期、节假日、学校假期、天气等字段进行构造与整理
- 进行异常值检查、字段转换和特征工程
- 构造周期性特征、分类编码特征、滞后特征和滚动统计特征

### 探索性分析

- 分析季节、月份、节假日、天气等因素与客流的关系
- 输出多张图表，用于解释业务规律和支持后续建模

### 模型训练

- 比较线性回归、Ridge、Lasso、随机森林、梯度提升等方法
- 当前部署模型保存在 `models/disney_attendance_model.joblib`
- 模型对象包含 `model`、`scaler`、`feature_columns` 和 `metrics`

### 产品化输出

- 使用 `app.py` 提供本地 Flask 页面
- 用户选择日期区间后，系统返回区间概览、逐日预测和建议摘要
- 页面默认使用 ModelScope 的 OpenAI 兼容接口生成摘要、运营建议、游客建议和逐日建议
- 页面默认接入高德天气 API，并使用上海天气数据辅助近未来日期判断

## Notebook 导览

如果希望从数据分析流程而不是页面入口开始阅读，可以按下面的顺序查看 notebook：

1. `01_data_collection.ipynb`
   这一部分展示数据来源调研、历史客流数据构建思路、特征补充和数据质量检查。
2. `02_data_analysis.ipynb`
   这一部分展示数据清洗、缺失值与异常值检查、探索性分析、节假日影响、季节性变化、天气影响和相关性分析。
3. `03_model_training.ipynb`
   这一部分展示特征准备、模型对比、评估指标、超参数调优和最终模型保存。

如果想直接查看带输出结果的分析版本，可优先打开：

- `01_data_collection_executed.ipynb`
- `02_data_analysis_executed.ipynb`
- `03_model_training.ipynb`

说明：
已执行版本主要用于展示结果快照；如果需要按当前目录结构重新运行，请优先使用未执行的源码版本。

图表输出统一保存在 `images/` 目录，数据产物统一保存在 `data/raw/` 与 `data/processed/` 目录。

## 核心发现

1. 节假日是最强影响因素之一。样本中节假日日均客流约为 `114,314` 人，较非节假日的 `69,738` 人提升约 `63.9%`。
2. 学校假期具有稳定拉动作用。学校假期日均客流约 `82,918` 人，较非学校假期高约 `17.6%`。
3. 客流存在明显季节性差异。夏季日均客流最高，约 `89,120` 人；春季次之；秋季最低。
4. 月度高峰主要集中在 `4-6 月`，其中 `6 月`、`4 月`、`5 月` 的平均客流最高。
5. 当前部署的优化随机森林模型表现较稳定，`R2=0.7563`，`MAE=14,096`，`MAPE=13.32%`，适合做趋势判断与业务展示。

## 快速开始

### 1. 激活环境

```bash
conda activate disney_business_py310
```

如果本地还没有环境，可执行：

```bash
conda create -y -n disney_business_py310 python=3.10
conda activate disney_business_py310
pip install -r requirements.txt
```

### 2. 启动页面

```bash
python app.py
```

启动后访问：

```text
http://localhost:5001
```

### 3. AI 建议生成

当前页面默认启用 AI 建议生成，后端会优先读取环境变量中的 `MODELSCOPE_ACCESS_TOKEN`；如果没有显式设置，则使用应用内的默认 Token。

如需覆盖为你自己的 Token，可在启动前设置：

```bash
export MODELSCOPE_ACCESS_TOKEN="你的 ModelScope Token"
python app.py
```

当前接口配置：

- `base_url`: `https://api-inference.modelscope.cn/v1/`
- `model`: `ZhipuAI/GLM-5.1`

### 4. 天气数据配置

页面默认使用高德天气 API，并默认查询上海 `adcode=310000` 的天气预报。  
如果需要切换 Key 或城市，可在启动前设置：

```bash
export AMAP_WEATHER_KEY="你的高德天气 Key"
export AMAP_CITY_CODE="310000"
python app.py
```

说明：

- 当所选日期落在高德可提供的预报窗口内时，页面优先使用真实天气预报
- 超出高德预报窗口的日期，系统会回退到基于历史分布的天气估计逻辑

## 页面输出内容

启动页面后，系统会围绕所选日期区间输出以下内容：

- 区间预计总客流
- 区间日均客流
- 峰值日期与压力等级
- 相对低峰的推荐窗口
- 关键驱动因素
- 运营建议
- 游客建议
- AI 管理摘要
- AI 生成的运营建议
- AI 生成的游客建议

## 项目结构

```text
disney/
├── app.py
├── templates/
│   └── index.html
├── images/
│   ├── attendance_analysis.png
│   ├── eda_holiday_impact.png
│   ├── model_comparison.png
│   └── ...
├── models/
│   └── disney_attendance_model.joblib
├── src/
│   ├── app.py
│   ├── generate_data.py
│   ├── process_data.py
│   └── train_model.py
├── data/
│   ├── raw/
│   │   ├── disney_attendance.csv
│   │   └── shanghai_disney_attendance.csv
│   └── processed/
│       ├── disney_attendance_cleaned.csv
│       ├── shanghai_disney_featured.csv
│       ├── train_data.csv
│       └── test_data.csv
├── 01_data_collection.ipynb
├── 02_data_analysis.ipynb
├── 03_model_training.ipynb
├── notebooks/
│   └── data_analysis.ipynb
├── requirements.txt
├── REFLECTION.md
└── SUBMISSION_CHECKLIST.md
```

## 局限性与未来优化

- 当前数据为模拟数据，适合分析展示，不等同于真实园区经营数据
- 当前只接入了高德的近未来天气预报，尚未补齐真实历史天气、门票价格、社交媒体热度或真实活动排期
- 目前主要支持本地运行，后续可进一步部署到云端
- 后续可增加真实历史天气回填、动态可视化图表和更细粒度的商业指标

## AI 使用说明

本项目使用了 AI 工具辅助进行界面文案润色、README 组织、代码调试和摘要生成接口联调；但选题设计、数据流程拆分、特征构造、模型比较、结果核查和最终结论整理均经过人工复核。  
所有关键分析结果均以 Python 输出和本地文件为准，AI 生成内容仅作为辅助表达工具。
