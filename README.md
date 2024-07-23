# [FrameSeqTools](https://github.com/1592063346/FrameSeqTools/tree/main)

用于 SBD 模型训练集生成。

## 1 使用

```python
from frame_seq_tools.utils import FrameSeqTools

# frames1, frames2 为两个形为 (n, height, width, 3) 的 np.array
# 使用有些方法时，frames2 不是必须的
tools = FrameSeqTools(frames1, frames2)

# Example 1: 以渐变的形式合并 frames1, frames2 两个片段
new_frames1, one_hot, multi_hot = tools.gradual_transition_merge(gradual_type=0)

# Example 2: 为 frames1 添加随机纯色滤镜
new_frames2 = tools.frame_glare()
```

## 2 方法列表

一些函数有部分内部参数基于随机化，以在一定程度上防止过拟合。

### 2.1 片段合并/裁切

**提供的帧序列应当是干净的片段，即本身不存在镜头切换。**

所有方法的输出均为 `frames`, `one_hot`, `multi_hot`：
- `frames`：合并后的帧序列
- `one_hot`：one-hot 标记序列（对于长度为偶数的渐变段，默认标记中心位置为除以 $2$ 下取整）
- `multi_hot`：multi-hot 标记序列


#### 2.1.1 <font size=5>`forced_merge`</font>

以硬过渡形式，直接合并两个帧序列片段。

案例见 `example/1/forced.mp4`。

#### 2.1.2 <font size=5>`gradual_transition_merge`</font>

过渡以渐出渐入（中间可加入黑幕或白幕）的形式，合并两个帧序列片段。

案例见 `example/1/gradual_1.mp4` 与 `example/1/gradual_2.mp4`。

**参数：** **`gradual_frame_num`**, **`gradual_type`**
  - **`gradual_frame_num`**：渐变段帧数。默认为 -1，表示取 $[5, 25]$ 内的随机整数值（对应 25 FPS 下的 $[0.2, 1]$ s）
  - **`gradual_type`**：渐变类型。0 表示正常渐变；1 表示渐变中间加入一定帧数的黑幕；2 表示渐变中间加入一定帧数的白幕。默认为 -1，表示取 $[0, 2]$ 内的随机整数值

注意：如果 `gradual_type` 为 1 或者 2，则整个切换部分形如：从片段 1 过渡到黑幕/白幕 → 一定帧数的黑幕/白幕 → 从黑幕/白幕过渡到片段 2。加入的黑幕/白幕的帧数为 $[12, 30]$ 内的随机整数值（对应 25 FPS 下的 $[0.5, 1.2]$ s），此时的渐变段总帧数实际上为 (`gradual_frame_num` 的两倍 + 黑幕/白幕帧数)。

#### 2.1.3 <font size=5>`push_transition_merge`</font>

过渡以推出的形式，合并两个帧序列片段。

案例见 `example/1/push_1.mp4` 与 `example/1/push_2.mp4`。

**参数：** **`gradual_frame_num`**, **`push_type`**
  - **`gradual_frame_num`**：渐变段帧数。默认为 -1，表示取 $[5, 25]$ 内的随机整数值（对应 25 FPS 下的 $[0.2, 1]$ s）
  - **`push_type`**：推出类型。0, 1, 2, 3 分别表示自下而上、自上而下、从左往右、从右往左推出；4, 5, 6, 7 分别表示左下往右上、右下往左上、右上往左下、左上往右下推出。默认为 -1，表示取 $[0, 7]$ 内的随机整数值

#### 2.1.4 <font size=5>`wipe_transition_merge`</font>

过渡以擦除的形式，合并两个帧序列片段。

案例见 `example/1/wipe_1.mp4`、`example/1/wipe_2.mp4`、`example/1/wipe_3.mp4`、`example/1/wipe_4.mp4` 与 `example/1/wipe_5.mp4`。

**参数：** **`gradual_frame_num`**, **`wipe_type`**
  - **`gradual_frame_num`**：渐变段帧数。默认为 -1，表示取 $[5, 25]$ 内的随机整数值（对应 25 FPS 下的 $[0.2, 1]$ s）
  - **`wipe_type`**：擦除类型。0, 2, 4, 6, 8 从左往右依次对应上面案例中的五种擦除形式；1, 3, 5, 7, 9 从左往右依次对应上面五种擦除的反向形式（如圆从内向外扩张对应的反向形式为由外向内收缩）。默认为 -1，表示取 $[0, 9]$ 内的随机整数值

#### 2.1.5 <font size=5>`frame_crop_split`</font>

将帧序列分为两段，第一段为原始序列，第二段为裁剪后的帧窗口（保持帧的原大小不变）。两段间使用硬过渡。

案例见 `example/1/crop_split.mp4`。

**参数：** **`location_type`**, **`window_ration`**
  - **`location_type`**：窗口位置。0~8 依次对应中心、上方、左侧、右侧、下方、左上角、右上角、左下角、右下角。默认为 -1，表示取 $[0, 8]$ 内的随机整数值
  - **`window_ration`**：窗口的高度和宽度占总视频的高度和宽度的比例。该值为 $(0, 1]$ 内的实数，默认为 0.8

#### 2.1.6 <font size=5>`merge`</font>

以一定概率对上述 5 个方法做随机调用。

**参数：** **`prob`**
  - **`prob`**：各方法调用概率。为一个长度为 5 的非负数组，表示各方法调用的概率权重。数组可以不必总和归一化。默认为 [0.36, 0.36, 0.1, 0.06, 0.12]

### 2.2 数据增强

所有方法的输出均为 `frames`：

- `frames`：处理后的帧序列

#### 2.2.1 <font size=5>`frame_darken`</font>

使帧序列昏暗。

案例见 `example/2/darken.mp4`。

**参数：** **`dark_coef`**
  - **`dark_coef`**：昏暗度参数。该值为 $[0, 1]$ 内的实数，0 为保持原色，1 为纯黑。默认为 0.85

#### 2.2.2 <font size=5>`frame_glare`</font>

为帧序列分段添加随机纯色滤镜。

案例见 `example/2/glare.mp4`。

**参数：** **`color_type`**
  - **`color_type`**：颜色种类。0~7 依次对应 RGB 表示下的红、绿、蓝、黄、青、橙、粉、紫色。默认为 -1，表示取 $[0, 7]$ 内的随机整数值

注意：`cv2` 库使用的是 BGR 表示。

#### 2.2.3 <font size=5>`frame_extract`</font>

等距抽取出帧重组视频（保持总帧数不变）。

案例见 `example/2/extract.mp4`。

**参数：** **`gap`**
  - **`gap`**：抽帧间距。默认为 2，表示每 2 帧中抽出 1 帧

#### 2.2.4 <font size=5>`frame_crop`</font>

裁剪帧窗口（保持帧的原大小不变）。

案例见 `example/2/crop.mp4`。

**参数：** **`location_type`**, **`window_ration`**
  - **`location_type`**：窗口位置。0~8 依次对应中心、上方、左侧、右侧、下方、左上角、右上角、左下角、右下角。默认为 -1，表示取 $[0, 8]$ 内的随机整数值
  - **`window_ration`**：窗口的高度和宽度占总视频的高度和宽度的比例。该值为 $(0, 1]$ 内的实数，默认为 0.8
