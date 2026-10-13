# GIWAXS 评估案例

`tools/eval_giwaxs_agent.py` 用“案例”回答两个问题：

1. **基线有没有出错**（`baseline`）：在案例的帧上跑一遍 `gimap_agent.py auto` 的流程，核对案例写定的事实。
2. **agent 写的报告怎么样**（`report`）：找出不该出现的说法，数一数转述了哪些基线事实、有哪些超出基线的发现。

一个案例是一个 JSON 文件：一次自己很熟悉的测量的帧、基线在这些帧上必须得到的事实、给报告打分用的规则。

## 案例放在哪里

- 工具默认读 `docs/agents/eval/local/` 里的全部 `*.json`。**这个文件夹被 git 忽略。**
- 案例里写着真实的文件名和在数据上测得的数值。所以建立在私有或未发表数据上的案例只放在 `local/`：
  它留在存着这些数据的电脑上，不提交，里面的名字和数值也不写进代码、测试和文档。
- 克隆出来的仓库里没有 `local/`：工具会说明没有案例并以退出码 2 结束，测试会跳过需要真实案例的那一项。
  要在另一台电脑上用同一个案例，把案例文件和数据一起拷过去，放进那边的 `local/`，再把 `data_root` 改成那边的路径。
- 只用公开数据（例如 `tests/data/external/`）的案例可以提交在本文件夹里（`local/` 之外），
  `data_root` 写成相对仓库根目录的路径，运行时用 `--case` 指定。

## 用法

```bash
python tools/eval_giwaxs_agent.py baseline                  # 在每个本地案例的数据上跑基线并核对
python tools/eval_giwaxs_agent.py baseline --out <文件夹>    # 把这次运行的输出写到指定文件夹（默认是临时文件夹）
python tools/eval_giwaxs_agent.py baseline --from <文件夹>   # 不重跑：读先前一次 auto 运行的 <文件夹>/summary.json
python tools/eval_giwaxs_agent.py report <报告.md>           # 给 agent 写的报告打分
python tools/eval_giwaxs_agent.py baseline --case <案例>     # 案例文件的路径，或 local/ 里的文件名
```

- `local/` 里有多个案例时，`baseline` 逐个运行；`--from` 和 `report` 只对应一个案例，要用 `--case` 指明。
- `--from` 读的是先前的运行，不会再写什么，所以不能和 `--out` 一起用；也可以直接给那个 `summary.json`。
- 基线运行不使用已保存的仪器配置：标定要么在帧附近找到，要么由 `notes` 点名。数据目录只读，输出写到 `--out` 或临时文件夹。
- 案例和报告按 UTF-8 读取。带 BOM 的 UTF-8、Windows PowerShell 5.1 的 `>` 写出的 UTF-16 也能读；
  其他编码（例如 GBK）会被拒绝并说明：照样打分的话，规则匹配不上乱码，看起来就像一份没有错误说法的报告。
- 退出码：
  - 0：每项核对都通过（`report`：没有发现错误说法）；
  - 1：有核对失败（`report`：发现了错误说法，命中的句子会打印出来，由人确认）；
  - 2：什么都没有评估，工具用一句话说明原因：这台电脑上没有案例、案例的数据不在这里，
    或者案例、报告用不了（格式写错的案例、空的报告、编码不对）。1 优先于 2。

## 格式

下面的例子是**编造的**（名字和数值都不来自任何真实数据），只用来说明各个键：

```json
{
  "name": "Invented example: a thin film on glass, two samples",
  "data_root": "D:/data/abc_beamtime",
  "frames": [
    "abc_film_runA_0001/det/abc_film_runA_0001_00001.nxs",
    "abc_film_runB_0002/det/abc_film_runB_0002_00001.nxs"
  ],
  "notes": "GIWAXS, alpha_i = 0.25 deg",
  "baseline": {
    "geometry_source_contains": "abc_standard_lab6_take2",
    "standard": "lab6",
    "max_line_error_percent": 0.2,
    "reliable_peaks": [1.5, 2.2],
    "peak_tolerance": 0.01,
    "spike": 4.2,
    "ring": {"q": 2.2, "f_min": 0.3, "f_max": 0.6},
    "no_series_hints": true,
    "open_questions": 0
  },
  "report": {
    "mistakes": [
      {
        "id": "artefact_called_a_peak",
        "all": ["4\\.2", "(?i)bragg|reflection|衍射峰"],
        "unless": "(?i)artefact|artifact|spike|伪影|坏点",
        "why": "The line at 4.2 is one bin wide and in every sample: an artefact."
      }
    ],
    "baseline_facts": [
      {"id": "calibration_stated", "any": ["(?i)lab6|六硼化镧"]}
    ],
    "findings": [
      {
        "id": "start_and_end_compared",
        "points": 2,
        "all": ["(?i)first frames|第\\s*1\\s*帧", "(?i)appear|出现"]
      }
    ]
  }
}
```

### 顶层

| 键 | 含义 |
| --- | --- |
| `name` | 案例的标题，运行时打印出来；不写就用文件名。 |
| `data_root` | 帧所在的文件夹。绝对路径；相对路径从仓库根目录算起（公开数据）。 |
| `frames` | 帧的路径列表，相对 `data_root`。每一项按一帧交给基线：一个序列文件算一项，基线默认对它最后 10 帧求和。 |
| `notes` | 交给基线的用户笔记原文（相当于 `--notes`），例如入射角。 |
| `baseline` | 基线必须得到的事实，见下。 |
| `report` | 给报告打分的规则，见下。 |

### `baseline`：基线不该出错的事

“每一帧都分析成功”和下面带“总是核对”的两项不需要写键也会核对：案例默认基线是从标样图像标定的。
基线直接采用一份 `.poni` 时没有拟合标样图像，也就没有标样线偏差，这一项通不过。

| 键 | 核对什么 |
| --- | --- |
| `geometry_source_contains` | 每一帧的几何来源里都含有这段文字，即用的是这份标定。 |
| `standard` | 总是核对：每一帧标定质量里的标样键都等于它（`agbh`、`lab6`、`ceo2`、`lab6_ceo2` 等）。 |
| `max_line_error_percent` | 总是核对：标样线的 q 平均偏差（%）存在并且不超过它。默认 0.2。 |
| `reliable_peaks` | q 的列表（Å⁻¹）。每个 q 附近，每一帧都有一个不带警告的峰。 |
| `peak_tolerance` | `reliable_peaks` 的相对容差，默认 0.01（即 1 %）。 |
| `spike` | 一个 q：它 0.1 % 以内的峰在每一帧都带有 `spike` 标记（是伪影，没有被当成峰）。 |
| `no_series_hints` | 为 `true` 时：任何一帧都没有给出晶面序列的提示。 |
| `ring` | 一个环的取向：`q`（2 % 以内的环）；同时给 `f_min` 和 `f_max` 时 Herman f 必须给出并落在其间，都不给时 f 必须不给出；`note_contains` 是环的说明或理由里必须出现的文字。 |
| `sector_note_contains` | 每一帧至少有一个峰的取向说明里含有这段文字。 |
| `open_questions` | 每一帧 Needs attention 里问题数的上限。 |

### `report`：给报告打分

报告先切成句子（按 `。！？!?` 和换行，以及后面隔着空格跟着大写字母或数字的英文句号；分号和小数点不切）。
规则逐句匹配，用的是 Python 的 `re.search`：

- `all`：列表里的每个正则都要在这一句里找到；
- `any`：列表里至少一个正则在这一句里找到；
- `unless`：这个正则在这一句里出现时，这一句不算（用来放过否定句，比如“没有……”）。

| 键 | 含义 |
| --- | --- |
| `mistakes` | 不该出现的说法。每条规则有 `id`、`all` / `any`、`unless`，以及打印给人看的理由 `why`。 |
| `baseline_facts` | 报告应当转述的基线事实。每条规则有 `id` 和 `all` / `any`（也可以有 `unless`）；结果是“转述了几条”。 |
| `findings` | 超出基线的发现。每条规则有 `id`、`all` / `any` 和分值 `points`（默认 1）；结果是得分和满分。 |

工具不读的键会被忽略，可以用来写备注。正则只是第一遍筛选，不是裁判：命中的句子要由人读过再下结论。

规则是手写在 JSON 里的正则（反斜杠要写两个，如 `"4\\.2"`），写错了工具会指出是哪一条并以退出码 2 结束，不会照样打分：

- 规则没有 `id`，或者 `all` / `any` 不是列表（一个字符串会被逐个字符地当成正则）；
- 规则在 `all` 和 `any` 里一个正则都没有（它会匹配每一句）；
- 某个正则编译不了，或者里面有控制字符：只写一个反斜杠的 `"\bno\b"` 在 JSON 里是退格符，不是单词边界，永远匹配不上；
- `points` 不是整数。

`baseline` 部分写错时（例如 `ring` 只给了 `f_min`、没有 `q`，或者该是数值的地方写了文字）也一样：一句话说明，退出码 2。

## 测试

- `tests/test_giwaxs_eval.py` 和 `tests/test_release2_evalcase_*.py` 用写在临时文件夹里的编造案例检查工具的逻辑，
  所以在没有任何案例的克隆上也能通过。
- 有一项测试会真的在本地案例的数据上跑基线（只读，输出写到临时文件夹，要一两分钟）。它只在你要求时运行：
  设环境变量 `GIMAP_EVAL_LOCAL=1`，并且 `local/` 里有案例、它的数据就在这台电脑上；否则这一项会跳过并说明原因。
  平时的主测试集不会主动去读私有数据；要看某个案例过不过，直接运行 `python tools/eval_giwaxs_agent.py baseline`。
- 在有本地案例的电脑上，`tests/test_release_docs_data_policy.py` 还会从案例里读出数据的名字和案例写定的数值，
  检查 git 跟踪的和将要加入的文件里都没有它们。数值指：峰位（三位以上小数的原样查找；在提到 q、峰或环的行里，
  还查找四舍五入到两位小数的写法）、案例名字里写的序列帧数（一百帧以上）和最后十帧的起始帧、样品编号。
  失败信息只给出文件（数值还给出行号），不打印名字和数值。
  只看数据在仓库之外的案例：在 `local/` 里用仓库自带的公开数据起草的案例，名字本来就是公开的。
- 只对某个私有案例有意义的测试（比如用模仿它的数据的句子检查它的正则）也放在 `local/` 里，主测试集不收集它，
  需要时手动运行：`python -m pytest -q -p no:cacheprovider docs/agents/eval/local`。
