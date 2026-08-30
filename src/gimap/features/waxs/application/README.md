# Application

WAXS image load、integration、batch 和 export use cases；只依赖 domain 与 ports。

`WaxsBatchRequest.sources` 允许一个任务包含多个 `WaxsBatchSource`。每个 source 拥有输入
folder、glob pattern 和简单且唯一的 output subfolder；旧的单 folder 字段仍作为兼容 fallback。
Batch 可独立请求 pixel 2D PNG、真实 q-grid 2D PNG、1D CSV 和出版级 1D PNG。可选 q range
只控制 q 图显示窗口，不改变 detector image 或 1D integration 的科学输入。
