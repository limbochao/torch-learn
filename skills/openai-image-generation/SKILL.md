---
name: openai-image-generation
description: Use this skill whenever the user asks Codex to generate, edit, inpaint, restyle, or create bitmap images through OpenAI or an OpenAI-compatible Responses API image_generation tool. 适用于生图、修图、局部重绘、参考图改风格、透明背景图标、产品图和 UI 素材生成；默认复用当前 Codex 的 config/auth 凭证，也支持自定义 base URL、model 和 API key 环境变量。
---

# OpenAI Image Generation

这个 skill 用来在 Codex 中调用 OpenAI 或 OpenAI-compatible `Responses API` 的 `image_generation` tool，生成或编辑 bitmap image，并把返回的 base64 image 写成文件。默认路径是使用 bundled script，因为它会统一处理 Codex config 发现、凭证读取、request 构造、base64 解码和输出文件。

## 适用场景

使用这个 skill 处理这些请求：

- 生成新图片、产品图、插画、图标、贴图、UI asset、社交媒体配图。
- 基于参考图进行 edit、restyle、variation。
- 使用 mask 做 inpainting / 局部重绘。
- 生成透明背景 PNG、指定尺寸和质量的图片。
- 用户要求“用当前 Codex 凭证生图”或“用自定义 OpenAI-compatible endpoint 生图”。

如果用户只是要改 SVG、CSS、canvas、已有 repo-native asset，优先直接编辑源码，不需要调用这个 skill。

## Runtime

- 默认使用 Python 3：`scripts/generate_image.py`。
- 脚本只依赖 Python standard library，不需要 `pip install`、OpenAI SDK、`curl`、`jq` 或系统 `base64` 命令。
- stdout 只输出最终文件路径或 JSON summary；等待 API 时的 progress message 输出到 stderr。
- 生图可能耗时较长。一次请求只启动一个 generation command，等待它成功或失败后再决定是否重试。

## 凭证策略

默认复用当前 Codex 凭证：

- 读取 `$CODEX_HOME/config.toml`，否则读取 `<home>/.codex/config.toml`。
- 使用顶层 `model_provider` 找到 `[model_providers.<name>]`。
- 使用 provider 的 `base_url` 作为 API base URL。
- 使用顶层 `model` 作为 Responses model，除非用户指定其他 `--response-model`。
- 读取 `$CODEX_HOME/auth.json` 或 `<home>/.codex/auth.json` 中的 `OPENAI_API_KEY`。

也支持自定义凭证和 endpoint：

- `--base-url <url>`：覆盖 Codex provider `base_url`。
- `--response-model <model>`：覆盖 Responses model。
- `--image-model <model>`：覆盖 image_generation tool 使用的 image model，例如 `gpt-image-1`。
- `--api-key-env <NAME>`：从指定环境变量读取 API key，例如 `MY_OPENAI_API_KEY`。
- `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`：Codex config 不可用时的 fallback。

安全约束：

- 不要在命令行传真实 key。脚本会拒绝 `--api-key <value>`。
- 不要 `cat`、打印、提交或复述 `auth.json` 内容。
- 调试凭证时用 `--dry-run`；它只显示 `has_api_key` 和 `api_key_source`，不会输出密钥。
- 如果用户把 key 放在其他环境变量里，只传变量名，不传 secret value。

## 默认工作流

1. 只在缺少关键创意约束时提问，例如 subject、style、aspect ratio、输出路径。能合理默认时直接执行。
2. 默认把图片保存到当前工作区的 `outputs/` 下；用户指定路径时尊重用户路径。
3. 运行 bundled script，一次只跑一个 command，并等待结束：

   ```bash
   python3 <skill-dir>/scripts/generate_image.py \
     --prompt "A precise image prompt" \
     --out outputs/result.png \
     --size 1024x1024 \
     --quality high
   ```

4. 如果使用自定义 provider，不要让用户把 key 写进聊天；让用户先在 shell 环境里设置变量，然后传变量名：

   ```bash
   python3 <skill-dir>/scripts/generate_image.py \
     --prompt "A precise image prompt" \
     --out outputs/custom-provider.png \
     --base-url "https://example.com/v1" \
     --response-model "gpt-5" \
     --image-model "gpt-image-1" \
     --api-key-env MY_OPENAI_API_KEY
   ```

5. 完成后报告输出文件路径、关键参数和 provider/model summary。不要贴 raw response，除非正在 debug。

## 常用命令

生成图片：

```bash
python3 <skill-dir>/scripts/generate_image.py \
  --prompt "A product photo of a matte black ceramic mug on a walnut desk, soft window light, no text" \
  --out outputs/mug.png \
  --size 1024x1024 \
  --quality high
```

透明背景图标：

```bash
python3 <skill-dir>/scripts/generate_image.py \
  --prompt "A clean app icon of a folded paper crane, centered, generous padding, no text" \
  --out outputs/crane-icon.png \
  --background transparent \
  --format png
```

参考图改风格：

```bash
python3 <skill-dir>/scripts/generate_image.py \
  --prompt "Restyle this image as a polished editorial illustration while preserving composition" \
  --image reference.png \
  --action edit \
  --input-fidelity high \
  --out outputs/reference-restyled.png
```

mask 局部重绘：

```bash
python3 <skill-dir>/scripts/generate_image.py \
  --prompt "Replace the masked region with a glass vase of yellow flowers" \
  --image room.png \
  --mask mask.png \
  --action edit \
  --out outputs/room-inpainted.png
```

只验证配置和 request shape，不调用 API：

```bash
python3 <skill-dir>/scripts/generate_image.py \
  --prompt "A quick dry run" \
  --out outputs/dry-run.png \
  --dry-run
```

## Supported Options

脚本支持这些常用参数：

- `--prompt <text>`、`--prompt-file <path>` 或 stdin prompt。
- `--out <path>`：输出图片路径，默认 `generated.png`。
- `--action generate|edit|auto`。
- `--image <path>`：输入参考图，可重复传多个。
- `--mask <path>`：inpainting mask。
- `--size <size>`：例如 `1024x1024`、`1024x1536`、`1536x1024`。
- `--quality low|medium|high|auto`。
- `--format png|webp|jpeg`。
- `--background transparent|opaque|auto`。
- `--input-fidelity high|low`。
- `--moderation auto|low`。
- `--output-compression <0-100>`。
- `--partial-images <0-3>`。
- `--response-model <model>`：Responses API model。
- `--image-model <model>`：`image_generation` tool model。
- `--base-url <url>`：API base URL，脚本会调用 `<base-url>/responses`。
- `--api-key-env <NAME>`：API key 环境变量名，默认 `OPENAI_API_KEY`。
- `--json`：输出 machine-readable summary。
- `--no-progress`：关闭等待期间的 progress message。

OpenAI Responses API 的 image generation tool 使用 `tools: [{"type": "image_generation"}]`，图片结果通常在 `output` 中以 `type == "image_generation_call"` 返回，`result` 为 base64 image data。

## Prompt Guidance

写 prompt 时包含这些信息更稳定：

- subject、setting、medium、lighting、composition、camera/viewpoint。
- aspect ratio、透明背景、是否包含文字、是否需要留白。
- 对 edit 请求，明确哪些内容必须保持不变，哪些内容需要改变。
- 对图标、UI asset、产品图，明确 padding、background、output format、no watermark / no extra text。

## Failure Handling

如果没有图片输出：

- 先看脚本 stderr 中的 API error、refusal、tool error 或 response summary。
- 用 `--dry-run` 验证 base URL、model、key source 和 request shape。
- 确认当前 provider 支持 `Responses API` 和 `image_generation` tool。
- 不要在原 command 还运行时启动第二个相同请求。
- 调试时继续保护 credentials，不打印 request headers 或 auth fields。
