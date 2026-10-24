from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent


def box(x, y, w, h, title, body, fill, stroke="#1f2937", title_fill="#0f172a"):
    body_lines = body if isinstance(body, list) else [body]
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        f'<text x="{x + 16}" y="{y + 28}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="{title_fill}">{title}</text>',
    ]
    ty = y + 54
    for line in body_lines:
        parts.append(
            f'<text x="{x + 16}" y="{ty}" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#334155">{line}</text>'
        )
        ty += 22
    return "\n".join(parts)


def arrow(x1, y1, x2, y2, label=""):
    parts = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" stroke-width="3" marker-end="url(#arrow)"/>'
    ]
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - 8
        parts.append(
            f'<text x="{mx}" y="{my}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#475569">{label}</text>'
        )
    return "\n".join(parts)


def svg_wrap(width, height, title, content):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#475569"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="36" y="42" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="#0f172a">{title}</text>
  {content}
</svg>
"""


def save(name, text):
    path = OUT_DIR / name
    path.write_text(text, encoding="utf-8")
    return path


def ct_swin_svg():
    content = "\n".join(
        [
            box(40, 90, 220, 130, "CT Input", ["3 aligned slices", "max / above / below", "shape: (B, 3, 128, 128)"], "#dbeafe"),
            box(320, 90, 240, 150, "Swin-Tiny Backbone", ["pretrained timm model", "forward_features", "output NHWC feature map"], "#e9d5ff"),
            box(620, 90, 220, 150, "Attention Refinement", ["NHWC -> NCHW", "optional SE", "CBAM enabled in config"], "#fde68a"),
            box(900, 90, 220, 150, "Global Pooling", ["spatial mean pooling", "feature dim = 768"], "#bbf7d0"),
            box(1180, 90, 220, 170, "Classifier Head", ["Linear 768 -> 64", "LayerNorm + GELU + Dropout", "Linear 64 -> 8", "LayerNorm + GELU + Dropout", "Linear 8 -> 1"], "#fecaca"),
            arrow(260, 155, 320, 155),
            arrow(560, 165, 620, 165),
            arrow(840, 165, 900, 165),
            arrow(1120, 175, 1180, 175),
            '<text x="40" y="305" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#475569">Code basis: ct_model_Swin_tiny/dataset.py and ct_model_Swin_tiny/model.py</text>',
        ]
    )
    return svg_wrap(1450, 360, "CT Swin-Tiny Binary Recurrence Model", content)


def pathology_svg():
    content = "\n".join(
        [
            box(40, 90, 250, 140, "Pathology Input", ["UNI2 slide feature", "npy per patient", "shape: (B, 1536)"], "#dbeafe"),
            box(360, 90, 310, 230, "MLP Encoder / Classifier", ["Dropout", "Linear 1536 -> 512 + LN + ReLU", "Dropout", "Linear 512 -> 128 + LN + ReLU", "Dropout", "Linear 128 -> 32 + LN + ReLU", "Dropout", "Linear 32 -> 1"], "#fecaca"),
            arrow(290, 160, 360, 160),
            '<text x="40" y="355" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#475569">Code basis: pathology_model/dataset.py and pathology_model/model.py</text>',
        ]
    )
    return svg_wrap(760, 400, "Pathology UNI2 Binary Recurrence Model", content)


def fusion_svg():
    content = "\n".join(
        [
            box(40, 110, 220, 130, "CT Branch Input", ["3 CT slices as channels", "(B, 3, 128, 128)", "max / above / below"], "#dbeafe"),
            box(40, 290, 220, 120, "Pathology Input", ["UNI2 feature", "(B, 1536)"], "#dcfce7"),
            box(40, 460, 220, 120, "Clinical Input", ["11 tabular variables", "z-score normalized"], "#fee2e2"),
            box(320, 90, 250, 170, "CT Encoder", ["Swin-Tiny", "optional SE / CBAM", "global average pooling", "projection: 768 -> 16"], "#e9d5ff"),
            box(320, 280, 250, 140, "Pathology Adapter", ["Dropout", "Linear 1536 -> 64", "LayerNorm + ReLU"], "#fde68a"),
            box(320, 450, 250, 140, "Clinical Adapter", ["Dropout", "Linear 11 -> 12", "LayerNorm + ReLU"], "#fbcfe8"),
            box(640, 120, 250, 120, "Token Projection", ["CT 16 -> 64", "Path 64 -> 64", "Clin 12 -> 64"], "#bfdbfe"),
            box(640, 280, 250, 130, "Token Builder", ["stack 3 modality tokens", "add learnable modality embedding", "shape: (B, 3, 64)"], "#c7d2fe"),
            box(960, 180, 270, 140, "Cross-Modal Transformer", ["2 encoder layers", "4 heads", "FFN dim 128", "global self-attention"], "#ddd6fe"),
            box(1280, 120, 280, 180, "Clinical-Prior Gating", ["use clinical token as query", "MLP -> 3 gate logits", "temperature = exp(log_temperature)", "sigmoid(logits / temperature)", "gate each modality token"], "#fde68a"),
            box(1280, 360, 280, 140, "Adaptive Aggregation", ["learnable modality weights", "softmax over 3 modalities", "weighted sum -> fused token"], "#bbf7d0"),
            box(1610, 250, 230, 140, "Classifier Head", ["Linear 64 -> 32", "LayerNorm + GELU + Dropout", "Linear 32 -> 1"], "#fecaca"),
            arrow(260, 175, 320, 175),
            arrow(260, 350, 320, 350),
            arrow(260, 520, 320, 520),
            arrow(570, 175, 640, 175),
            arrow(570, 350, 640, 350),
            arrow(570, 520, 640, 350, "project to token space"),
            arrow(890, 350, 960, 250),
            arrow(1230, 250, 1280, 250),
            arrow(1420, 300, 1420, 360),
            arrow(1560, 430, 1610, 320),
            '<text x="40" y="640" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#475569">Code basis: fusion_model/dataset.py and fusion_model/model.py</text>',
        ]
    )
    return svg_wrap(1880, 700, "Multimodal CMT-CPG Fusion Model", content)


def main():
    save("ct_model_swin_tiny_architecture.svg", ct_swin_svg())
    save("pathology_model_architecture.svg", pathology_svg())
    save("fusion_model_architecture.svg", fusion_svg())


if __name__ == "__main__":
    main()
