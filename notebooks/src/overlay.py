# =============================================================
# src/overlay.py
# Renderizado de overlays para el agente de percepción-acción
# Equipo8-Grupo8-SIC-2025
# =============================================================

import cv2
import numpy as np
from typing import Optional


# Paleta Cityscapes 19 clases — BGR para OpenCV
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],   #  0 road
    [244,  35, 232],   #  1 sidewalk
    [ 70,  70,  70],   #  2 building
    [102, 102, 156],   #  3 wall
    [190, 153, 153],   #  4 fence
    [153, 153, 153],   #  5 pole
    [250, 170,  30],   #  6 traffic light
    [220, 220,   0],   #  7 traffic sign
    [107, 142,  35],   #  8 vegetation
    [152, 251, 152],   #  9 terrain
    [ 70, 130, 180],   # 10 sky
    [220,  20,  60],   # 11 person
    [255,   0,   0],   # 12 rider
    [  0,   0, 142],   # 13 car
    [  0,   0,  70],   # 14 truck
    [  0,  60, 100],   # 15 bus
    [  0,  80, 100],   # 16 train
    [  0,   0, 230],   # 17 motorcycle
    [119,  11,  32],   # 18 bicycle
], dtype=np.uint8)

CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "t-light", "t-sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


class OverlayRenderer:
    """
    Renderiza anotaciones visuales sobre un frame BGR:
    máscaras de segmentación, bounding boxes YOLO y panel HUD.
    """

    def __init__(self, alpha_seg: float = 0.45, alpha_hud: float = 0.6):
        self.palette     = CITYSCAPES_PALETTE
        self.alpha_seg   = alpha_seg
        self.alpha_hud   = alpha_hud
        self.risk_colors = {
            "low":    (  0, 200,   0),   # verde  BGR
            "medium": (  0, 165, 255),   # naranja BGR
            "high":   (  0,   0, 220),   # rojo   BGR
        }

    # ----------------------------------------------------------
    # Segmentación semántica
    # ----------------------------------------------------------

    def _draw_seg_mask(
        self, frame: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Superpone máscara semántica semitransparente sobre el frame.

        mask : np.ndarray int/uint, shape HxW, valores 0-18
        """
        mask_np = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
        mask_np = mask_np.astype(np.int32).clip(0, len(self.palette) - 1)

        # Indexación vectorizada — (H, W, 3) RGB
        colored = self.palette[mask_np]          # uint8 RGB
        colored_bgr = colored[..., ::-1].copy()  # RGB → BGR

        return cv2.addWeighted(frame, 1 - self.alpha_seg,
                               colored_bgr, self.alpha_seg, 0)

    # ----------------------------------------------------------
    # Bounding boxes
    # ----------------------------------------------------------

    def _draw_boxes(self, frame: np.ndarray, boxes) -> np.ndarray:
        """
        Dibuja bounding boxes del resultado YOLO sobre el frame.
        """
        if boxes is None or not hasattr(boxes, "boxes") or boxes.boxes is None:
            return frame

        det = boxes.boxes
        if len(det) == 0:
            return frame

        xyxy_all = det.xyxy.cpu().numpy().astype(int)
        conf_all = det.conf.cpu().numpy()
        cls_all  = det.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(xyxy_all, conf_all, cls_all):
            rgb = self.palette[cls % len(self.palette)]
            color_bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # RGB → BGR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

            label = f"{boxes.names.get(cls, str(cls))} {conf:.2f}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness  = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            ty = max(y1 - 4, th + 4)

            # Shadow (negro) + texto (blanco)
            cv2.putText(frame, label, (x1 + 1, ty + 1),
                        font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, ty),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    # ----------------------------------------------------------
    # Panel HUD
    # ----------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        risk_score: Optional[float],
        level: str,
        frame_count: int,
        uncertainty: Optional[float] = None,
    ) -> np.ndarray:
        """
        Dibuja panel HUD semitransparente en la esquina superior izquierda.
        """
        color = self.risk_colors.get(level, (255, 255, 255))
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness  = 1
        line_h     = 22
        pad        = 8

        # Construir líneas
        score_str = f"{risk_score:.4f}" if risk_score is not None else "N/A"
        lines = [
            (f"Risk:  {score_str}",       color),
            (f"Level: {level.upper()}",   color),
            (f"Frame: {frame_count:04d}", (220, 220, 220)),
        ]
        if uncertainty is not None:
            lines.append((f"sigma: {uncertainty:.3f}", (180, 180, 255)))

        # Tamaño del panel
        max_w = max(
            cv2.getTextSize(txt, font, font_scale, thickness)[0][0]
            for txt, _ in lines
        )
        panel_w = max_w + pad * 2
        panel_h = line_h * len(lines) + pad * 2

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.alpha_hud, frame, 1 - self.alpha_hud,
                        0, frame)

        # Texto con sombra
        for i, (txt, col) in enumerate(lines):
            x = pad
            y = pad + (i + 1) * line_h - 4
            cv2.putText(frame, txt, (x + 1, y + 1),
                        font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, txt, (x, y),
                        font, font_scale, col, thickness, cv2.LINE_AA)

        return frame

    # ----------------------------------------------------------
    # Render completo
    # ----------------------------------------------------------

    def render(
        self,
        frame_bgr: np.ndarray,
        perception_out: dict,
        level: str,
        frame_count: int,
        uncertainty: Optional[float] = None,
    ) -> np.ndarray:
        """
        Aplica todos los overlays sobre una copia del frame.

        Orden: máscara seg → boxes → HUD

        uncertainty : sigma del BNN pasado desde el agente (tiene
                      precedencia sobre perception_out["uncertainty"])
        """
        out = frame_bgr.copy()

        if perception_out.get("mask") is not None:
            out = self._draw_seg_mask(out, perception_out["mask"])

        if perception_out.get("boxes") is not None:
            out = self._draw_boxes(out, perception_out["boxes"])

        unc = uncertainty if uncertainty is not None \
              else perception_out.get("uncertainty")

        out = self._draw_hud(
            out,
            risk_score=perception_out.get("risk_score"),
            level=level,
            frame_count=frame_count,
            uncertainty=unc,
        )

        return out


# =============================================================
# Bloque de prueba — ejecutar directamente para verificar
# =============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    BASE     = Path("/mnt/bigdata/pipeline_samsung/Equipo8-Grupo8-SIC-2025-/notebooks")
    OUT_DIR  = BASE / "data/spatial_outputs/video_outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Imagen real de curated
    imgs = list((BASE / "data/curated").rglob("*_curated.jpg"))
    if not imgs:
        print("[ERROR] No se encontró ninguna imagen _curated.jpg")
        sys.exit(1)

    frame_bgr = cv2.imread(str(imgs[0]))
    print(f"[Test] Imagen base : {imgs[0].name}  shape={frame_bgr.shape}")

    renderer = OverlayRenderer(alpha_seg=0.45, alpha_hud=0.6)

    # Máscara sintética 416×416
    np.random.seed(42)
    H, W = frame_bgr.shape[:2]
    mascara_sint = np.random.randint(0, 19, (H, W))

    pruebas = [
        {
            "tag":         "low",
            "level":       "low",
            "risk_score":  0.50,
            "mask":        None,
            "frame_count": 10,
            "uncertainty": None,
        },
        {
            "tag":         "medium",
            "level":       "medium",
            "risk_score":  0.862,
            "mask":        mascara_sint,
            "frame_count": 42,
            "uncertainty": None,
        },
        {
            "tag":         "high",
            "level":       "high",
            "risk_score":  0.871,
            "mask":        mascara_sint,
            "frame_count": 99,
            "uncertainty": 0.023,
        },
    ]

    for p in pruebas:
        perception_out = {
            "mask":        p["mask"],
            "boxes":       None,
            "risk_score":  p["risk_score"],
            "uncertainty": p["uncertainty"],
        }
        result = renderer.render(
            frame_bgr,
            perception_out,
            level=p["level"],
            frame_count=p["frame_count"],
        )

        save_path = OUT_DIR / f"overlay_test_{p['tag']}.jpg"
        cv2.imwrite(str(save_path), result)
        print(f"[{p['tag'].upper():6s}] shape={result.shape}  →  {save_path}")

    print("\n[✓] OverlayRenderer.render() verificado para low / medium / high.")
