# =============================================================
# src/model_manager.py
# Registro y gestión de modelos PyTorch para el Agente 05
# Equipo8-Grupo8-SIC-2025
# =============================================================

import gc
import torch
from pathlib import Path
from typing import Callable, Optional


class ModelRegistry:
    """
    Registro centralizado de modelos con lazy loading y offloading.

    Ningún modelo se carga en memoria hasta que se solicite
    explícitamente. Mantiene un registro de qué modelos están
    en VRAM para gestionar el límite de 6 GB de la GTX 1660 Ti.
    """

    def __init__(self, vram_limit_gb: float = 5.0):
        """
        Parámetros
        ----------
        vram_limit_gb : float
            Límite conservador de VRAM en GB. Default 5.0 para
            dejar margen al sistema operativo y drivers en la
            GTX 1660 Ti (VRAM total: 5.8 GB usable).
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vram_limit_gb: float = vram_limit_gb
        self.registry: dict[str, dict] = {}
        self.loaded_on_gpu: set[str] = set()

        print(f"[ModelRegistry] Dispositivo : {self.device}")
        if self.device.type == "cuda":
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[ModelRegistry] VRAM libre  : {free_gb:.2f} GB / {total_gb:.2f} GB")
            print(f"[ModelRegistry] Límite fijado: {self.vram_limit_gb} GB")

    # ----------------------------------------------------------
    # Registro
    # ----------------------------------------------------------

    def register_model(
        self,
        name: str,
        checkpoint_path: Path,
        loader_fn: Callable,
        vram_cost_gb: float,
    ) -> None:
        """
        Registra un modelo sin cargarlo en memoria.

        Parámetros
        ----------
        name : str
            Clave lógica del modelo. Ej: "yolo", "segformer",
            "resnet", "gcn", "lstm", "bnn".
        checkpoint_path : Path
            Ruta al archivo de checkpoint (.pt / .safetensors).
            Se verifica existencia aquí — falla rápido si falta.
        loader_fn : Callable
            Función (path) → modelo. Cada modelo tiene su propio
            loader (ultralytics para YOLO, HuggingFace para
            SegFormer, torch.load para el resto).
        vram_cost_gb : float
            Costo estimado en VRAM al cargar en GPU.
            Referencia: yolo≈0.6 · segformer≈1.2 · resnet≈0.3
                        gcn≈0.1  · lstm≈0.1   · bnn≈0.2
        """
        # Validaciones rápidas antes de registrar
        if name in self.registry:
            print(f"[ModelRegistry] AVISO: '{name}' ya registrado — sobreescribiendo.")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"[ModelRegistry] Checkpoint no encontrado para '{name}':\n"
                f"  {checkpoint_path}"
            )

        if vram_cost_gb <= 0:
            raise ValueError(
                f"[ModelRegistry] vram_cost_gb debe ser > 0, recibido: {vram_cost_gb}"
            )

        self.registry[name] = {
            "path"      : checkpoint_path,
            "loader_fn" : loader_fn,
            "vram_gb"   : vram_cost_gb,
            "model"     : None,   # None hasta que se cargue
            "on_gpu"    : False,  # False hasta que esté en VRAM
        }

        print(f"[ModelRegistry] Registrado  : '{name}' | "
              f"VRAM estimada: {vram_cost_gb:.1f} GB | "
              f"checkpoint: {checkpoint_path.name}")

    # ----------------------------------------------------------
    # Carga y offloading
    # ----------------------------------------------------------

    def load_model(self, name: str) -> object:
        """
        Carga un modelo en GPU. Si ya está en GPU lo devuelve directo.
        Si está en CPU (offloadeado) lo mueve a GPU. Si no está
        cargado, llama a loader_fn y lo mueve a GPU.
        """
        if name not in self.registry:
            raise KeyError(f"[ModelRegistry] '{name}' no está registrado.")

        info = self.registry[name]

        # Caso 1 — ya está en GPU, devolver directo
        if info["on_gpu"]:
            return info["model"]

        # Caso 2 — cargado en CPU, solo mover a GPU
        if info["model"] is not None:
            print(f"[ModelRegistry] Moviendo '{name}' CPU → GPU ...")
            info["model"] = info["model"].to(self.device)
            info["on_gpu"] = True
            self.loaded_on_gpu.add(name)
            self._print_vram(f"tras mover '{name}' a GPU")
            return info["model"]

        # Caso 3 — no cargado, llamar a loader_fn
        print(f"[ModelRegistry] Cargando '{name}' desde disco ...")
        model = info["loader_fn"](info["path"])

        if hasattr(model, "to"):
            model = model.to(self.device)

        if hasattr(model, "eval"):
            model.eval()

        info["model"]  = model
        info["on_gpu"] = True
        self.loaded_on_gpu.add(name)
        self._print_vram(f"tras cargar '{name}'")
        return info["model"]

    def offload_model(self, name: str) -> None:
        """
        Mueve un modelo de GPU a CPU liberando VRAM.
        El modelo permanece en RAM para recargarlo rápido.
        """
        if name not in self.registry:
            raise KeyError(f"[ModelRegistry] '{name}' no está registrado.")

        info = self.registry[name]

        if not info["on_gpu"]:
            return

        print(f"[ModelRegistry] Offloading '{name}' GPU → CPU ...")

        if hasattr(info["model"], "to"):
            info["model"] = info["model"].to("cpu")

        info["on_gpu"] = False
        self.loaded_on_gpu.discard(name)

        gc.collect()
        torch.cuda.empty_cache()
        self._print_vram(f"tras offload '{name}'")

    def offload_all(self) -> None:
        """Offloadea todos los modelos en GPU."""
        nombres_en_gpu = list(self.loaded_on_gpu)
        if not nombres_en_gpu:
            print("[ModelRegistry] No hay modelos en GPU.")
            return
        for name in nombres_en_gpu:
            self.offload_model(name)
        print("[ModelRegistry] Todos los modelos offloadeados.")

    # ----------------------------------------------------------
    # Utilidades de inspección
    # ----------------------------------------------------------

    def _print_vram(self, contexto: str = "") -> None:
        """Imprime VRAM real usada por PyTorch."""
        if self.device.type != "cuda":
            return
        usada = torch.cuda.memory_allocated() / 1024**3
        libre = torch.cuda.mem_get_info()[0]  / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        tag   = f" ({contexto})" if contexto else ""
        print(f"[ModelRegistry] VRAM{tag}: "
              f"PyTorch={usada:.2f} GB | libre={libre:.2f} GB / {total:.2f} GB")

    def status(self) -> None:
        """Imprime el estado actual de todos los modelos registrados."""
        if not self.registry:
            print("[ModelRegistry] Sin modelos registrados.")
            return

        vram_en_uso = sum(
            v["vram_gb"] for v in self.registry.values() if v["on_gpu"]
        )
        print(f"\n[ModelRegistry] Estado — VRAM en uso: {vram_en_uso:.1f} GB "
              f"/ límite: {self.vram_limit_gb} GB")
        print(f"  {'Nombre':<14} {'Cargado':<10} {'En GPU':<8} {'VRAM (GB)':<10} {'Checkpoint'}")
        print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*10} {'-'*30}")
        for name, info in self.registry.items():
            cargado = "sí" if info["model"] is not None else "no"
            en_gpu  = "sí" if info["on_gpu"] else "no"
            print(f"  {name:<14} {cargado:<10} {en_gpu:<8} "
                  f"{info['vram_gb']:<10.1f} {info['path'].name}")
        print()

    def is_registered(self, name: str) -> bool:
        return name in self.registry

    def is_loaded(self, name: str) -> bool:
        return (name in self.registry and
                self.registry[name]["model"] is not None)

    def is_on_gpu(self, name: str) -> bool:
        return name in self.loaded_on_gpu

    @property
    def vram_en_uso_gb(self) -> float:
        """VRAM estimada actualmente ocupada por modelos en GPU."""
        return sum(
            v["vram_gb"] for v in self.registry.values() if v["on_gpu"]
        )


# =============================================================
# Bloque de prueba — ejecutar directamente para verificar
# =============================================================

if __name__ == "__main__":
    from pathlib import Path
    from ultralytics import YOLO

    BASE = Path("/mnt/bigdata/pipeline_samsung/Equipo8-Grupo8-SIC-2025-/notebooks")
    CKPT = BASE / "data/spatial_outputs"

    def loader_yolo(path):
        return YOLO(str(path))

    def loader_torch(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    registry = ModelRegistry(vram_limit_gb=5.0)

    print("\n--- Registrando modelos ---")
    registry.register_model("yolo",      BASE / "yolov8n.pt",                                     loader_yolo,  0.6)
    registry.register_model("segformer", CKPT / "segformer_best.pt",                              loader_torch, 1.2)
    registry.register_model("resnet",    CKPT / "resnet_best.pt",                                 loader_torch, 0.3)
    registry.register_model("gcn",       CKPT / "checkpoints_03/gcn_best.pt",                     loader_torch, 0.1)
    registry.register_model("lstm",      CKPT / "checkpoints_03/lstm_best.pt",                    loader_torch, 0.1)
    registry.register_model("bnn",       CKPT / "prediction_outputs/checkpoints_bnn/bnn_best.pt", loader_torch, 0.2)

    print("\n--- Estado inicial (nada cargado) ---")
    registry.status()

    print("--- Cargando 'yolo' ---")
    modelo_yolo = registry.load_model("yolo")
    print(f"Tipo      : {type(modelo_yolo)}")
    print(f"¿en GPU?  : {registry.is_on_gpu('yolo')}")

    print("\n--- Offloading 'yolo' ---")
    registry.offload_model("yolo")
    print(f"¿en GPU tras offload? : {registry.is_on_gpu('yolo')}")
    print(f"¿sigue en RAM?        : {registry.is_loaded('yolo')}")

    print("\n--- Recargando 'yolo' CPU → GPU ---")
    registry.load_model("yolo")
    print(f"¿en GPU? : {registry.is_on_gpu('yolo')}")

    print("\n--- offload_all() ---")
    registry.offload_all()

    print("\n--- Estado final ---")
    registry.status()

    print("[✓] load_model(), offload_model() y offload_all() verificados.")