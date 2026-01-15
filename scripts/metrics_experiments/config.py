"""
Configuration centralisée pour les expériences SAM
"""
from pathlib import Path

class Config:
    """Configuration des paramètres d'expérimentation"""
    
    # Chemins
    TIF_PATH = '../tests/Romane_Martin_urne_sature_10-4.tif'
    SAM_WEIGHTS = "sam_b.pt"
    YOLO_WEIGHTS = "yolo11n.pt"
    OUTPUT_DIR = "experiments_results"
    
    # Paramètres SAM
    GRID_STRIDE = 64
    POINTS_PER_CALL = 30
    MIN_AREA = 300
    CONF_THR = 0.5
    DEDUP_IOU_THR = 0.90
    MIN_MASK_REGION_AREA = 200
    POINT_LABEL = 1
    MAX_AREA_RATIO = 0.05
    
    # Paramètres modèle
    DEVICE = "cuda"
    IMGSZ = 640
    IOU = 0.25
    MAX_DET = 100
    
    # Paramètres prétraitement
    URNA_THRESHOLD = 60
    USE_CLAHE = True
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Paramètres expériences
    SLICE_RANGE = 10  # Nombre de slices à analyser pour IoU
    DELTA_Z_VALUES = [1, 2, 3, 5, 7, 10]
    FUSION_MODES = ['mean', 'max', 'median']
    
    @classmethod
    def get_sam_params(cls):
        """Retourne les paramètres SAM par défaut"""
        return {
            'device': cls.DEVICE,
            'imgsz': cls.IMGSZ,
            'iou': cls.IOU,
            'max_det': cls.MAX_DET,
            'conf': cls.CONF_THR,
            'min_area': cls.MIN_AREA,
            'max_area_ratio': cls.MAX_AREA_RATIO,
            'grid_stride': cls.GRID_STRIDE
        }
