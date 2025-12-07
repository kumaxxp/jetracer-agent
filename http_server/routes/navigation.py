"""ナビゲーション用API - 状況分析"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Set
import numpy as np
import time

from ..core.camera_manager import camera_manager
from ..core.road_mapping import get_road_mapping

router = APIRouter()

# OneFormerの結果を共有（oneformer.pyからインポート）
from .oneformer import (
    _latest_seg_masks,
    get_segmenter,
    get_road_label_ids_from_segmenter,
    run_oneformer_internal  # 内部関数を使用
)


def analyze_road_distribution(seg_mask: np.ndarray, road_label_ids: Set[int]) -> dict:
    """
    セグメンテーションマスクからROAD領域の分布を分析
    
    Args:
        seg_mask: セグメンテーションマスク (H, W)
        road_label_ids: ROAD属性を持つラベルIDのセット
    
    Returns:
        分析結果
    """
    h, w = seg_mask.shape
    
    # ROADマスクを作成
    road_mask = np.zeros_like(seg_mask, dtype=bool)
    for label_id in road_label_ids:
        road_mask |= (seg_mask == label_id)
    
    # 全体のROAD比率
    total_road_ratio = float(road_mask.sum() / road_mask.size)
    
    # グリッド分析（3行 x 3列）
    # 行: far(遠方), mid(中間), near(足元)
    # 列: left, center, right
    grid = analyze_road_grid(road_mask, rows=3, cols=3)
    
    # 左/中央/右の分割（3等分）- 従来互換用
    left_region = road_mask[:, :w//3]
    center_region = road_mask[:, w//3:2*w//3]
    right_region = road_mask[:, 2*w//3:]
    
    left_ratio = float(left_region.sum() / left_region.size)
    center_ratio = float(center_region.sum() / center_region.size)
    right_ratio = float(right_region.sum() / right_region.size)
    
    # 上部/下部の分割（前方の壁検出用）
    top_region = road_mask[:h//3, :]
    bottom_region = road_mask[2*h//3:, :]
    
    top_ratio = float(top_region.sum() / top_region.size)
    bottom_ratio = float(bottom_region.sum() / bottom_region.size)
    
    # 前方に壁があるか（上部のROAD比率が低い場合）
    # 閾値: 50%未満なら前方に障害物があると判定
    wall_ahead = bool(top_ratio < 0.5)
    
    # 左右の境界検出（端にROADがない場合）
    left_edge = road_mask[:, :w//10]  # 左端10%
    right_edge = road_mask[:, -w//10:]  # 右端10%
    
    left_boundary = bool(left_edge.sum() / left_edge.size < 0.3)
    right_boundary = bool(right_edge.sum() / right_edge.size < 0.3)
    
    # 主要な道の方向
    if center_ratio > left_ratio and center_ratio > right_ratio:
        dominant_direction = "center"
    elif left_ratio > right_ratio:
        dominant_direction = "left"
    else:
        dominant_direction = "right"
    
    # 走行可能パスの判定
    path_analysis = analyze_path(grid)
    
    return {
        "road_ratio": {
            "left": round(left_ratio, 3),
            "center": round(center_ratio, 3),
            "right": round(right_ratio, 3),
            "total": round(total_road_ratio, 3),
            "top": round(top_ratio, 3),
            "bottom": round(bottom_ratio, 3)
        },
        "grid": grid,  # 3x3グリッド
        "path": path_analysis,  # 走行可能パス
        "wall_ahead": wall_ahead,
        "boundary": {
            "left": left_boundary,
            "right": right_boundary
        },
        "dominant_direction": dominant_direction
    }


def analyze_road_grid(road_mask: np.ndarray, rows: int = 3, cols: int = 3) -> dict:
    """
    グリッドベースのROAD分析
    
    3x3グリッド:
    - 行: far(遠方/上), mid(中間), near(足元/下)
    - 列: left, center, right
    
    Returns:
        {"far": {"left": 0.1, "center": 0.3, "right": 0.1},
         "mid": {"left": 0.4, "center": 0.8, "right": 0.5},
         "near": {"left": 0.7, "center": 0.9, "right": 0.8}}
    """
    h, w = road_mask.shape
    row_height = h // rows
    col_width = w // cols
    
    row_names = ["far", "mid", "near"]
    col_names = ["left", "center", "right"]
    
    grid = {}
    for r, row_name in enumerate(row_names):
        row_data = {}
        for c, col_name in enumerate(col_names):
            y1, y2 = r * row_height, (r + 1) * row_height
            x1, x2 = c * col_width, (c + 1) * col_width
            
            cell = road_mask[y1:y2, x1:x2]
            ratio = float(cell.sum() / cell.size)
            row_data[col_name] = round(ratio, 2)
        grid[row_name] = row_data
    
    return grid


def analyze_path(grid: dict) -> dict:
    """
    グリッドから走行可能パスを分析
    
    Returns:
        {"can_go_straight": True/False,
         "can_go_left": True/False,
         "can_go_right": True/False,
         "best_direction": "center"/"left"/"right"/"none",
         "recommended_action": "forward"/"turn_left"/"turn_right"/"stop"}
    """
    # 閾値: 40%以上のROADがあれば通行可能
    THRESHOLD = 0.4
    
    # 各方向の通行可能性を判定
    # 直進: far-centerとmid-centerが両方通行可能
    can_go_straight = (
        grid["far"]["center"] >= THRESHOLD and 
        grid["mid"]["center"] >= THRESHOLD
    )
    
    # 左折: far-leftとmid-leftが通行可能
    can_go_left = (
        grid["far"]["left"] >= THRESHOLD and 
        grid["mid"]["left"] >= THRESHOLD
    )
    
    # 右折: far-rightとmid-rightが通行可能
    can_go_right = (
        grid["far"]["right"] >= THRESHOLD and 
        grid["mid"]["right"] >= THRESHOLD
    )
    
    # 最適方向の決定
    directions = {
        "center": grid["far"]["center"] + grid["mid"]["center"],
        "left": grid["far"]["left"] + grid["mid"]["left"],
        "right": grid["far"]["right"] + grid["mid"]["right"]
    }
    
    best_direction = max(directions, key=directions.get)
    best_score = directions[best_direction]
    
    # 推奨アクション
    if best_score < THRESHOLD * 2:  # どの方向も通行困難
        recommended_action = "stop"
        best_direction = "none"
    elif can_go_straight and best_direction == "center":
        recommended_action = "forward"
    elif can_go_left and best_direction == "left":
        recommended_action = "turn_left"
    elif can_go_right and best_direction == "right":
        recommended_action = "turn_right"
    elif can_go_straight:
        recommended_action = "forward"
    elif can_go_left:
        recommended_action = "turn_left"
    elif can_go_right:
        recommended_action = "turn_right"
    else:
        recommended_action = "stop"
        best_direction = "none"
    
    return {
        "can_go_straight": can_go_straight,
        "can_go_left": can_go_left,
        "can_go_right": can_go_right,
        "best_direction": best_direction,
        "recommended_action": recommended_action
    }


def generate_description(analysis: dict, camera_name: str) -> str:
    """分析結果から説明文を生成"""
    parts = []
    
    grid = analysis.get("grid", {})
    path = analysis.get("path", {})
    road = analysis["road_ratio"]
    
    # パス分析に基づく説明
    if path:
        action = path.get("recommended_action", "stop")
        if action == "forward":
            parts.append("前進可能")
        elif action == "turn_left":
            parts.append("左折推奨")
        elif action == "turn_right":
            parts.append("右折推奨")
        elif action == "stop":
            parts.append("停止推奨")
        
        # 通行可能方向の詳細
        directions = []
        if path.get("can_go_straight"):
            directions.append("直進○")
        else:
            directions.append("直進×")
        if path.get("can_go_left"):
            directions.append("左○")
        else:
            directions.append("左×")
        if path.get("can_go_right"):
            directions.append("右○")
        else:
            directions.append("右×")
        parts.append(f"[{' '.join(directions)}]")
    
    # グリッド情報
    if grid:
        far = grid.get("far", {})
        # 遠方の状況
        far_center = far.get("center", 0)
        if far_center < 0.2:
            parts.append("前方障害物")
        elif far_center >= 0.6:
            parts.append("前方開けている")
    
    # 境界
    if analysis["boundary"]["left"]:
        parts.append("左壁")
    if analysis["boundary"]["right"]:
        parts.append("右壁")
    
    if not parts:
        parts.append("通常の道")
    
    return "、".join(parts)


@router.get("/navigation/situation")
def get_situation():
    """
    両カメラの状況を分析してLLM用のデータを返す
    
    OneFormerのセグメンテーション結果を使用して、
    走行判断に必要な数値データと説明を生成します。
    """
    start_time = time.time()
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "front_camera": None,
        "ground_camera": None,
        "summary": "",
        "process_time_ms": 0
    }
    
    # セグメンテーションデータがある場合のみROADラベルIDを取得
    has_any_segmentation = 0 in _latest_seg_masks or 1 in _latest_seg_masks
    
    road_label_ids = set()
    if has_any_segmentation:
        try:
            road_label_ids = get_road_label_ids_from_segmenter()
        except Exception as e:
            print(f"[Navigation] Failed to get road labels: {e}")
            # エラーでも継続（空のroad_label_idsで）
    
    # 正面カメラ (camera_id=0)
    if 0 in _latest_seg_masks:
        front_analysis = analyze_road_distribution(
            _latest_seg_masks[0], road_label_ids
        )
        front_analysis["description"] = generate_description(front_analysis, "正面")
        result["front_camera"] = front_analysis
    else:
        result["front_camera"] = {
            "error": "No segmentation data. Run /oneformer/0 first.",
            "road_ratio": {"left": 0, "center": 0, "right": 0, "total": 0, "top": 0, "bottom": 0},
            "grid": {
                "far": {"left": 0, "center": 0, "right": 0},
                "mid": {"left": 0, "center": 0, "right": 0},
                "near": {"left": 0, "center": 0, "right": 0}
            },
            "path": {
                "can_go_straight": False,
                "can_go_left": False,
                "can_go_right": False,
                "best_direction": "none",
                "recommended_action": "stop"
            },
            "wall_ahead": True,
            "boundary": {"left": False, "right": False},
            "dominant_direction": "unknown",
            "description": "セグメンテーション未実行"
        }
    
    # 足元カメラ (camera_id=1)
    if 1 in _latest_seg_masks:
        ground_analysis = analyze_road_distribution(
            _latest_seg_masks[1], road_label_ids
        )
        ground_analysis["description"] = generate_description(ground_analysis, "足元")
        result["ground_camera"] = ground_analysis
    else:
        result["ground_camera"] = {
            "error": "No segmentation data. Run /oneformer/1 first.",
            "road_ratio": {"left": 0, "center": 0, "right": 0, "total": 0, "top": 0, "bottom": 0},
            "grid": {
                "far": {"left": 0, "center": 0, "right": 0},
                "mid": {"left": 0, "center": 0, "right": 0},
                "near": {"left": 0, "center": 0, "right": 0}
            },
            "path": {
                "can_go_straight": False,
                "can_go_left": False,
                "can_go_right": False,
                "best_direction": "none",
                "recommended_action": "stop"
            },
            "wall_ahead": True,
            "boundary": {"left": False, "right": False},
            "dominant_direction": "unknown",
            "description": "セグメンテーション未実行"
        }
    
    # サマリー生成
    summaries = []
    if result["front_camera"] and "description" in result["front_camera"]:
        summaries.append(f"正面: {result['front_camera']['description']}")
    if result["ground_camera"] and "description" in result["ground_camera"]:
        summaries.append(f"足元: {result['ground_camera']['description']}")
    
    result["summary"] = " / ".join(summaries) if summaries else "状況不明"
    result["process_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return result


@router.post("/navigation/update-situation")
def update_situation(run_segmentation: bool = True):
    """
    状況を更新（セグメンテーションを実行してから状況を取得）
    
    Args:
        run_segmentation: セグメンテーションを実行するか
    """
    start_time = time.time()
    
    segmentation_results = {}
    
    if run_segmentation:
        # 両カメラでセグメンテーション実行
        for camera_id in [0, 1]:
            try:
                print(f"[Navigation] Running segmentation for camera {camera_id}...")
                seg_result = run_oneformer_internal(camera_id=camera_id, highlight_road=True)
                segmentation_results[camera_id] = {
                    "success": True,
                    "process_time_ms": seg_result.get("process_time_ms", 0),
                    "num_classes": seg_result.get("num_classes", 0)
                }
                print(f"[Navigation] Camera {camera_id} segmentation done: {seg_result.get('process_time_ms', 0):.0f}ms")
            except HTTPException as e:
                print(f"[Navigation] Camera {camera_id} HTTPException: {e.detail}")
                segmentation_results[camera_id] = {
                    "success": False,
                    "error": e.detail
                }
            except Exception as e:
                print(f"[Navigation] Camera {camera_id} Exception: {e}")
                import traceback
                traceback.print_exc()
                segmentation_results[camera_id] = {
                    "success": False,
                    "error": str(e)
                }
    
    # 状況取得
    try:
        situation = get_situation()
    except HTTPException as e:
        situation = {
            "error": e.detail,
            "front_camera": None,
            "ground_camera": None,
            "summary": f"エラー: {e.detail}"
        }
    except Exception as e:
        print(f"[Navigation] get_situation Exception: {e}")
        import traceback
        traceback.print_exc()
        situation = {
            "error": str(e),
            "front_camera": None,
            "ground_camera": None,
            "summary": f"エラー: {e}"
        }
    
    return {
        "segmentation": segmentation_results,
        "situation": situation,
        "total_time_ms": round((time.time() - start_time) * 1000, 2)
    }


class MoveCommand(BaseModel):
    """移動コマンド"""
    action: str  # forward, backward, turn_left, turn_right, spin, stop
    distance_cm: Optional[float] = None  # forward/backward用
    angle_deg: Optional[float] = None  # turn用
    direction: Optional[str] = None  # spin用 (left/right)


@router.post("/navigation/move")
def execute_move(command: MoveCommand):
    """
    移動コマンドを実行（現在はモック）
    
    実際のモーター制御は後で実装します。
    現在はコマンドを受け付けて成功を返すだけです。
    """
    print(f"[Navigation] MOCK Execute: {command.action}")
    print(f"[Navigation]   params: distance={command.distance_cm}, angle={command.angle_deg}, direction={command.direction}")
    
    # モック実行結果
    return {
        "success": True,
        "mock": True,
        "message": f"MOCK: Would execute {command.action}",
        "command": {
            "action": command.action,
            "distance_cm": command.distance_cm,
            "angle_deg": command.angle_deg,
            "direction": command.direction
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/navigation/status")
def get_navigation_status():
    """ナビゲーションの状態を取得"""
    return {
        "state": "idle",  # idle, moving, error
        "has_front_segmentation": 0 in _latest_seg_masks,
        "has_ground_segmentation": 1 in _latest_seg_masks,
        "motor_connected": False,  # 後で実装
        "timestamp": datetime.now().isoformat()
    }
