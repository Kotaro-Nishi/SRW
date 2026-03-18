import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----- ファイルの命名規則とディレクトリ設定 -----
DATA_DIR = "/home/nishi/SRW/env/python/srwpy/examples/data_URI/"
# 例: BeamY_1.000mm_BeamYP_-0.100mrad_ScreenYmin_-20.000mm.txt
FILE_PATTERN = re.compile(r"BeamY_([-+]?\d+\.\d+)mm_BeamYP_([-+]?\d+\.\d+)mrad_ScreenYmin_([-+]?\d+\.\d+)mm\.txt")

# ----- 本来あるべき全パラメータ範囲（グリッド）の設定 -----
# 計算コードの設定と一致させてください
BEAM_Y_GRID = np.arange(0.e-3, 21.e-3, 1.e-3) * 1e3     # mm
BEAM_YP_GRID = np.arange(-2.e-3, 2.e-3, 0.1e-3) * 1e3   # mrad
SCREEN_Y_CONFIG_LIST = [ [-20.e-3+i*2.e-3, -18.e-3+i*2.e-3, 51] for i in range(20)]
SCREEN_Y_MIN_GRID = np.array([config[0] for config in SCREEN_Y_CONFIG_LIST]) * 1e3 # mm

def plot_simulation_tiling_progress():
    # 1. ディレクトリ内の全ファイルをスキャンしてパラメータを抽出
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    files = os.listdir(DATA_DIR)
    completed_set = set() # 高速検索のために set を使用

    for f in files:
        match = FILE_PATTERN.search(f)
        if match:
            # (y, yp, ymin) のタプルとして保存 (単位はmm, mrad)
            y, yp, ymin = [float(group) for group in match.groups()]
            # 小数点以下の精度による不一致を避けるため、文字列に変換してキーにする
            completed_set.add(f"{y:.3f}_{yp:.3f}_{ymin:.3f}")

    if not completed_set:
        print("計算済みのファイルが見つかりません。")
        #return # 全て未計算として描画を続行

    # 2. サブプロットのレイアウト決定（ScreenYmin ごとにパネルを作成）
    num_panels = len(SCREEN_Y_MIN_GRID)
    cols = 5
    rows = (num_panels // cols) + (1 if num_panels % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    # タイルのサイズ（グリッド間隔）を計算
    dy = BEAM_Y_GRID[1] - BEAM_Y_GRID[0] if len(BEAM_Y_GRID) > 1 else 1.0
    dyp = BEAM_YP_GRID[1] - BEAM_YP_GRID[0] if len(BEAM_YP_GRID) > 1 else 0.1

    print("全タスク数:", len(BEAM_Y_GRID) * len(BEAM_YP_GRID) * len(SCREEN_Y_MIN_GRID))
    print("完了タスク数:", len(completed_set))

    # 3. 各 ScreenYmin パネルごとにグリッドをスキャンして描画
    for i, ymin in enumerate(np.sort(SCREEN_Y_MIN_GRID)):
        ax = axes[i]
        
        # パネル内の全グリッド点をループ
        for y in BEAM_Y_GRID:
            for yp in BEAM_YP_GRID:
                # 計算済みか確認
                key = f"{y:.3f}_{yp:.3f}_{ymin:.3f}"
                is_completed = key in completed_set
                
                # タイル（四角形）の色を決定
                # 完了: 緑 (green), 未完了: 赤 (red)
                color = 'green' if is_completed else 'red'
                
                # 四角形（Patch）を作成して追加
                # (yp - dyp/2, y - dy/2) が左下の座標
                rect = patches.Rectangle(
                    (yp - dyp/2, y - dy/2), 
                    dyp, dy, 
                    linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.8
                )
                ax.add_patch(rect)

        # 軸の設定
        ax.set_xlim(BEAM_YP_GRID.min() - dyp, BEAM_YP_GRID.max() + dyp)
        ax.set_ylim(BEAM_Y_GRID.min() - dy, BEAM_Y_GRID.max() + dy)
        ax.set_title(f"ScreenYmin: {ymin:.1f} mm")
        ax.set_xlabel("BeamYP [mrad]")
        ax.set_ylabel("BeamY [mm]")
        ax.grid(True, linestyle='--', alpha=0.3)

    # 4. 余ったサブプロットを消す
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 凡例用のダミープロット
    green_patch = patches.Patch(color='green', label='Completed', alpha=0.8)
    red_patch = patches.Patch(color='red', label='Remaining', alpha=0.8)
    fig.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=12)

    plt.suptitle(f"SRW Simulation Progress Tiling (Total: {len(completed_set)}/{len(BEAM_Y_GRID)*len(BEAM_YP_GRID)*len(SCREEN_Y_MIN_GRID)})", fontsize=20)
    plt.show()

if __name__ == "__main__":
    plot_simulation_tiling_progress()