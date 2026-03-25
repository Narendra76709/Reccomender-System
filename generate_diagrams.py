import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.patheffects as pe
import numpy as np
import os

os.makedirs(r"D:\Major\movie-recommender\diagrams", exist_ok=True)

BG = '#0d1b2a'
FONT = 'DejaVu Sans'

# ─────────────────────────────────────────────────────────
# Diagram 1 – SDLC
# ─────────────────────────────────────────────────────────
def draw_sdlc():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off')

    ax.text(0, 1.05, 'Software Development Life Cycle',
            color='white', fontsize=22, fontweight='bold',
            ha='center', va='center', fontfamily=FONT)
    ax.text(0, 0.96, 'Hybrid Two-Stage Movie Recommender System',
            color='#a0c4ff', fontsize=13, ha='center', va='center', fontfamily=FONT)

    phases = [
        ('Requirements', '#264653',
         ['Gather MovieLens 25M dataset',
          'Define P@K, NDCG@K metrics',
          'Identify user tiers (cold/light/medium/power)',
          'Define API requirements']),
        ('System Design', '#2a9d8f',
         ['Architecture: two-stage pipeline',
          'Generator selection (NMF, SVD, Pop)',
          'Feature engineering plan (44 features)',
          'LightGBM reranker design']),
        ('Implementation', '#e9c46a',
         ['Code NMF-ANN & SVD-ANN generators',
          'Bayesian Popularity generator',
          'Weighted RRF Fusion (45/40/15%)',
          'FastAPI endpoints + frontend']),
        ('Testing', '#f4a261',
         ['Offline eval: P@K, R@K, NDCG@K',
          'HR@K, MRR evaluation',
          'Unit tests per module',
          'API endpoint testing']),
        ('Deployment', '#e76f51',
         ['FastAPI server (port 8000)',
          'Web UI (index.html)',
          'Model artifacts (.pkl, .ann)',
          'Docker / production setup']),
        ('Maintenance', '#6d6875',
         ['Retrain pipeline automation',
          '/api/train endpoint',
          'Model performance monitoring',
          'Data drift detection']),
    ]

    n = len(phases)
    radius = 0.68
    box_w, box_h = 0.38, 0.30

    angles = [np.pi / 2 - i * 2 * np.pi / n for i in range(n)]
    centers = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    # Draw curved arrows between boxes
    for i in range(n):
        x0, y0 = centers[i]
        x1, y1 = centers[(i + 1) % n]
        ax.annotate('',
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#a0c4ff',
                                   lw=2.0, connectionstyle='arc3,rad=0.18'))

    # Draw center circle label
    circle = plt.Circle((0, 0), 0.24, color='#1a1a2e', zorder=3,
                         linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.text(0, 0.04, 'SDLC', color='#a0c4ff', fontsize=16,
            fontweight='bold', ha='center', va='center', zorder=4)
    ax.text(0, -0.06, 'Movie\nRecommender', color='#7ecfff', fontsize=9,
            ha='center', va='center', zorder=4)

    for i, (title, color, bullets) in enumerate(phases):
        cx, cy = centers[i]

        fancy = FancyBboxPatch((cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                               boxstyle='round,pad=0.02',
                               facecolor=color, edgecolor='white',
                               linewidth=1.5, zorder=5, alpha=0.92)
        ax.add_patch(fancy)

        ax.text(cx, cy + box_h / 2 - 0.045, title,
                color='white', fontsize=11, fontweight='bold',
                ha='center', va='center', zorder=6, fontfamily=FONT)

        for j, bullet in enumerate(bullets):
            ax.text(cx, cy + box_h / 2 - 0.095 - j * 0.055,
                    f'• {bullet}',
                    color='#f0f0f0', fontsize=6.8,
                    ha='center', va='center', zorder=6, fontfamily=FONT)

    plt.tight_layout()
    out = r'D:\Major\movie-recommender\diagrams\sdlc.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Saved: {out}  ({os.path.getsize(out):,} bytes)')


# ─────────────────────────────────────────────────────────
# Diagram 2 – Sequence Diagram
# ─────────────────────────────────────────────────────────
def draw_sequence():
    fig, ax = plt.subplots(figsize=(20, 11), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    participants = [
        'User\n(Browser)',
        'Frontend\n(index.html)',
        'FastAPI\n(api.py)',
        'Hybrid\nRecommender',
        'Stage 1\nGenerators',
        'RRF\nFusion',
        'LightGBM\nReranker',
        'MovieLens\nData',
    ]
    colors = ['#4cc9f0', '#4361ee', '#7209b7', '#f72585',
              '#3a0ca3', '#480ca8', '#b5179e', '#06d6a0']

    n = len(participants)
    xs = np.linspace(0.04, 0.96, n)
    header_y = 0.93
    lifeline_top = 0.88
    lifeline_bot = 0.04

    # Headers
    for i, (p, c) in enumerate(zip(participants, colors)):
        bbox = FancyBboxPatch((xs[i] - 0.055, header_y - 0.04), 0.11, 0.08,
                              boxstyle='round,pad=0.01',
                              facecolor=c, edgecolor='white', linewidth=1.2,
                              transform=ax.transAxes, zorder=5)
        ax.add_patch(bbox)
        ax.text(xs[i], header_y, p, color='white', fontsize=8.5,
                ha='center', va='center', fontweight='bold',
                transform=ax.transAxes, zorder=6, fontfamily=FONT)

    # Lifelines
    for i, c in enumerate(colors):
        ax.plot([xs[i], xs[i]], [lifeline_top - 0.005, lifeline_bot],
                color=c, lw=1.2, linestyle='--', alpha=0.55,
                transform=ax.transAxes, zorder=2)

    messages = [
        (0, 1, 'Enter user_id, click Recommend', False),
        (1, 2, 'GET /api/recommend?user_id=X', False),
        (2, 3, 'get_recommendations(user_id)', False),
        (3, 7, 'load user ratings', False),
        (7, 3, 'ratings DataFrame', True),
        (3, 4, 'classify tier → run generators', False),
        (4, 5, 'candidate lists (top-100 each)', False),
        (5, 3, 'top-200 fused candidates', True),
        (3, 6, 'extract 44 features, predict()', False),
        (6, 3, 'ranked scores', True),
        (3, 2, 'top-10 movies', True),
        (2, 1, 'JSON response', True),
        (1, 0, 'Display movie cards', True),
    ]

    total = len(messages)
    ys = np.linspace(lifeline_top - 0.04, lifeline_bot + 0.02, total)

    act_ranges = {i: [] for i in range(n)}
    for step_i, (src, dst, label, is_return) in enumerate(messages):
        act_ranges[src].append(ys[step_i])
        act_ranges[dst].append(ys[step_i])

    # Activation boxes
    act_color = {i: c for i, c in enumerate(colors)}
    for i in range(n):
        if act_ranges[i]:
            y_min = min(act_ranges[i]) - 0.015
            y_max = max(act_ranges[i]) + 0.015
            rect = FancyBboxPatch((xs[i] - 0.008, y_min), 0.016, y_max - y_min,
                                  boxstyle='square,pad=0',
                                  facecolor=act_color[i], alpha=0.35,
                                  edgecolor=act_color[i], linewidth=1,
                                  transform=ax.transAxes, zorder=3)
            ax.add_patch(rect)

    for step_i, (src, dst, label, is_return) in enumerate(messages):
        y = ys[step_i]
        x0, x1 = xs[src], xs[dst]
        style = 'dashed' if is_return else 'solid'
        clr = '#a8dadc' if is_return else '#ffffff'
        ax.annotate('',
                    xy=(x1, y), xytext=(x0, y),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=clr, lw=1.5,
                                   linestyle=style))
        mid_x = (x0 + x1) / 2
        offset = 0.012
        ax.text(mid_x, y + offset, label,
                color=clr, fontsize=7.8, ha='center', va='bottom',
                transform=ax.transAxes, fontfamily=FONT,
                bbox=dict(boxstyle='round,pad=0.1', facecolor=BG,
                          edgecolor='none', alpha=0.7))
        ax.text(xs[src] - 0.005 if xs[src] < xs[dst] else xs[src] + 0.005,
                y, str(step_i + 1),
                color='#ffd166', fontsize=6.5, ha='center', va='center',
                transform=ax.transAxes, zorder=7)

    ax.text(0.5, 0.985, 'Sequence Diagram — Movie Recommender System',
            color='white', fontsize=18, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes, fontfamily=FONT)

    out = r'D:\Major\movie-recommender\diagrams\sequence.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Saved: {out}  ({os.path.getsize(out):,} bytes)')


# ─────────────────────────────────────────────────────────
# Diagram 3 – Activity Diagram
# ─────────────────────────────────────────────────────────
def draw_activity():
    fig, ax = plt.subplots(figsize=(14, 20), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 26)
    ax.axis('off')

    ax.text(5, 25.4, 'Activity Diagram — Recommendation Workflow',
            color='white', fontsize=18, fontweight='bold',
            ha='center', va='center', fontfamily=FONT)

    def action_box(cx, cy, w, h, text, color, fontsize=10):
        box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                             boxstyle='round,pad=0.15',
                             facecolor=color, edgecolor='white',
                             linewidth=1.5, zorder=4)
        ax.add_patch(box)
        ax.text(cx, cy, text, color='white', fontsize=fontsize,
                ha='center', va='center', fontweight='bold',
                fontfamily=FONT, zorder=5,
                multialignment='center')

    def diamond(cx, cy, w, h, text, color='#f4a261'):
        dx = w / 2
        dy = h / 2
        xs = [cx, cx + dx, cx, cx - dx, cx]
        ys = [cy + dy, cy, cy - dy, cy, cy + dy]
        ax.fill(xs, ys, color=color, zorder=4, alpha=0.9)
        ax.plot(xs, ys, color='white', lw=1.5, zorder=5)
        ax.text(cx, cy, text, color='white', fontsize=9.5,
                ha='center', va='center', fontweight='bold',
                fontfamily=FONT, zorder=6)

    def start_end(cx, cy, r, color):
        c = plt.Circle((cx, cy), r, color=color, zorder=4)
        ax.add_patch(c)
        if color == 'white':
            c2 = plt.Circle((cx, cy), r * 0.6, color=BG, zorder=5)
            ax.add_patch(c2)

    def arrow(x0, y0, x1, y1, color='#a0c4ff', label='', lw=1.8):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx + 0.15, my, label, color=color, fontsize=9,
                    va='center', fontfamily=FONT)

    # Start
    start_end(5, 24.5, 0.3, '#4cc9f0')
    arrow(5, 24.2, 5, 23.55)

    # Step 1
    action_box(5, 23.1, 6.5, 0.8, 'Load user ratings from MovieLens 25M', '#264653')
    arrow(5, 22.7, 5, 22.05)

    # Step 2
    action_box(5, 21.6, 5.5, 0.8, 'Classify User Tier', '#2a9d8f')
    arrow(5, 21.2, 5, 20.4)

    # Decision diamond
    diamond(5, 19.8, 3.2, 1.1, 'User Tier?')

    # Branches
    # Cold branch (left)
    ax.annotate('', xy=(1.5, 18.0), xytext=(3.4, 19.8),
                arrowprops=dict(arrowstyle='->', color='#ef476f', lw=1.8))
    ax.text(2.2, 19.2, 'cold (≤5)', color='#ef476f', fontsize=9,
            fontweight='bold', fontfamily=FONT)
    action_box(1.5, 17.4, 2.6, 0.9, 'Popularity\nGenerator only', '#e63946', fontsize=9)

    # Light branch (center-left)
    ax.annotate('', xy=(3.8, 17.5), xytext=(4.4, 18.75),
                arrowprops=dict(arrowstyle='->', color='#ffd166', lw=1.8))
    ax.text(3.6, 18.4, 'light\n(6–20)', color='#ffd166', fontsize=8.5,
            fontweight='bold', ha='center', fontfamily=FONT)
    action_box(3.8, 16.9, 2.8, 0.9, 'SVD-ANN +\nPopularity', '#e9c46a', fontsize=9)

    # Medium/Power branch (right)
    ax.annotate('', xy=(7.8, 17.5), xytext=(6.6, 18.75),
                arrowprops=dict(arrowstyle='->', color='#06d6a0', lw=1.8))
    ax.text(7.5, 18.5, 'medium/power\n(>20)', color='#06d6a0', fontsize=8.5,
            fontweight='bold', ha='center', fontfamily=FONT)
    action_box(7.8, 16.9, 2.9, 0.9, 'NMF-ANN + SVD-ANN\n+ Popularity', '#118ab2', fontsize=9)

    # Merge arrows to step below
    for bx, by in [(1.5, 16.45), (3.8, 16.45), (7.8, 16.45)]:
        ax.plot([bx, bx], [by, 15.85], color='#a0c4ff', lw=1.8)
    ax.plot([1.5, 7.8], [15.85, 15.85], color='#a0c4ff', lw=1.8)
    arrow(5, 15.85, 5, 15.2)

    action_box(5, 14.75, 7.0, 0.8,
               'All active generators produce top-100 candidates each', '#6d6875')
    arrow(5, 14.35, 5, 13.7)

    action_box(5, 13.25, 7.0, 0.8,
               'Weighted RRF Fusion  (NMF 45% | SVD 40% | Pop 15%, k=60)', '#7209b7')
    arrow(5, 12.85, 5, 12.2)

    action_box(5, 11.75, 5.5, 0.8, 'Select top-200 fused candidates', '#480ca8')
    arrow(5, 11.35, 5, 10.7)

    action_box(5, 10.25, 7.5, 0.8,
               'Extract 44 features per (user, movie) pair', '#3a0ca3')
    # Feature sub-boxes
    sub_labels = ['User Stats (5)', 'Movie Stats (3)', 'Retrieval Scores (6)', 'Genome PCA (30)']
    sub_colors = ['#4cc9f0', '#4361ee', '#7209b7', '#f72585']
    sub_xs = [2.0, 3.9, 6.1, 8.0]
    for sx, slabel, sc in zip(sub_xs, sub_labels, sub_colors):
        action_box(sx, 9.35, 1.7, 0.55, slabel, sc, fontsize=8)
    for sx in sub_xs:
        ax.plot([5, sx], [9.85, 9.62], color='#a0c4ff', lw=1.2, linestyle='--')
    # Merge from sub-boxes
    for sx in sub_xs:
        ax.plot([sx, sx], [9.08, 8.75], color='#a0c4ff', lw=1.2)
    ax.plot([2.0, 8.0], [8.75, 8.75], color='#a0c4ff', lw=1.2)
    arrow(5, 8.75, 5, 8.1)

    action_box(5, 7.65, 6.5, 0.8,
               'LightGBM predict interaction probability', '#b5179e')
    arrow(5, 7.25, 5, 6.6)

    action_box(5, 6.15, 5.0, 0.8, 'Sort by predicted score', '#06d6a0', fontsize=10)
    arrow(5, 5.75, 5, 5.1)

    action_box(5, 4.65, 5.5, 0.8, 'Return top-10 recommendations', '#2a9d8f')
    arrow(5, 4.25, 5, 3.6)

    # End
    start_end(5, 3.2, 0.3, '#4cc9f0')
    inner = plt.Circle((5, 3.2), 0.18, color='#0d1b2a', zorder=5)
    ax.add_patch(inner)

    plt.tight_layout(pad=0.5)
    out = r'D:\Major\movie-recommender\diagrams\activity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Saved: {out}  ({os.path.getsize(out):,} bytes)')


# ─────────────────────────────────────────────────────────
# Diagram 4 – Use Case Diagram
# ─────────────────────────────────────────────────────────
def draw_usecase():
    fig, ax = plt.subplots(figsize=(18, 11), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')

    ax.text(9, 10.55, 'Use Case Diagram — Movie Recommender System',
            color='white', fontsize=20, fontweight='bold',
            ha='center', va='center', fontfamily=FONT)

    # System boundary
    sys_rect = FancyBboxPatch((2.5, 0.5), 13.0, 9.5,
                              boxstyle='round,pad=0.1',
                              facecolor='#111827', edgecolor='#4cc9f0',
                              linewidth=2.5, zorder=1)
    ax.add_patch(sys_rect)
    ax.text(9, 9.7, 'Movie Recommender System', color='#4cc9f0',
            fontsize=13, ha='center', va='center', fontweight='bold',
            fontfamily=FONT)

    def stick_figure(cx, cy, label, color):
        # head
        head = plt.Circle((cx, cy + 0.7), 0.22, color=color,
                           fill=False, linewidth=2, zorder=6)
        ax.add_patch(head)
        # body
        ax.plot([cx, cx], [cy + 0.48, cy - 0.2], color=color, lw=2, zorder=6)
        # arms
        ax.plot([cx - 0.35, cx + 0.35], [cy + 0.15, cy + 0.15],
                color=color, lw=2, zorder=6)
        # legs
        ax.plot([cx, cx - 0.3], [cy - 0.2, cy - 0.7], color=color, lw=2, zorder=6)
        ax.plot([cx, cx + 0.3], [cy - 0.2, cy - 0.7], color=color, lw=2, zorder=6)
        ax.text(cx, cy - 0.95, label, color=color, fontsize=11,
                ha='center', va='top', fontweight='bold', fontfamily=FONT)

    stick_figure(1.3, 5.5, 'User', '#4cc9f0')
    stick_figure(16.7, 5.5, 'Admin', '#f72585')

    def oval(cx, cy, w, h, text, color, fontsize=9):
        ell = mpatches.Ellipse((cx, cy), w, h, facecolor=color,
                               edgecolor='white', linewidth=1.5, zorder=4)
        ax.add_patch(ell)
        ax.text(cx, cy, text, color='white', fontsize=fontsize,
                ha='center', va='center', fontweight='bold',
                fontfamily=FONT, zorder=5, multialignment='center')
        return (cx, cy)

    # User use cases
    uc_user = [
        oval(5.5, 8.5, 3.5, 0.8, 'Get Movie Recommendations', '#264653'),
        oval(5.5, 7.2, 3.5, 0.8, 'Search Movies by Title', '#2a9d8f'),
        oval(5.5, 5.9, 3.0, 0.8, 'View Popular Movies', '#e9c46a'),
        oval(5.5, 4.6, 3.0, 0.8, 'View Movie Details', '#f4a261'),
        oval(5.5, 3.3, 2.8, 0.8, 'Filter by Genre', '#e76f51'),
    ]

    # Admin use cases
    uc_admin = [
        oval(12.5, 8.5, 3.2, 0.8, 'Train NMF-ANN Model', '#7209b7'),
        oval(12.5, 7.2, 3.2, 0.8, 'Train SVD-ANN Model', '#560bad'),
        oval(12.5, 5.9, 3.2, 0.8, 'Train LightGBM Reranker', '#480ca8'),
        oval(12.5, 4.6, 3.5, 0.8, 'Trigger Full Retraining\n(/api/train)', '#3a0ca3'),
        oval(12.5, 3.3, 3.0, 0.8, 'View System Status', '#b5179e'),
    ]

    # Include ovals
    inc_ovals = [
        oval(9.0, 7.0, 2.8, 0.75, 'Classify User Tier', '#1b4332'),
        oval(9.0, 5.5, 2.8, 0.75, 'Stage 1: Candidate\nGeneration', '#1b4332', 8),
        oval(9.0, 4.0, 2.8, 0.75, 'Stage 2: LightGBM\nRanking', '#1b4332', 8),
    ]

    # User actor to use cases
    for uc in uc_user:
        ax.plot([1.3 + 0.3, uc[0] - 1.75], [5.5, uc[1]],
                color='#4cc9f0', lw=1.2, zorder=3, alpha=0.6)

    # Admin actor to use cases
    for uc in uc_admin:
        ax.plot([16.7 - 0.3, uc[0] + 1.6], [5.5, uc[1]],
                color='#f72585', lw=1.2, zorder=3, alpha=0.6)

    def dashed_arrow(x0, y0, x1, y1, label, color='#a8dadc'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.3, linestyle='dashed'))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.1, label, color=color, fontsize=7.5,
                ha='center', va='bottom', fontfamily=FONT,
                style='italic')

    # Include relationships from "Get Movie Recommendations"
    dashed_arrow(7.25, 8.5, 8.6, 7.0, '«include»')
    dashed_arrow(7.25, 8.3, 8.6, 5.65, '«include»')
    dashed_arrow(7.25, 8.1, 8.6, 4.1, '«include»')

    # Extend relationships to "Trigger Full Retraining"
    dashed_arrow(10.85, 8.5, 11.35, 4.85, '«extend»', '#ffd166')
    dashed_arrow(10.85, 7.2, 11.35, 4.75, '«extend»', '#ffd166')
    dashed_arrow(10.85, 5.9, 11.35, 4.65, '«extend»', '#ffd166')

    plt.tight_layout(pad=0.3)
    out = r'D:\Major\movie-recommender\diagrams\usecase.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Saved: {out}  ({os.path.getsize(out):,} bytes)')


# ─────────────────────────────────────────────────────────
# Diagram 5 – System Architecture
# ─────────────────────────────────────────────────────────
def draw_architecture():
    fig, ax = plt.subplots(figsize=(20, 14), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')

    ax.text(10, 13.55, 'System Architecture — Hybrid Two-Stage Movie Recommender',
            color='white', fontsize=20, fontweight='bold',
            ha='center', va='center', fontfamily=FONT)

    layers = [
        # (y_center, height, label, bg_color, label_color, items)
        (12.5, 1.4, 'Layer 1 — Presentation', '#0a3d62', '#74b9ff',
         [('Web Browser', 9.5, '#1e3799', 2.8, 0.65),
          ('Frontend: index.html  (HTML / JS / CSS)', 14.5, '#0c2461', 4.2, 0.65)]),

        (10.6, 1.5, 'Layer 2 — API', '#134f1c', '#55efc4',
         [('FastAPI Server (api.py)  —  Port 8000\n'
           '/api/recommend  |  /api/search  |  /api/popular  |  /api/movie/{id}  |  /api/train',
           10.5, '#1e8449', 8.0, 0.85)]),

        (8.75, 1.5, 'Layer 3 — Business Logic', '#4a2800', '#fdcb6e',
         [('HybridRecommender  (recommender.py)\nUser Tier Classifier  |  Generator Orchestrator',
           10.0, '#784212', 7.5, 0.85)]),

        (6.55, 1.75, 'Layer 4 — Generators', '#2c1250', '#a29bfe',
         [('NMF-ANN Generator\nnmf_model.pkl + nmf.ann', 4.2, '#4a235a', 2.6, 0.95),
          ('SVD-ANN Generator\nsvd_model.pkl + svd.ann', 8.0, '#4a235a', 2.6, 0.95),
          ('Bayesian Popularity\nGenerator', 11.8, '#4a235a', 2.5, 0.95),
          ('Weighted RRF Fusion\n(k=60) → top-200', 15.6, '#6c3483', 2.6, 0.95)]),

        (4.3, 1.75, 'Layer 5 — Ranking', '#4a1a1a', '#ff7675',
         [('Feature Engineering  (44 features)\nUser Stats(5) | Movie Stats(3) | Retrieval(6) | Genome PCA(30)',
           7.5, '#7b241c', 5.5, 0.95),
          ('LightGBM Reranker\nlgbm_reranker.pkl  →  top-10',
           14.5, '#7b241c', 4.0, 0.95)]),

        (2.0, 1.7, 'Layer 6 — Data', '#0b3542', '#00cec9',
         [('MovieLens 25M Dataset\nratings.csv (25M) | movies.csv (62K) | genome-scores.csv | genome-tags.csv',
           8.5, '#117a65', 7.0, 0.95),
          ('Cache Layer\n(pickle / .ann files)', 15.2, '#1a5276', 3.5, 0.95)]),
    ]

    layer_ys = []
    for ly, lh, llabel, lbg, llc, items in layers:
        # Layer background band
        band = FancyBboxPatch((0.3, ly - lh / 2), 19.4, lh,
                              boxstyle='round,pad=0.05',
                              facecolor=lbg, edgecolor='#2d3436',
                              linewidth=1.5, zorder=2, alpha=0.55)
        ax.add_patch(band)
        ax.text(0.65, ly, llabel, color=llc, fontsize=10.5,
                ha='left', va='center', fontweight='bold',
                fontfamily=FONT, rotation=0, zorder=3)

        for (text, cx, fc, bw, bh) in items:
            box = FancyBboxPatch((cx - bw / 2, ly - bh / 2), bw, bh,
                                 boxstyle='round,pad=0.1',
                                 facecolor=fc, edgecolor='white',
                                 linewidth=1.3, zorder=4, alpha=0.9)
            ax.add_patch(box)
            ax.text(cx, ly, text, color='white', fontsize=8.5,
                    ha='center', va='center', fontweight='bold',
                    fontfamily=FONT, zorder=5, multialignment='center')

        layer_ys.append(ly)

    # Downward arrows between layers
    for i in range(len(layer_ys) - 1):
        y_top = layer_ys[i] - layers[i][1] / 2
        y_bot = layer_ys[i + 1] + layers[i + 1][1] / 2
        for cx in [10.0, 12.0]:
            ax.annotate('', xy=(cx, y_bot + 0.02), xytext=(cx, y_top - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#dfe6e9',
                                       lw=1.8, alpha=0.7))

    # Horizontal arrow inside presentation layer
    ax.annotate('', xy=(12.2, 12.5), xytext=(10.75, 12.5),
                arrowprops=dict(arrowstyle='<->', color='#74b9ff', lw=1.8))
    ax.text(11.47, 12.65, 'HTTP', color='#74b9ff', fontsize=8.5,
            ha='center', va='bottom', fontfamily=FONT)

    plt.tight_layout(pad=0.4)
    out = r'D:\Major\movie-recommender\diagrams\architecture.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Saved: {out}  ({os.path.getsize(out):,} bytes)')


if __name__ == '__main__':
    print('Generating diagrams...')
    draw_sdlc()
    draw_sequence()
    draw_activity()
    draw_usecase()
    draw_architecture()
    print('All diagrams generated successfully.')
