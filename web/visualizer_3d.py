import os
import sys
import json
import torch
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Model ve Token Veritabanı
CATALOG = [
    {
        "id": "v2_run1",
        "name": "Dorina V2 (Run 1)",
        "path": "models/dorina_44m_moe/run1/model.pt",
        "tokens": 50_000_000,
        "tokens_label": "50M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run2",
        "name": "Dorina V2 (Run 2)",
        "path": "models/dorina_44m_moe/run2/model.pt",
        "tokens": 100_000_000,
        "tokens_label": "100M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run3",
        "name": "Dorina V2 (Run 3)",
        "path": "models/dorina_44m_moe/run3/model.pt",
        "tokens": 150_000_000,
        "tokens_label": "150M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run4",
        "name": "Dorina V2 (Run 4)",
        "path": "models/dorina_44m_moe/run4/model.pt",
        "tokens": 200_000_000,
        "tokens_label": "200M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run5",
        "name": "Dorina V2 (Run 5)",
        "path": "models/dorina_44m_moe/run5/best.pt",
        "tokens": 260_000_000,
        "tokens_label": "260M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run6",
        "name": "Dorina V2 (Run 6)",
        "path": "models/dorina_44m_moe/run6/best.pt",
        "tokens": 356_300_000,
        "tokens_label": "356.3M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run7",
        "name": "Dorina V2 (Run 7)",
        "path": "models/dorina_44m_moe/run7/best.pt",
        "tokens": 454_600_000,
        "tokens_label": "454.6M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run8",
        "name": "Dorina V2 (Run 8)",
        "path": "models/dorina_44m_moe/run8/best.pt",
        "tokens": 552_900_000,
        "tokens_label": "552.9M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run9",
        "name": "Dorina V2 (Run 9)",
        "path": "models/dorina_44m_moe/run9/best.pt",
        "tokens": 651_200_000,
        "tokens_label": "651.2M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run10",
        "name": "Dorina V2 (Run 10)",
        "path": "models/dorina_44m_moe/run10/best.pt",
        "tokens": 749_500_000,
        "tokens_label": "749.5M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run11",
        "name": "Dorina V2 (Run 11)",
        "path": "models/dorina_44m_moe/run11/best.pt",
        "tokens": 847_800_000,
        "tokens_label": "847.8M Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v2_run12",
        "name": "Dorina V2 (Run 12 - 1B Milat)",
        "path": "models/dorina_44m_moe/run12/best.pt",
        "tokens": 1_000_000_000,
        "tokens_label": "1.000M (1 Milyar) Token",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v3_shard1",
        "name": "Dorina V3 (Shard 1 Tamamlandı)",
        "path": "models/dorina_v3_180m/shard1/dorina_v3_shard1_final.pt",
        "tokens": 200_000_000,
        "tokens_label": "200M Token (Shard 1)",
        "is_training": False,
        "is_target": False
    },
    {
        "id": "v3_shard2_live",
        "name": "Dorina V3 Shard 2 (Kaggle Eğitimde)",
        "path": "models/dorina_v3_180m/shard1/dorina_v3_shard1_final.pt",
        "tokens": 400_000_000,
        "tokens_label": "400M Token (Eğitimde)",
        "is_training": True,
        "is_target": False
    },
    {
        "id": "v3_target_final",
        "name": "Dorina V3 (Nihai Hedef Model - 3.6B)",
        "path": "models/dorina_v3_180m/shard1/dorina_v3_shard1_final.pt",
        "tokens": 3_600_000_000,
        "tokens_label": "3.6 Milyar Token (Nihai Hedef)",
        "is_training": False,
        "is_target": True
    }
]

def analyze_model_deep(path):
    if not os.path.exists(path):
        return None
    
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
        
    total_params = 0
    cat_counts = {
        "embed": 0,
        "attn": 0,
        "norm": 0,
        "router": 0,
        "experts": 0,
        "shared": 0
    }
    
    layer_indices = set()
    n_experts = 1
    d_model = 512
    vocab_size = 32768
    
    for k, v in sd.items():
        p_count = v.numel()
        total_params += p_count
        shape_list = list(v.shape)
        
        cat = "shared"
        if "tok_emb" in k or "embed" in k:
            cat = "embed"
            if len(shape_list) == 2:
                vocab_size, d_model = shape_list[0], shape_list[1]
        elif "attn" in k or "qkv" in k or "q_proj" in k:
            cat = "attn"
        elif "norm" in k or "ln" in k:
            cat = "norm"
        elif "router" in k or "gate" in k:
            cat = "router"
        elif "moe.w_in" in k or "moe.w_out" in k or "experts" in k:
            cat = "experts"
            if len(shape_list) == 3:
                n_experts = shape_list[0]
        elif "shared" in k:
            cat = "shared"
            
        cat_counts[cat] = cat_counts.get(cat, 0) + p_count
        
        if "blocks." in k:
            try:
                layer_indices.add(int(k.split("blocks.")[1].split(".")[0]))
            except Exception:
                pass
        elif "layers." in k:
            try:
                layer_indices.add(int(k.split("layers.")[1].split(".")[0]))
            except Exception:
                pass

    n_layers = len(layer_indices) if layer_indices else 12
    
    if n_experts == 1:
        expert_ids = set()
        for k in sd.keys():
            if "experts." in k:
                try:
                    expert_ids.add(int(k.split("experts.")[1].split(".")[0]))
                except Exception:
                    pass
        if expert_ids:
            n_experts = len(expert_ids)
            
    if n_experts > 1:
        if "v3" in path or n_experts == 32:
            active_params = int(cat_counts["embed"] + cat_counts["attn"] + cat_counts["norm"] + cat_counts["router"] + cat_counts["shared"] + (cat_counts["experts"] * 2 / 32))
        else:
            active_params = int(cat_counts["embed"] + cat_counts["attn"] + cat_counts["norm"] + cat_counts["router"] + (cat_counts["experts"] * 2 / max(1, n_experts)))
    else:
        active_params = total_params

    total_voxels = round(total_params / 1000)
    cube_side = round(total_voxels ** (1/3))

    voxels_by_cat = {
        "embed": round(cat_counts["embed"] / 1000),
        "attn": round(cat_counts["attn"] / 1000),
        "norm": round(cat_counts["norm"] / 1000),
        "router": round(cat_counts["router"] / 1000),
        "experts": round(cat_counts["experts"] / 1000),
        "shared": round(cat_counts["shared"] / 1000)
    }

    return {
        "file": os.path.basename(path),
        "total_params": total_params,
        "active_params": active_params,
        "total_voxels": total_voxels,
        "cube_side": cube_side,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "voxels_by_cat": voxels_by_cat,
        "params_by_cat": cat_counts
    }

def get_full_lineup_data():
    results = []
    for item in CATALOG:
        info = analyze_model_deep(item["path"])
        if info:
            res = dict(item)
            res.update(info)
            results.append(res)
    # Sıralama: Küçükten büyüğe (parametre, ardından token)
    results.sort(key=lambda x: (x["total_params"], x["tokens"]))
    return results

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dorina Neural Universe & Synaptic Network Visualizer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
        body { background: #05070c; color: #e2e8f0; overflow: hidden; display: flex; height: 100vh; }
        
        #sidebar {
            width: 440px; background: rgba(9, 13, 22, 0.96); border-right: 1px solid rgba(255, 255, 255, 0.08);
            padding: 20px; overflow-y: auto; z-index: 10; backdrop-filter: blur(14px); display: flex; flex-direction: column; gap: 14px;
        }
        #canvas-container { flex: 1; position: relative; }
        
        h1 { font-size: 19px; font-weight: 800; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
        .subtitle { font-size: 11px; color: #94a3b8; }
        
        .card { background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 10px; padding: 14px; }
        .card-title { font-size: 11px; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .stat-box { background: rgba(0, 0, 0, 0.35); padding: 9px; border-radius: 6px; border-left: 3px solid #38bdf8; }
        .stat-lbl { font-size: 10px; color: #94a3b8; text-transform: uppercase; }
        .stat-num { font-size: 14px; font-weight: 700; color: #ffffff; margin-top: 3px; }
        
        .legend-row { display: flex; align-items: center; justify-content: space-between; font-size: 11px; padding: 5px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }
        .legend-color { width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; display: inline-block; }
        
        .wasd-badge {
            background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8;
            padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 11px; margin: 0 2px;
        }
        
        #hud {
            position: absolute; bottom: 20px; right: 20px; background: rgba(9, 13, 22, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 10px; padding: 14px 20px;
            font-size: 12px; color: #94a3b8; backdrop-filter: blur(10px); pointer-events: none;
        }
        .hud-highlight { color: #38bdf8; font-weight: 700; }
        
        .btn-action {
            width: 100%; padding: 10px; border-radius: 6px; font-size: 12px; font-weight: 700; cursor: pointer; border: none;
            transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 6px; background: #0284c7; color: white;
        }
        .btn-action:hover { background: #0369a1; }
        
        .btn-toggle-synapse {
            width: 100%; padding: 9px; border-radius: 6px; font-size: 11px; font-weight: 700; cursor: pointer;
            border: 1px solid #38bdf8; background: rgba(56, 189, 248, 0.1); color: #38bdf8; transition: 0.2s;
        }
        .btn-toggle-synapse:hover { background: rgba(56, 189, 248, 0.25); }
    </style>
</head>
<body>

    <div id="sidebar">
        <div>
            <h1>🧠 Dorina Neural Universe</h1>
            <div class="subtitle">3D Voxel Podyumu & Canlı Sinaps Ağı</div>
        </div>

        <!-- WASD Gezinme Rehberi -->
        <div class="card" style="border-color: #38bdf8;">
            <div class="card-title">🚀 WASD Serbest Uçuş Aktif</div>
            <p style="font-size: 11px; color: #cbd5e1; line-height: 1.6;">
                <span class="wasd-badge">W</span> İleri | <span class="wasd-badge">S</span> Geri | 
                <span class="wasd-badge">A</span> Sola | <span class="wasd-badge">D</span> Sağa<br>
                <span class="wasd-badge">Q</span> / <span class="wasd-badge">E</span> Aşağı / Yukarı Uçuş | <span class="wasd-badge">Shift</span> Hızlı Uçuş<br>
                <span style="color: #94a3b8;">🖱️ Mouse sol tık ile serbestçe bak, modellerin içine ve sinaps hatlarına kadar yaklaş!</span>
            </p>
        </div>

        <!-- Sinaps Ağı & Hedef Model -->
        <div class="card" style="border-color: #a855f7;">
            <div class="card-title">⚡ Nöral Sinaps & Hedef Model</div>
            <p style="font-size: 11px; color: #94a3b8; line-height: 1.5; margin-bottom: 8px;">
                ● <b>Sinaptik Bağlar:</b> Modeller arasında akan nöral veri ve ağırlık transfer hatları (parlayan dinamik sinapslar).<br>
                ● <b>Hedef Dorina V3 (3.6B):</b> Podyumun en sağındaki devasa altın taç; 3.6 Milyar tokenla tam doygunluğa ulaşacak olan nihai şampiyon modelimiz.
            </p>
            <button class="btn-toggle-synapse" id="btn-synapse" onclick="toggleSynapses()">⚡ Sinaps Parıltısını Aç / Kapat</button>
        </div>

        <!-- Renk Kodları (Nöral Ağ Dağılımı) -->
        <div class="card">
            <div class="card-title">🎨 Nöral Ağ Dağılımı</div>
            <div class="legend-row">
                <div><span class="legend-color" style="background: #a855f7;"></span> MoE Uzmanlar (FFN Experts)</div>
                <span style="color: #94a3b8; font-size: 10px;">En Büyük Blok</span>
            </div>
            <div class="legend-row">
                <div><span class="legend-color" style="background: #0284c7;"></span> Self-Attention (Q, K, V, O)</div>
                <span style="color: #94a3b8; font-size: 10px;">Mavi Küpler</span>
            </div>
            <div class="legend-row">
                <div><span class="legend-color" style="background: #f59e0b;"></span> Embedding / Çıkış Vocab</div>
                <span style="color: #94a3b8; font-size: 10px;">Altın Sarısı</span>
            </div>
            <div class="legend-row">
                <div><span class="legend-color" style="background: #ec4899;"></span> MoE Router (Yönlendirici)</div>
                <span style="color: #94a3b8; font-size: 10px;">Pembe Küpler</span>
            </div>
            <div class="legend-row">
                <div><span class="legend-color" style="background: #22c55e;"></span> Shared Always-On Expert</div>
                <span style="color: #94a3b8; font-size: 10px;">Yeşil Çekirdek</span>
            </div>
            <div class="legend-row">
                <div><span class="legend-color" style="background: #64748b;"></span> RMSNorm / LayerNorm</div>
                <span style="color: #94a3b8; font-size: 10px;">Gümüş Küpler</span>
            </div>
            <div class="legend-row" style="border-top: 1px dashed rgba(255,255,255,0.1); margin-top: 6px; padding-top: 8px;">
                <div><span class="legend-color" style="background: #eab308; border: 1px solid #ffffff;"></span> <b>Arka Veri Bloğu (Token)</b></div>
                <span style="color: #facc15; font-size: 10px; font-weight: 700;">Görülen Token Sayısı</span>
            </div>
        </div>

        <!-- Odaktaki Model Bilgisi -->
        <div class="card">
            <div class="card-title">🔍 Odaktaki Model</div>
            <div id="active-model-card">
                <div style="font-size: 13px; font-weight: 700; color: #38bdf8;" id="f-name">Yükleniyor...</div>
                <div class="stat-grid" style="margin-top: 8px;">
                    <div class="stat-box">
                        <div class="stat-lbl">Parametre (Küp)</div>
                        <div class="stat-num" id="f-params">-</div>
                    </div>
                    <div class="stat-box" style="border-left-color: #eab308;">
                        <div class="stat-lbl">Eğitim Verisi</div>
                        <div class="stat-num" id="f-tokens" style="color: #facc15;">-</div>
                    </div>
                </div>
            </div>
        </div>

        <button class="btn-action" onclick="resetFlyCamera()">🎯 Başlangıç Noktasına Dön</button>
    </div>

    <div id="canvas-container">
        <div id="hud">
            <div>🎮 <span class="hud-highlight">WASD:</span> Serbest Uçuş | <span class="hud-highlight">Q/E:</span> Yüksel/Alçal | <span class="hud-highlight">Mouse Sol:</span> Açı Çevir</div>
            <div style="margin-top: 4px; font-size: 11px; color: #64748b;">Modeller küçükten büyüğe sıralı. Modeller arası parlayan sinaps hatları aktif!</div>
        </div>
    </div>

    <script>
        // ----------------- Three.js Sahne Kurulumu -----------------
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x05070c, 0.0010);

        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 8000);
        camera.position.set(0, 70, 260);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.maxDistance = 4000;

        // Işıklar
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.4);
        dirLight1.position.set(120, 300, 120);
        scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0xa855f7, 0.9);
        dirLight2.position.set(-120, -100, -120);
        scene.add(dirLight2);

        // Zemin Grid Podyumu
        const grid = new THREE.GridHelper(1600, 160, 0x1e293b, 0x0f172a);
        grid.position.y = -30;
        scene.add(grid);

        const worldGroup = new THREE.Group();
        scene.add(worldGroup);

        const synapseGroup = new THREE.Group();
        scene.add(synapseGroup);

        let showSynapses = true;
        function toggleSynapses() {
            showSynapses = !showSynapses;
            synapseGroup.visible = showSynapses;
            document.getElementById('btn-synapse').innerText = showSynapses ? "⚡ Sinaps Parıltısını Kapat" : "⚡ Sinaps Parıltısını Aç";
        }

        // ----------------- WASD Klavye Gezinme Sistemi -----------------
        const keys = { w: false, a: false, s: false, d: false, q: false, e: false, shift: false };
        window.addEventListener('keydown', (e) => {
            const k = e.key.toLowerCase();
            if (keys.hasOwnProperty(k)) keys[k] = true;
            if (e.key === 'Shift') keys.shift = true;
        });
        window.addEventListener('keyup', (e) => {
            const k = e.key.toLowerCase();
            if (keys.hasOwnProperty(k)) keys[k] = false;
            if (e.key === 'Shift') keys.shift = false;
        });

        function handleWASD() {
            const speed = keys.shift ? 5.0 : 2.2;
            const forward = new THREE.Vector3();
            camera.getWorldDirection(forward);
            forward.y = 0;
            forward.normalize();

            const right = new THREE.Vector3();
            right.crossVectors(camera.up, forward).normalize();

            if (keys.w) {
                camera.position.addScaledVector(forward, speed);
                controls.target.addScaledVector(forward, speed);
            }
            if (keys.s) {
                camera.position.addScaledVector(forward, -speed);
                controls.target.addScaledVector(forward, -speed);
            }
            if (keys.d) {
                camera.position.addScaledVector(right, -speed);
                controls.target.addScaledVector(right, -speed);
            }
            if (keys.a) {
                camera.position.addScaledVector(right, speed);
                controls.target.addScaledVector(right, speed);
            }
            if (keys.e) {
                camera.position.y += speed;
                controls.target.y += speed;
            }
            if (keys.q) {
                camera.position.y -= speed;
                controls.target.y -= speed;
            }
        }

        function resetFlyCamera() {
            camera.position.set(0, 70, 260);
            controls.target.set(0, 0, 0);
            controls.update();
        }

        // ----------------- Model Voxel Podu & Sinaps Bağları -----------------
        const CAT_COLORS = {
            embed: 0xf59e0b,
            attn: 0x0284c7,
            norm: 0x64748b,
            router: 0xec4899,
            experts: 0xa855f7,
            shared: 0x22c55e
        };

        const synapseNodes = []; // Sinaps bağlantı koordinatları

        function buildModelVoxelPod(model, posX) {
            const group = new THREE.Group();
            group.position.set(posX, 0, 0);

            const side = model.cube_side;
            const totalVoxels = model.total_voxels;
            const unitSize = 0.88;
            const step = 1.0;
            const halfSide = (side * step) / 2;

            const isTarget = model.is_target;
            const isLive = model.is_training;

            const catVoxels = model.voxels_by_cat;
            const geom = new THREE.BoxGeometry(unitSize, unitSize, unitSize);
            const catOrder = ['shared', 'router', 'norm', 'attn', 'embed', 'experts'];
            
            const meshGroups = {};
            catOrder.forEach(cat => {
                const count = Math.max(1, catVoxels[cat] || 0);
                const mat = new THREE.MeshStandardMaterial({
                    color: isTarget ? 0xf59e0b : (CAT_COLORS[cat] || 0x38bdf8),
                    roughness: 0.25,
                    metalness: isTarget ? 0.85 : 0.6,
                    emissive: isTarget ? 0xd97706 : ((cat === 'shared' || cat === 'router') ? CAT_COLORS[cat] : 0x000000),
                    emissiveIntensity: isTarget ? 0.35 : 0.25
                });
                const instMesh = new THREE.InstancedMesh(geom, mat, count);
                instMesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
                meshGroups[cat] = { mesh: instMesh, count: 0, max: count };
                group.add(instMesh);
            });

            const matrix = new THREE.Matrix4();
            let currentCatIdx = 0;
            let currentCatName = catOrder[currentCatIdx];
            let placedInCurrentCat = 0;

            outerLoop:
            for (let z = 0; z < side; z++) {
                for (let y = 0; y < side; y++) {
                    for (let x = 0; x < side; x++) {
                        const px = x * step - halfSide;
                        const py = y * step - halfSide + (side * step) / 2 - 30;
                        const pz = z * step - halfSide;

                        matrix.setPosition(px, py, pz);

                        while (currentCatIdx < catOrder.length && placedInCurrentCat >= (catVoxels[currentCatName] || 0)) {
                            currentCatIdx++;
                            if (currentCatIdx >= catOrder.length) break outerLoop;
                            currentCatName = catOrder[currentCatIdx];
                            placedInCurrentCat = 0;
                        }

                        const grp = meshGroups[currentCatName];
                        if (grp && grp.count < grp.max) {
                            grp.mesh.setMatrixAt(grp.count, matrix);
                            grp.count++;
                            placedInCurrentCat++;
                        }
                    }
                }
            }

            Object.values(meshGroups).forEach(grp => {
                grp.mesh.count = grp.count;
                grp.mesh.instanceMatrix.needsUpdate = true;
            });

            // Sınır Çerçevesi
            const wireGeom = new THREE.BoxGeometry(side * step, side * step, side * step);
            const wire = new THREE.LineSegments(
                new THREE.EdgesGeometry(wireGeom),
                new THREE.LineBasicMaterial({
                    color: isTarget ? 0xfde047 : (isLive ? 0xef4444 : 0x38bdf8),
                    linewidth: isTarget ? 3 : 2,
                    transparent: true,
                    opacity: isTarget ? 0.8 : 0.4
                })
            );
            wire.position.y = (side * step) / 2 - 30;
            group.add(wire);

            // ================= VERİ TOKEN BLOĞU =================
            const tokenMillions = model.tokens / 1e6;
            const tokenVolume = Math.max(10, tokenMillions * 12);
            const tokenSide = Math.cbrt(tokenVolume * 50);
            const tokenHeight = tokenSide * 1.5;

            const tokenGeom = new THREE.BoxGeometry(tokenSide, tokenHeight, tokenSide);
            const tokenMat = new THREE.MeshStandardMaterial({
                color: isTarget ? 0xec4899 : (isLive ? 0xef4444 : 0xeab308),
                emissive: isTarget ? 0xbe185d : (isLive ? 0xb91c1c : 0xca8a04),
                emissiveIntensity: (isLive || isTarget) ? 0.75 : 0.35,
                roughness: 0.2,
                metalness: 0.8,
                transparent: true,
                opacity: 0.85
            });

            const tokenMesh = new THREE.Mesh(tokenGeom, tokenMat);
            tokenMesh.position.set(0, tokenHeight / 2 - 30, - (halfSide + tokenSide / 2 + 25));
            group.add(tokenMesh);

            const tokenWire = new THREE.LineSegments(
                new THREE.EdgesGeometry(tokenGeom),
                new THREE.LineBasicMaterial({ color: isTarget ? 0xf472b6 : (isLive ? 0xf87171 : 0xfde047), linewidth: 2 })
            );
            tokenMesh.add(tokenWire);

            // Tabela
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 160;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(0, 0, 512, 160);
            ctx.strokeStyle = isTarget ? '#f59e0b' : (isLive ? '#ef4444' : '#38bdf8');
            ctx.lineWidth = 6;
            ctx.strokeRect(0, 0, 512, 160);

            ctx.fillStyle = isTarget ? '#fde047' : (isLive ? '#f87171' : '#38bdf8');
            ctx.font = 'bold 30px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(model.name, 256, 50);

            ctx.fillStyle = '#f1f5f9';
            ctx.font = '24px Arial';
            ctx.fillText(`${(model.total_params/1e6).toFixed(1)}M Param (${model.total_voxels.toLocaleString()} Küp)`, 256, 95);

            ctx.fillStyle = isTarget ? '#f472b6' : '#eab308';
            ctx.font = 'bold 24px Arial';
            ctx.fillText(`📚 Veri: ${model.tokens_label}`, 256, 138);

            const tex = new THREE.CanvasTexture(canvas);
            const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex }));
            sprite.scale.set(42, 13, 1);
            sprite.position.set(0, side * step - 18, 0);
            group.add(sprite);

            // Sinaps Bağlantı Düğümü (Merkez Koordinat)
            synapseNodes.push({
                center: new THREE.Vector3(posX, (side * step) / 2 - 30, 0),
                tokenCenter: new THREE.Vector3(posX, tokenHeight / 2 - 30, - (halfSide + tokenSide / 2 + 25)),
                side: side,
                name: model.name
            });

            return group;
        }

        // ----------------- Canlı Sinaps Ağı Oluşturma -----------------
        const synapseParticles = [];

        function buildSynapseNetwork() {
            while (synapseGroup.children.length > 0) {
                synapseGroup.remove(synapseGroup.children[0]);
            }

            for (let i = 0; i < synapseNodes.length - 1; i++) {
                const n1 = synapseNodes[i];
                const n2 = synapseNodes[i + 1];

                // 1. Modeller Arası Akış Eğrisi (Bezier Curve)
                const midX = (n1.center.x + n2.center.x) / 2;
                const midY = Math.max(n1.center.y, n2.center.y) + 25;
                const curve = new THREE.QuadraticBezierCurve3(
                    n1.center,
                    new THREE.Vector3(midX, midY, 15),
                    n2.center
                );

                const points = curve.getPoints(40);
                const lineGeom = new THREE.BufferGeometry().setFromPoints(points);
                const lineMat = new THREE.LineBasicMaterial({
                    color: 0x38bdf8,
                    transparent: true,
                    opacity: 0.35,
                    linewidth: 1
                });
                const curveLine = new THREE.Line(lineGeom, lineMat);
                synapseGroup.add(curveLine);

                // 2. Model ile Arkasındaki Veri Arası Sinaps Hattı
                const dataCurve = new THREE.LineCurve3(n1.center, n1.tokenCenter);
                const dataGeom = new THREE.BufferGeometry().setFromPoints(dataCurve.getPoints(10));
                const dataMat = new THREE.LineDashedMaterial({
                    color: 0xeab308,
                    dashSize: 3,
                    gapSize: 2,
                    transparent: true,
                    opacity: 0.4
                });
                const dataLine = new THREE.Line(dataGeom, dataMat);
                dataLine.computeLineDistances();
                synapseGroup.add(dataLine);

                // 3. Eğri Üzerinde Akan Sinaps Işık Parçacıkları
                for (let p = 0; p < 3; p++) {
                    const sparkGeom = new THREE.SphereGeometry(1.2, 8, 8);
                    const sparkMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
                    const sparkMesh = new THREE.Mesh(sparkGeom, sparkMat);
                    synapseGroup.add(sparkMesh);

                    synapseParticles.push({
                        mesh: sparkMesh,
                        curve: curve,
                        progress: Math.random(),
                        speed: 0.003 + Math.random() * 0.003
                    });
                }
            }

            // Son model ile kendi veri bloğu hattı
            if (synapseNodes.length > 0) {
                const last = synapseNodes[synapseNodes.length - 1];
                const dataCurve = new THREE.LineCurve3(last.center, last.tokenCenter);
                const dataGeom = new THREE.BufferGeometry().setFromPoints(dataCurve.getPoints(10));
                const dataMat = new THREE.LineDashedMaterial({ color: 0xec4899, dashSize: 3, gapSize: 2, transparent: true, opacity: 0.6 });
                const dataLine = new THREE.Line(dataGeom, dataMat);
                dataLine.computeLineDistances();
                synapseGroup.add(dataLine);
            }
        }

        // Podyumu İnşa Et
        let lineupModels = [];

        async function initLineup() {
            try {
                const res = await fetch('/api/lineup');
                lineupModels = await res.json();
                
                let currentX = - ((lineupModels.length - 1) * 85) / 2;

                lineupModels.forEach((m, idx) => {
                    const pod = buildModelVoxelPod(m, currentX);
                    worldGroup.add(pod);
                    m.worldX = currentX;
                    currentX += 85;
                });

                buildSynapseNetwork();

                if (lineupModels.length > 0) {
                    updateFocusedCard(lineupModels[0]);
                }

            } catch (err) {
                console.error("Lineup yüklenemedi:", err);
            }
        }

        function updateFocusedCard(m) {
            document.getElementById('f-name').innerText = m.name;
            document.getElementById('f-params').innerText = (m.total_params/1e6).toFixed(1) + "M (" + m.total_voxels.toLocaleString() + " Küp)";
            document.getElementById('f-tokens').innerText = m.tokens_label;
        }

        function trackNearestModel() {
            if (!lineupModels.length) return;
            let nearest = lineupModels[0];
            let minDist = Infinity;
            lineupModels.forEach(m => {
                const dist = Math.abs(camera.position.x - m.worldX);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = m;
                }
            });
            if (nearest) updateFocusedCard(nearest);
        }

        // ----------------- Animasyon Döngüsü -----------------
        function animate() {
            requestAnimationFrame(animate);
            handleWASD();
            controls.update();
            trackNearestModel();

            // Sinaps Işık Parçacıklarının Akışı
            if (showSynapses) {
                synapseParticles.forEach(sp => {
                    sp.progress += sp.speed;
                    if (sp.progress > 1.0) sp.progress = 0.0;
                    const pos = sp.curve.getPoint(sp.progress);
                    sp.mesh.position.copy(pos);
                });
            }

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });

        initLineup();
    </script>
</body>
</html>
"""

class VisualizerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/" or parsed.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            
        elif parsed.path == "/api/lineup":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            data = get_full_lineup_data()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            
        else:
            super().do_GET()

def start_server(port=7860):
    server = HTTPServer(('127.0.0.1', port), VisualizerHandler)
    print(f"Dorina 3D Synapse Universe baslatildi: http://localhost:{port}")
    server.serve_forever()

if __name__ == "__main__":
    start_server(7860)
