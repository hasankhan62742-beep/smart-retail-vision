import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
from groq import Groq
from fpdf import FPDF
from datetime import datetime
import plotly.graph_objects as go
import tempfile
import os

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Smart Retail Vision | Qadeer Automations",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS — MODERN ENTERPRISE STYLE
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 50%, #7c3aed 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 40px rgba(79, 70, 229, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: rgba(255,255,255,0.88);
        font-weight: 400;
        line-height: 1.5;
        max-width: 780px;
    }
    .badge-row { margin-top: 1rem; }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
        border: 1px solid rgba(255,255,255,0.25);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #1e293b 0%, #172033 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 1.1rem 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 800; }

    /* Section headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 0.5rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Alert cards */
    .alert-card {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #fecaca;
        font-weight: 500;
    }
    .info-card {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #bfdbfe;
        font-weight: 500;
    }
    .ok-card {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        color: #bbf7d0;
        font-weight: 500;
    }

    /* AI Summary box */
    .ai-summary-box {
        background: linear-gradient(135deg, rgba(79,70,229,0.12) 0%, rgba(124,58,237,0.12) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        color: #e2e8f0;
        line-height: 1.6;
        font-size: 0.98rem;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding: 1.5rem 0 0.5rem 0;
    }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148,163,184,0.1);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# =========================================================
# CORE CV FUNCTIONS
# =========================================================
def blur_faces(image, boxes, model_names, blur_strength=51):
    img_copy = image.copy()
    for box in boxes:
        cls = model_names[int(box.cls)]
        if cls == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_h = int((y2 - y1) * 0.25)
            face_region = img_copy[y1:y1 + face_h, x1:x2]
            if face_region.size > 0:
                blurred = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
                img_copy[y1:y1 + face_h, x1:x2] = blurred
    return img_copy


def detect_queue(person_boxes, distance_threshold=100, min_cluster_size=3):
    if len(person_boxes) < min_cluster_size:
        return 0, 0
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in person_boxes]
    centers = np.array(centers)
    visited, clusters = set(), []
    for i in range(len(centers)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(len(centers)):
            if j != i and j not in visited and distance.euclidean(centers[i], centers[j]) < distance_threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    max_queue = max([len(c) for c in clusters], default=0)
    return len(clusters), max_queue


def analyze_shelf_stock(image, model, grid_rows=3, grid_cols=4, empty_threshold=1):
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_rows, w // grid_cols
    grid_status, annotated = [], image.copy()
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell_img = image[y1:y2, x1:x2]
            results = model(cell_img, verbose=False)
            num_objects = len(results[0].boxes)
            status = "EMPTY" if num_objects <= empty_threshold else "STOCKED"
            color = (60, 60, 255) if status == "EMPTY" else (80, 220, 120)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, status, (x1 + 8, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            grid_status.append({"status": status})
    stock_pct = sum(1 for g in grid_status if g['status'] == 'STOCKED') / len(grid_status) * 100
    return annotated, stock_pct


def full_retail_analysis(image, model):
    results = model(image, verbose=False)
    person_boxes, product_count = [], 0
    for box in results[0].boxes:
        cls = model.names[int(box.cls)]
        if cls == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))
        else:
            product_count += 1

    blurred_image = blur_faces(image, results[0].boxes, model.names)
    num_clusters, max_queue_len = detect_queue(person_boxes)
    annotated_shelf, stock_pct = analyze_shelf_stock(image, model)

    alerts = []
    if max_queue_len >= 4:
        alerts.append(("alert", f"High queue detected — {max_queue_len} people waiting. Consider opening another counter."))
    if stock_pct < 60:
        alerts.append(("alert", f"Low shelf stock — only {stock_pct:.1f}% of shelves currently stocked."))
    if len(person_boxes) == 0:
        alerts.append(("info", "No customers currently detected in frame."))
    if not alerts:
        alerts.append(("ok", "All systems normal — no active alerts."))

    summary = {
        "total_people": len(person_boxes),
        "queue_clusters": num_clusters,
        "max_queue_length": max_queue_len,
        "shelf_stock_pct": round(stock_pct, 1),
        "products_detected": product_count,
        "alerts": alerts
    }
    return summary, blurred_image, annotated_shelf


def generate_ai_summary(summary):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
        if not api_key:
            return "AI summary unavailable — GROQ_API_KEY not configured in app secrets."
        client = Groq(api_key=api_key)
        alert_text = "; ".join([a[1] for a in summary['alerts']])
        prompt = f"""You are a retail operations analyst. Based on this real-time store data, write a concise 3-4 sentence executive summary for a store manager. Be direct, specific, and actionable — no fluff, no markdown.

Data:
- Customers in frame: {summary['total_people']}
- Queue clusters: {summary['queue_clusters']}
- Longest queue: {summary['max_queue_length']} people
- Shelf stock level: {summary['shelf_stock_pct']}%
- Products detected: {summary['products_detected']}
- Alerts: {alert_text}

Write the summary now:"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=220
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI summary unavailable — {str(e)}"


def clean_text_for_pdf(text):
    text = text.encode('latin-1', 'ignore').decode('latin-1')
    return text


def generate_pdf_report(summary, ai_summary, shelf_image):
    temp_img_path = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    cv2.imwrite(temp_img_path, shelf_image)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    content_width = pdf.epw

    def reset_x():
        pdf.set_x(pdf.l_margin)

    def section_title(text, color=(30, 41, 59)):
        reset_x()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*color)
        pdf.cell(content_width, 10, text, new_x="LMARGIN", new_y="NEXT")
        reset_x()

    pdf.set_fill_color(30, 58, 138)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(0, 8)
    pdf.cell(210, 12, "AI Smart Retail Vision - Executive Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(32)
    reset_x()

    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(content_width, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    reset_x()

    section_title("Executive Summary")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    reset_x()
    pdf.multi_cell(content_width, 6, clean_text_for_pdf(ai_summary) or "No summary available.")
    pdf.ln(4)
    reset_x()

    section_title("Key Metrics")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    for m in [
        f"Total Customers Detected: {summary['total_people']}",
        f"Queue Clusters: {summary['queue_clusters']}",
        f"Longest Queue: {summary['max_queue_length']} people",
        f"Shelf Stock Level: {summary['shelf_stock_pct']}%",
        f"Products Detected: {summary['products_detected']}",
    ]:
        reset_x()
        pdf.cell(content_width, 7, f"- {m}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    reset_x()

    section_title("Active Alerts", color=(185, 28, 28))
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(50, 50, 50)
    for _, msg in summary['alerts']:
        clean_msg = clean_text_for_pdf(msg) or "N/A"
        reset_x()
        pdf.multi_cell(content_width, 6, f"- {clean_msg}")
    pdf.ln(4)
    reset_x()

    section_title("Shelf Stock Visualization")
    reset_x()
    pdf.image(temp_img_path, w=content_width)

    pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    pdf.output(pdf_path)
    os.remove(temp_img_path)
    return pdf_path


def render_alerts(alerts):
    css_map = {"alert": "alert-card", "info": "info-card", "ok": "ok-card"}
    icon_map = {"alert": "⚠️", "info": "ℹ️", "ok": "✅"}
    html = ""
    for kind, msg in alerts:
        html += f'<div class="{css_map[kind]}">{icon_map[kind]} {msg}</div>'
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏬 AI Smart Retail Vision System</div>
    <div class="hero-subtitle">
        Enterprise-grade in-store intelligence — footfall analytics, queue monitoring, shelf stock tracking,
        and AI-generated executive summaries, powered by YOLOv8 and privacy-first design.
    </div>
    <div class="badge-row">
        <span class="badge">🔒 Privacy-First (Face Blurring)</span>
        <span class="badge">⚡ Real-Time Detection</span>
        <span class="badge">🤖 AI Executive Summaries</span>
        <span class="badge">📄 Exportable Reports</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption("Adjust detection sensitivity for your store layout.")
    queue_threshold = st.slider("Queue alert threshold (people)", 2, 10, 4)
    stock_threshold = st.slider("Low stock alert threshold (%)", 20, 90, 60)
    st.divider()
    st.markdown("### 📊 About This System")
    st.caption(
        "This system uses YOLOv8 object detection combined with rule-based analytics "
        "to deliver real-time retail insights without storing or identifying individual "
        "customer faces — designed with US privacy regulations (e.g. BIPA) in mind."
    )
    st.divider()
    st.caption("Built by **Qadeer Automations**")

# =========================================================
# MAIN TABS
# =========================================================
tab1, tab2 = st.tabs(["📷  Image Analysis", "🎥  Video Analysis"])

# ---------------- IMAGE MODE ----------------
with tab1:
    uploaded_file = st.file_uploader("Upload a store or shelf image", type=["jpg", "jpeg", "png"], key="img")

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Running detection pipeline..."):
            summary, blurred_img, shelf_img = full_retail_analysis(image, model)
            ai_summary = generate_ai_summary(summary)

        st.markdown('<div class="section-title">📊 Live Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Customers", summary['total_people'])
        c2.metric("⏱️ Longest Queue", summary['max_queue_length'])
        c3.metric("📦 Shelf Stock", f"{summary['shelf_stock_pct']}%")
        c4.metric("🧾 Products Detected", summary['products_detected'])

        st.markdown('<div class="section-title">🚨 Alerts</div>', unsafe_allow_html=True)
        render_alerts(summary['alerts'])

        st.markdown('<div class="section-title">🤖 AI Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-summary-box">{ai_summary}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">🔒 Privacy-Compliant View</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.markdown('<div class="section-title">📦 Shelf Stock Analysis</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(shelf_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        pdf_path = generate_pdf_report(summary, ai_summary, shelf_img)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Executive PDF Report", f, file_name="Retail_Vision_Report.pdf", use_container_width=False)
    else:
        st.info("👆 Upload a store or shelf image above to run the analysis.")

# ---------------- VIDEO MODE ----------------
with tab2:
    uploaded_video = st.file_uploader("Upload store video (.mp4)", type=["mp4"], key="vid")

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        progress = st.progress(0, text="Processing video frames...")
        cap = cv2.VideoCapture(tfile.name)
        frame_count, processed = 0, 0
        footfall_trend, queue_trend = [], []
        max_frames = 60

        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 15 == 0:
                results = model(frame, verbose=False)
                person_boxes = [tuple(map(int, box.xyxy[0])) for box in results[0].boxes if model.names[int(box.cls)] == 'person']
                _, max_q = detect_queue(person_boxes)
                footfall_trend.append(len(person_boxes))
                queue_trend.append(max_q)
                processed += 1
                progress.progress(min(processed / max_frames, 1.0), text=f"Processing frame {processed}/{max_frames}...")
            frame_count += 1
        cap.release()
        os.unlink(tfile.name)
        progress.empty()

        st.markdown('<div class="section-title">📊 Session Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Footfall", round(np.mean(footfall_trend), 1) if footfall_trend else 0)
        c2.metric("Peak Footfall", max(footfall_trend) if footfall_trend else 0)
        c3.metric("Peak Queue Length", max(queue_trend) if queue_trend else 0)

        st.markdown('<div class="section-title">📈 Trend Over Time</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=footfall_trend, mode='lines+markers', name='Footfall',
                                  line=dict(color="#4f46e5", width=3)))
        fig.add_trace(go.Scatter(y=queue_trend, mode='lines+markers', name='Queue Length',
                                  line=dict(color="#ef4444", width=3)))
        fig.update_layout(
            template="plotly_dark",
            title="Footfall & Queue Trend Across Sampled Frames",
            xaxis_title="Sampled Frame",
            yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Upload a short store video above to analyze footfall and queue trends over time.")

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer-note">Built with YOLOv8 + Groq LLaMA + Streamlit &nbsp;|&nbsp; '
    'Privacy-first design, no facial recognition &nbsp;|&nbsp; © Qadeer Automations</div>',
    unsafe_allow_html=True
)
