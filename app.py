import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np
import cv2

# إعدادات الصفحة
st.set_page_config(page_title="Optical Surface Inspector", layout="wide")

# تحميل النموذج (تأكد من وجود الملف في نفس المسار)
@st.cache_resource
def load_model():
    return YOLO("best.onnx", task="detect")

model = load_model()

st.title("🔍 نظام فحص وتدقيق الأسطح البصرية")
st.write("ارفع صورة العدسة أو الشريحة الزجاجية لتحليل التضاريس واكتشاف العيوب آلياً.")

# واجهة رفع الملفات
uploaded_file = st.sidebar.file_uploader("اختر صورة للتحليل...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # قراءة الصورة
    image = PIL.Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("الصورة الأصلية")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("نتائج التحليل الذكي")
        # إجراء التنبؤ
        results = model.predict(image, conf=0.25)
        res_plotted = results[0].plot()
        
        # عرض الصورة وعليها النتائج
        st.image(res_plotted, caption='العيوب المكتشفة', use_column_width=True)

    # قسم التحليلات (Analytics)
    st.divider()
    st.header("📊 التقرير الفني")
    
    defect_count = len(results[0].boxes)
    avg_conf = np.mean(results[0].boxes.conf.cpu().numpy()) if defect_count > 0 else 0
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        status = "❌ معيب (Defective)" if defect_count > 0 else "✅ سليم (Regular)"
        st.metric("حالة السطح", status)
        
    with metrics_col2:
        st.metric("عدد العيوب المرصودة", defect_count)
        
    with metrics_col3:
        st.metric("متوسط ثقة النموذج", f"{avg_conf:.2%}")

    if defect_count > 0:
        st.warning("⚠️ تم رصد عيوب في تضاريس السطح. يوصى بمراجعة القطعة قبل الاستخدام.")
    else:
        st.success("✨ السطح مطابق للمواصفات الفنية والانتظام البصري.")