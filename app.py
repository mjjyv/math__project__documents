import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự báo Tuyển dụng", page_icon="🎓")

# --- LOAD MODEL & ASSETS ---
@st.cache_resource
def load_assets():
    try:
        clf = joblib.load('models/placement_classifier.pkl')
        reg = joblib.load('models/salary_regressor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        college_freq = joblib.load('models/college_freq_map.pkl')
        model_cols = joblib.load('models/model_columns.pkl')
        return clf, reg, scaler, college_freq, model_cols
    except Exception as e:
        st.error(f"Lỗi không tìm thấy file models: {e}")
        return None, None, None, None, None

clf_model, reg_model, scaler, college_freq, model_columns = load_assets()

if clf_model is not None:
    # --- GIAO DIỆN NHẬP LIỆU ---
    st.title("🎓 Dự báo Tuyển dụng & Lương")
    st.write("Nhập thông tin sinh viên để dự đoán khả năng trúng tuyển.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Tuổi", 20, 30, 24)
        gender = st.selectbox("Giới tính", ['Male', 'Female'])
        stream = st.selectbox("Chuyên ngành", [
            'Electronics and Communication', 'Computer Science',
            'Information Technology', 'Mechanical Engineering',
            'Electrical Engineering', 'Civil Engineering'
        ])

    with col2:
        gpa = st.slider("Điểm GPA", 0.0, 4.0, 3.5, 0.1)
        experience = st.slider("Năm kinh nghiệm", 0, 10, 1)
        # Lấy danh sách trường từ file map đã lưu
        college_list = list(college_freq.index) if college_freq is not None else []
        college = st.selectbox("Trường Đại học", college_list)

    # --- XỬ LÝ DỰ BÁO ---
    if st.button("🚀 Dự báo ngay", type="primary"):
        # 1. Tạo DataFrame thô
        input_data = pd.DataFrame({
            'age': [age],
            'gpa': [gpa],
            'years_of_experience': [experience],
            'gender': [gender],
            'stream': [stream],
            'college_name': [college]
        })

        # 2. Xử lý: Frequency Encoding cho College
        # Nếu trường mới không có trong map, dùng giá trị trung bình
        mean_freq = college_freq.mean()
        val_freq = college_freq.get(college, mean_freq)
        input_data['college_name_freq'] = val_freq
        input_data.drop(columns=['college_name'], inplace=True)

        # 3. Xử lý: One-Hot Encoding
        input_encoded = pd.get_dummies(input_data, columns=['gender', 'stream'], drop_first=True)

        # 4. Xử lý: Đồng bộ cột (Missing columns alignment)
        # Tạo lại đầy đủ các cột như lúc train, điền 0 nếu thiếu
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

        # 5. Xử lý: Scaling (Chuẩn hóa)
        numeric_cols = ['age', 'gpa', 'years_of_experience']
        input_final[numeric_cols] = scaler.transform(input_final[numeric_cols])

        # 6. Dự báo
        try:
            pred_prob = clf_model.predict_proba(input_final)[0][1]
            pred_class = clf_model.predict(input_final)[0]

            st.divider()
            if pred_class == 1:
                salary_pred = reg_model.predict(input_final)[0]
                st.success(f"🎉 **KẾT QUẢ: TRÚNG TUYỂN** (Xác suất: {pred_prob:.1%})")
                st.metric(label="💰 Mức lương dự kiến", value=f"${salary_pred:,.0f}")
            else:
                st.error(f"⚠️ **KẾT QUẢ: CHƯA TRÚNG TUYỂN** (Xác suất đậu: {pred_prob:.1%})")
                st.info("💡 Gợi ý: Cải thiện GPA hoặc tích lũy thêm kinh nghiệm thực tế.")

        except Exception as e:
            st.error(f"Lỗi khi dự báo: {e}")
