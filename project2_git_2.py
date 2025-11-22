import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ==========================================================
# 1. CACHED LOADERS
# ==========================================================

@st.cache_resource
def load_models():
    with open('xe_cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)

    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    return cosine_sim, vectorizer, tfidf_matrix, kmeans, scaler, pca


@st.cache_data
def load_and_clean_data():
    # Load & rename columns
    data = pd.read_excel('data_motobikes.xlsx').rename(columns={
        'Tiêu đề': 'title',
        'Địa chỉ': 'address',
        'Mô tả chi tiết': 'description',
        'Giá': 'price',
        'Khoảng giá min': 'min_price',
        'Khoảng giá max': 'max_price',
        'Thương hiệu': 'brand',
        'Dòng xe': 'model',
        'Năm đăng ký': 'registration_year',
        'Số Km đã đi': 'mileage_km',
        'Tình trạng': 'condition',
        'Loại xe': 'bike_type',
        'Dung tích xe': 'engine_capacity',
        'Xuất xứ': 'origin',
        'Chính sách bảo hành': 'warranty_policy',
        'Trọng lượng': 'weight'
    })

    df = data.copy()
    df1 = data.copy()

    # ============= CLEANING df1 FOR CLUSTERING =============
    cols_drop = ['title', 'address', 'description', 'Href']
    df1 = df1.drop(columns=[c for c in cols_drop if c in df1.columns], errors='ignore')
    df1 = df1.drop(columns=['warranty_policy', 'weight', 'condition'], errors='ignore')
    df1 = df1.dropna()

    # Clean price
    df1['price'] = (
        df1['price'].astype(str)
        .str.replace('[^0-9]', '', regex=True)
        .replace('', np.nan).astype(float)
    )

    def parse_price(s):
        if pd.isna(s): return np.nan
        s = str(s).lower().replace("tr", "").replace(" ", "")
        try: return float(s) * 1_000_000
        except: return np.nan

    df1['min_price'] = df1['min_price'].apply(parse_price)
    df1['max_price'] = df1['max_price'].apply(parse_price)

    df1 = df1[~(df1['price'] == 0)]

    # Remove invalid engine_capacity
    df1 = df1[~df1['engine_capacity'].astype(str).str.contains("Nhật Bản", na=False)]

    # Clean origin
    df1 = df1[~df1['origin'].astype(str).str.contains('Bảo hành hãng', case=False, na=False)]
    df1['origin'] = df1['origin'].replace(['Đang cập nhật', 'Nước khác'], 'Nước khác')

    # Registration year
    df1['registration_year'] = (
        df1['registration_year'].astype(str)
        .str.lower()
        .str.replace('trước năm', '1980')
        .str.extract('(\d{4})')[0]
    ).astype(float)

    df1.loc[(df1['registration_year'] < 1980) | (df1['registration_year'] > 2025),
            'registration_year'] = np.nan

    df1["age"] = 2025 - df1["registration_year"]

    # Log transforms
    numeric_cols = ['age', 'mileage_km', 'min_price', 'max_price', 'price']
    for c in numeric_cols:
        df1[f"log_{c}"] = np.log1p(df1[c])

    df1 = df1.dropna(subset=numeric_cols)

    return df, df1


@st.cache_data
def compute_clusters(df1):
    # models are accessed from global scope:
    global scaler, kmeans, pca

    num_cols = ['age', 'mileage_km', 'min_price', 'max_price', 'log_price']

    X_scaled = scaler.transform(df1[num_cols])
    df1['cluster_label'] = kmeans.predict(X_scaled)

    pca_points = pca.transform(X_scaled)
    df1['x'] = pca_points[:, 0]
    df1['y'] = pca_points[:, 1]

    return df1, num_cols

# ==========================================================
# LOAD EVERYTHING (CACHED)
# ==========================================================
cosine_sim, vectorizer, tfidf_matrix, kmeans, scaler, pca = load_models()
df, df1 = load_and_clean_data()
df1, num_cols = compute_clusters(df1)



# ==========================================================
# FUNCTIONS
# ==========================================================
def get_similar_bikes(title, top_n=5):
    idx = df.index[df["title"] == title][0]
    scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    return [df.iloc[i[0]]["title"] for i in scores[1:top_n+1]]


def search_by_keyword(keyword, top_n=5):
    keyword_vec = vectorizer.transform([keyword])
    sim_scores = cosine_similarity(keyword_vec, tfidf_matrix).flatten()
    df["score"] = sim_scores
    return df.sort_values(by="score", ascending=False).head(top_n)["title"].tolist()


def preprocess_user_input(price, min_price, max_price, mileage_km, registration_year):
    age = 2025 - registration_year
    log_price = np.log1p(price)
    X = np.array([[age, mileage_km, min_price, max_price, log_price]])
    return scaler.transform(X)


# ==========================================================
# STREAMLIT PAGES
# ==========================================================
st.sidebar.title("🚗 Motorbike Recommendation and Clustering")
page = st.sidebar.selectbox("Menu", ["Home", "Recommendation system", "Clustering analysis"])


if page == "Home":
    st.title("Motorbike Data Science Project")
    # st.write("Hệ thống gợi ý và phân cụm xe máy.")

    st.header('Giới thiệu dự án')
    st.text('''Đây là Project 2 trong khóa đồ án tốt nghiệp Data Science and Machine Learning 2024 lớp DL07_K308 của nhóm 6. \nThành viên nhóm gồm có:
        \n1. Vũ Thị Ngọc Anh \n2. Nguyễn Phạm Quỳnh Anh''')
    st.write("""### Có 2 chủ đề trong khóa học:    
- Topic 1: Dự đoán giá xe máy cũ, phát hiện xe máy bất thường
- Topic 2: Hệ thống gợi ý xe máy dựa trên nội dung, phân cụm xe máy
            """)
    
    st.header('Mục tiêu của dự án')
    # st.text('''1. Tạo mô hình đề xuất xe máy tương tự đối với mẫu xe được chọn hoặc từ khóa tìm kiếm do người dùng cung cấp.\n2. Phân khúc thị trường xe máy''')
    st.write("""
Mục tiêu dự án:
- Tạo mô hình gợi ý xe máy tương tự dựa trên mẫu xe được chọn hoặc từ khóa người dùng cung cấp, giúp hỗ trợ tìm kiếm và lựa chọn xe phù hợp.
- Phân khúc thị trường xe máy dựa trên dữ liệu thu thập được, nhằm nhận diện các nhóm xe đặc trưng theo giá, thương hiệu, phân khối và nhu cầu người dùng.
""")



    st.header("Thu thập dữ liệu")

    st.markdown("""
    **Dữ liệu xe máy đã qua sử dụng** được thu thập từ nền tảng **Chợ Tốt**  
    (trước ngày 01/07/2025).  

    Bộ dữ liệu bao gồm các thông tin sau:

    - **id**: số thứ tự của sản phẩm trong bộ dữ liệu  
    - **Tiêu đề**: tựa đề bài đăng bán sản phẩm  
    - **Giá**: giá bán của xe máy  
    - **Khoảng giá min**: giá sàn ước tính của xe máy  
    - **Khoảng giá max**: giá trần ước tính của xe máy  
    - **Địa chỉ**: địa chỉ giao dịch (phường, quận, thành phố Hồ Chí Minh)  
    - **Mô tả chi tiết**: mô tả thêm về sản phẩm — đặc điểm nổi bật, tình trạng, thông tin khác  
    - **Thương hiệu**: hãng sản xuất (Honda, Yamaha, Piaggio, SYM…)  
    - **Dòng xe**: dòng xe cụ thể (Air Blade, Vespa, Exciter, LEAD, Vario, …)  
    - **Năm đăng ký**: năm đăng ký lần đầu của xe  
    - **Số km đã đi**: số kilomet xe đã vận hành  
    - **Tình trạng**: tình trạng hiện tại (ví dụ: đã sử dụng)  
    - **Loại xe**: Xe số, Tay ga, Tay côn/Moto  
    - **Dung tích xe**: dung tích xi-lanh (ví dụ: Dưới 50cc, 50–100cc, 100–175cc, …)  
    - **Xuất xứ**: quốc gia sản xuất (Việt Nam, Đài Loan, Nhật Bản, ...)  
    - **Chính sách bảo hành**: thông tin bảo hành nếu có  
    - **Trọng lượng**: trọng lượng ước tính của xe  
    - **Href**: đường dẫn tới bài đăng sản phẩm  
    """)


    # with open("data/data_motobikes.xlsx", "rb") as f:
    #     st.download_button(
    #         label="📥 Tải xuống dữ liệu xe máy (Excel)",
    #         data=f,
    #         file_name="data_motobikes.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #     )


elif page == "Recommendation system":
    st.title("Recommendation system")
    # theo xe có sẵn
    st.header("Gợi ý xe theo mẫu có sẵn")
    selected = st.selectbox("Chọn mẫu xe:", df["title"])

    if st.button("Gợi ý"):
        similar_titles = get_similar_bikes(selected)

        # Filter dataframe to only the similar bikes
        result_df = df[df["title"].isin(similar_titles)][
            ["id", "title", "brand", "model", "price", "registration_year"]
        ]

        st.dataframe(result_df, width='stretch')
        
    # theo từ khóa
    st.header("Tìm kiếm theo từ khóa")
    keyword = st.text_input("Nhập từ khóa")
    if st.button("Tìm xe tương tự") and keyword.strip():
        similar_titles = search_by_keyword(keyword)

        # Filter dataframe to only the similar bikes
        result_search_df = df[df["title"].isin(similar_titles)][
            ["id", "title", "brand", "model", "price", "registration_year"]
        ]

        st.dataframe(result_search_df, width='stretch')

elif page == "Clustering analysis":
    st.title("K-Means Motorbike Clustering")

    st.write('''Trong 3 mô hình phân cụm KMeans, Bisect KMeans và Agglomerate thì KMeans với k = 3 cho kết quả phân cụm tốt nhất.
               \nMô hình phân cụm xe được chọn là KMeans với k = 3.''')
    st.write("Trực quan hóa kết quả phân cụm với PCA:")

    # ====== PLOT PCA CLUSTERS ======
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df1["x"], df1["y"], c=df1["cluster_label"], s=10)
    ax.set_title("PCA Visualization")
    st.pyplot(fig)

    # ====== CLUSTER SUMMARY ======
    # st.subheader("Thống kê theo từng cụm:")

    # cluster_summary = (
    #     df1.groupby('cluster_label')
    #        .agg(
    #            count=('cluster_label', 'size'),
    #            avg_price=('price', 'mean'),
    #            avg_age=('age', 'mean'),
    #            avg_mileage=('mileage_km', 'mean')
    #        )
    #        .sort_values('avg_price')
    # )

    # st.dataframe(cluster_summary, width='stretch')

    st.subheader("Thống kê theo từng cụm:")

    cluster_summary = (
        df1.groupby('cluster_label')
        .agg(
            count=('cluster_label', 'size'),
            avg_price=('price', 'mean'),
            avg_age=('age', 'mean'),
            avg_mileage=('mileage_km', 'mean')
        )
        .sort_values('avg_price')
    )

    # Đổi tên cột
    cluster_summary = cluster_summary.rename(columns={
        "count": "Số lượng (xe)",
        "avg_price": "Giá trung bình (VND)",
        "avg_age": "Tuổi trung bình (năm)",
        "avg_mileage": "Số km trung bình (km)"
    })

    # Format số nguyên và thêm dấu phẩy
    cluster_summary["Giá trung bình (VND)"] = (
        cluster_summary["Giá trung bình (VND)"]
            .round(0).astype(int)
            .map(lambda x: f"{x:,}")
    )

    cluster_summary["Số km trung bình (km)"] = (
        cluster_summary["Số km trung bình (km)"]
            .round(0).astype(int)
            .map(lambda x: f"{x:,}")
    )

    st.dataframe(cluster_summary, width='stretch')


    st.subheader("Tóm tắt ý nghĩa từng cụm:")

    st.markdown("""
    - **Cụm 0:** Xe phổ thông – giá rẻ, tuổi xe trung bình, số km trung bình → **nhóm chiếm thị phần lớn nhất**.
    - **Cụm 1:** Xe mới hơn – giá cao hơn, chạy ít hơn → **phân khúc chất lượng tốt**.
    - **Cụm 2:** Xe rất cũ – giá thấp nhất, số km cực cao → **phân khúc xuống cấp hoặc dữ liệu km không chính xác**.
    """)

    bike_labels = {0: "Xe phổ thông giá rẻ",
                   1: "Xe tương đối mới",
                   2: "Xe cũ xuống cấp hoặc dữ liệu cung cấp không chính xác"}


    # ====== CLUSTER NEW BIKE ======
    st.subheader("Phân cụm xe mới")

    st.write("Vui lòng nhập các thông số của xe cần xác định")

    price = st.number_input("Giá xe (VND)", min_value=500_000, step=100_000)
    min_price = st.number_input("Khoảng giá min", min_value=500_000, step=100_000)
    max_price = st.number_input("Khoảng giá max", min_value=500_000, step=100_000)
    mileage_km = st.number_input("Số km đã đi", min_value=0, step=100)
    registration_year = st.slider("Năm đăng ký", 1980, 2025)

    if st.button("Phân cụm"):
        X_new = preprocess_user_input(price, min_price, max_price, mileage_km, registration_year)
        cluster = int(kmeans.predict(X_new)[0])
        st.success(f"Xe thuộc cụm số **{cluster}**")

        st.write(bike_labels.get(cluster, "Không có mô tả cho cụm này"))
