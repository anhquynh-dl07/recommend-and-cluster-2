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
# st.set_page_config(page_title="Motorbike Recommendation and Motorbike Segmentation by Clustering", layout="wide")
# st.title("Motorbike Recommendation and Motorbike Segmentation by Clustering")

menu = ["Giới thiệu", "Bài toán nghiệp vụ", "Đánh giá mô hình và Báo cáo", "Gợi ý mẫu xe tương tự", "Phân cụm phân khúc xe máy"]
page = st.sidebar.selectbox('Menu', menu)


if page == 'Giới thiệu':
    st.title("Hệ thống gợi ý xe máy tương tự và phân cụm xe máy")
    # st.markdown("Ứng dụng cho phép: \n1) Gợi ý mẫu xe máy tương tự (nhập thông số xe) \n2) Xác định phân khúc xe máy bằng phương pháp phân cụm (nhập thông số hoặc upload file)")
    st.image("xe_may_cu.jpg", caption="Xe máy cũ")
    st.subheader("[Trang chủ Chợ Tốt](https://www.chotot.com/)")
    
    st.header('Giới thiệu dự án')
    st.markdown('''Đây là dự án xây dựng hệ thống hỗ trợ **gợi ý mẫu xe máy tương tự** và **phân khúc xe máy bằng phương pháp phân cụm** trên nền tảng *Chợ Tốt* - trong khóa đồ án tốt nghiệp Data Science and Machine Learning 2024 lớp DL07_K308 của nhóm 6. \nThành viên nhóm gồm có:
        \n1. Vũ Thị Ngọc Anh \n2. Nguyễn Phạm Quỳnh Anh''')
    
    st.header('Mục tiêu của dự án')
    # st.text('''1. Tạo mô hình đề xuất xe máy tương tự đối với mẫu xe được chọn hoặc từ khóa tìm kiếm do người dùng cung cấp.\n2. Phân khúc thị trường xe máy bằng phương pháp phân cụm''')
    st.markdown("""
        ### Mục tiêu của dự án:
        **1. Xây dựng mô hình đề xuất thông minh:**
        - Đề xuất các mẫu xe máy tương đồng cho một mẫu được chọn hoặc theo từ khóa tìm kiếm của người dùng.
        - Kết hợp nhiều nguồn thông tin (thông số kỹ thuật, hình ảnh, mô tả, giá, đánh giá) để tăng độ chính xác.
             
        **2. Phân khúc thị trường xe máy:**
        - Phân loại sản phẩm theo nhóm theo tệp giá, tuổi xe, khoảng giá tối thiểu/ tối đa, 
        giúp cho việc định giá xe hiệu quả hơn và chiến lược marketing hiệu quả hơn.
        """)

    st.subheader('Phân công công việc')
    st.write("""
        - Xử lý dữ liệu: Ngọc Anh và Quỳnh Anh
        - Gợi ý xe máy bằng Gensim: Quỳnh Anh
        - Gợi ý xe máy bằng Cosine similarity: Ngọc Anh
        - Phân khúc xe máy bằng phương pháp phân cụm: Ngọc Anh
        - Làm slide: Ngọc Anh và Quỳnh Anh
        - Giao diện streamlit: Quỳnh Anh

        """)
    
elif page == "Bài toán nghiệp vụ":
    st.header("Bài toán nghiệp vụ")

    st.markdown("""

        ### Vấn đề nghiệp vụ
        - Người dùng gặp khó khăn khi tìm xe phù hợp trong hàng trăm lựa chọn.
        - Chưa có hệ thống gợi ý xe tương tự khi người dùng chọn một mẫu cụ thể hoặc tìm kiếm theo từ khóa.
        - Thị trường xe máy rất đa dạng → khó nhận diện các phân khúc rõ ràng.
        - Cần hệ thống gợi ý & phân khúc tự động để hỗ trợ người dùng và đội ngũ phân tích.


        ### Bài toán đặt ra
        - Xây dựng mô hình **Gợi ý xe tương tự**:
            - Sử dụng các đặc trưng từ mô tả xe và thông số kỹ thuật
            - Gợi ý các mẫu xe tương tự với xe được chọn hoặc theo từ khóa tìm kiếm.

        - Xây dựng mô hình **Phân khúc thị trường xe bằng phương pháp phân cụm**:
            - Phân cụm thị trường xe máy dựa các đặc trưng giá xe, tuổi xe, số km đã chạy, khoảng giá tối thiểu, tối đa.
            - Giúp nhận diện các nhóm sản phẩm theo các phân khúc khác nhau


        ### Phạm vi triển khai
        - **Tiền xử lý dữ liệu và chuẩn hóa**:
            - Chuẩn hóa các thông số của xe.
            - Làm sạch dữ liệu và chuẩn hóa trường thông tin cho mô hình.

        - **Trích xuất đặc trưng văn bản**:
            - Sử dụng **TF-IDF Vectorizer** để mã hóa mô tả và thông tin kỹ thuật.
            - Tính độ tương đồng bằng **gensim similarity** và **cosine similarity**.
            - Chọn phương pháp cho **điểm cao hơn** và **nghĩa đúng hơn** để đưa vào hệ thống gợi ý.

        - **Phân cụm thị trường (Clustering)**:
            - Thử nghiệm trên các thuật toán:  
                - **KMeans**  
                - **Agglomerative Clustering**  
                - **Bisecting KMeans**
            - Đánh giá bằng inertia, silhouette score, tính diễn giải.
            - **Chọn KMeans** vì có hiệu suất ổn định, dễ diễn giải và ranh giới cụm phù hợp hơn với dữ liệu.

        - **Xây dựng GUI trên Streamlit**:
            - Cho phép người dùng *chọn xe trong danh sách* hoặc **nhập mô tả xe** → trả về **danh sách mẫu xe tương tự có trong sàn**.
            - Cho phép **nhập tên xe** → hiển thị **xe thuộc cụm/phân khúc nào**.


        ### Thu thập dữ liệu
        - Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).
        - Bộ dữ liệu bao gồm các thông tin sau:
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


elif page == "Đánh giá mô hình và Báo cáo":    
    st.header("Đánh giá mô hình và Báo cáo")  

    st.subheader("I. Thống kê mô tả sơ bộ")

    # st.markdown("""
    # **1. Thống kê mô tả sơ bộ** 
    # """)
    st.markdown("""        
    Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).  
                """)
    # --- Vẽ biểu đồ ---

    # Hiển thị 4 biểu đồ dạng lưới 2x2
    col1, col2 = st.columns(2)
    with col1:
        st.image("brand_grouped_count.png")
        st.image("age_bin_stats.png")

    with col2:
        st.image("price_bin_stats.png")
        st.image("mileage_bin_stats.png")

    st.subheader("II. Mô hình gợi ý xe máy tương tự")

    # with open("data/data_motobikes.xlsx", "rb") as f:
    #     st.download_button(
    #         label="📥 Tải xuống dữ liệu xe máy (Excel)",
    #         data=f,
    #         file_name="data_motobikes.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #     )
    st.markdown('#### 1. Hướng xử lý')
    st.write('''
             - Chuẩn hóa và làm sạch dữ liệu.
             - Chia khoảng một số đặc trưng kiểu số để tạo thêm các đặc trưng phân loại mới (khoảng giá, tình trạng dựa theo số km chạy, tuổi xe, dung tích xe)
             - Gom các đặc trưng phân loại thành biến text
             - Làm sạch text và tokenize, xây dựng ma trận tương đồng (sparse matrix) giữa các văn bản để đánh giá mức độ giống nhau
             - Tính độ tương đồng bằng gensim và cosine similarity
                 - Trường hợp 1: gợi ý xe theo id sản phẩm được chọn
                    - Người dùng chọn xe từ danh sách xe trong tập dữ liệu
                    - Dựa trên ma trận tương đồng, tìm các xe có similarity score cao nhất.
                    - Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 7000 mẫu trong tập dữ liệu và tính trung bình.

                 - Trường hợp 2: gợi ý xe theo cụm từ khóa tìm kiểm (vd: “honda vision xanh dưới 15 triệu”)
                    - Người dùng nhập từ khóa tìm kiếm. 
                    - Xử lý từ khóa và chuyển từ khóa thành vector số dựa trên từ điển và TF-IDF
                    - Tính độ tương đồng giữa từ khóa và tất cả xe trong dữ liệu. 
                    - Sắp xếp và lấy ra 5 xe gợi ý phù hợp nhất.
                    - Cho danh sách 10 cụm từ khóa tìm kiếm. Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 10 cụm từ trên và tính trung bình
             ''')
    
    st.markdown('#### 2. Kết quả')
    st.write('Giữa 02 mô hình Gensim và Cosine similarity, Cosine similarity, trong cả 2 trường hợp chọn xe có sẵn hoặc tìm bằng từ khóa, cho điểm tương đồng trung bình cao hơn so với Gensim và cho các gợi ý sát nghĩa hơn Gensim.\nMô hình dùng để dự đoán xe trong ứng dụng này là Cosine similarity.') 

    st.subheader("III. Mô hình phân khúc xe máy bằng phương pháp phân cụm")
    
    st.markdown('#### 1. Xử lý dữ liệu')
    st.write('Dữ liệu được làm sạch, các đặc trưng biến số liên tục như giá, khoảng giá thấp nhất, lớn nhất, tuổi xe, số km đã đi được chọn để tạo mô hình phân cụm')

    st.markdown('#### 2. Phân cụm bằng các phương pháp khác nhau')
    st.write('''
    Mô hình phân cụm được xây dựng trên 02 môi trường: máy học truyền thống (sci-kit learn) và PySpark.
    - Máy học truyền thống: KMeans, Bisect Kmeans, Agglomerative clustering
    - PySpark: Kmeans, Bisecting Kmeans, GMM.

    ''')

    st.markdown('#### 3. Kết quả')
    st.markdown('''
    Số cụm được tạo thành trên mô hình máy học truyền thống: **03 cụm**
    Số cụm được tạo thành trên PySpark: **02 cụm**
             
    KMeans trên môi trường máy học truyền thống cho kết quả silhoutte score cao nhất và kết quả phân cụm dễ diễn giải hơn.
    
    **Phân loại phân khúc xe**:                
    1/ Cụm 0: Phân khúc Xe Phổ Thông – Trung cấp (Mid-range Popular Motorcycles): Xe tuổi trung bình, giá vừa phải, phù hợp đại đa số người mua.   
    2/ Cụm 1: Phân khúc Xe Cao Cấp – Premium / High-end Motorcycles: Rõ ràng là các dòng SH, Vespa cao cấp, phân khối lớn, xe mới chạy ít.          
    3/ Cụm 2: Phân khúc Xe Cũ – Tiết Kiệm (Budget Used Motorcycles): Giá rẻ nhất, xe tuổi cao, chạy nhiều — phù hợp khách cần xe rẻ để di chuyển cơ bản.
    ''')
    st.write('''Trong 3 mô hình phân cụm KMeans, Bisect KMeans và Agglomerate thì KMeans với k = 3 cho kết quả phân cụm tốt nhất.
            nên mô hình phân cụm xe được sử dụng trong ứng dụng này là KMeans với k = 3.''')

    st.markdown('#### 4. Thống kê theo từng cụm:')

    st.write('Trực quan hóa')
    st.image('pca_clusters.png')

    cluster_summary = (
        df1.groupby('cluster_label')
        .agg(
            count=('cluster_label', 'size'),
            avg_price=('price', 'mean'),
            avg_age=('age', 'mean'),
            avg_mileage=('mileage_km', 'mean')
        )
        .sort_values('cluster_label')
    )


    # Rename the index (cluster_label → Nhãn cụm xe)
    cluster_summary = cluster_summary.rename_axis("Nhãn cụm xe")

    # Rename columns
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


elif page == "Gợi ý mẫu xe tương tự":
    st.title("Gợi ý mẫu xe tương tự")
    # theo xe có sẵn
    st.header("Gợi ý xe theo mẫu có sẵn")
    selected = st.selectbox("Chọn mẫu xe:", df["title"])

    if st.button("Gợi ý"):
        similar_titles = get_similar_bikes(selected)
        # Show thêm chính bảng ghi của xe đã chọn
        selected_row = df[df["title"] == selected][
            ["id", "title", "description", "price", "brand", "model",
            "bike_type", "origin", "condition", "mileage_km",
            "registration_year", "engine_capacity"]
        ]

        selected_row = selected_row.rename(columns={
            "id": "id",
            "title": "Tiêu đề",
            "description": "Mô tả",
            "price": "Giá",
            "brand": "Hãng",
            "model": "Dòng xe",
            "bike_type": "Loại xe",
            "origin": "Xuất xứ",
            "condition": "Tình trạng",
            "mileage_km": "Số km",
            "registration_year": "Năm đăng ký",
            "engine_capacity": "Dung tích xe"
        })

        st.markdown("**Xe bạn đã chọn:**")
        st.dataframe(selected_row, width='stretch')

        # Filter dataframe to only the similar bikes
        result_df = df[df["title"].isin(similar_titles)][
            ["id", "title", "description", "price", "brand", "model", "bike_type", "origin", "condition", "mileage_km" ,"registration_year", "engine_capacity"]
        ]
        result_df = result_df.rename(columns={
                "id": "id",
                "title": "Tiêu đề",
                "description": "Mô tả",
                "price": "Giá",
                "brand": "Hãng",
                "model": "Dòng xe",
                "bike_type": "Loại xe",
                "origin": "Xuất xứ",
                "condition": "Tình trạng",
                "mileage_km": "Số km",
                "registration_year": "Năm đăng ký",
                "engine_capacity": "Dung tích xe"
            })
        
        st.markdown("**Các mẫu xe gợi ý:**")
        st.dataframe(result_df, width='stretch')
        
    # theo từ khóa
    st.header("Tìm kiếm theo từ khóa")
    keyword = st.text_input("Nhập từ khóa")
    if st.button("Tìm xe tương tự") and keyword.strip():
        similar_titles = search_by_keyword(keyword)

        # Filter dataframe to only the similar bikes
        result_search_df = df[df["title"].isin(similar_titles)][
            ["id", "title", "description", "price", "brand", "model", "bike_type", "origin", "condition", "mileage_km" ,"registration_year", "engine_capacity"]
        ]
        result_search_df = result_search_df.rename(columns={
                "id": "id",
                "title": "Tiêu đề",
                "description": "Mô tả",
                "price": "Giá",
                "brand": "Hãng",
                "model": "Dòng xe",
                "bike_type": "Loại xe",
                "origin": "Xuất xứ",
                "condition": "Tình trạng",
                "mileage_km": "Số km",
                "registration_year": "Năm đăng ký",
                "engine_capacity": "Dung tích xe"
            })


        st.dataframe(result_search_df, width='stretch')

elif page == "Phân cụm phân khúc xe máy":
    st.title("Phân cụm phân khúc xe máy")

    st.markdown("""
    <style>
    .cluster-card {
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 15px;
        color: white;
        font-size: 16px;
    }
    .cluster-0 {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
    }
    .cluster-1 {
        background: linear-gradient(135deg, #1976D2, #0D47A1);
    }
    .cluster-2 {
        background: linear-gradient(135deg, #F57C00, #E65100);
    }
    .cluster-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .cluster-desc {
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)


    # # st.markdown("""
    # # - **Cụm 0:** Xe phổ thông – giá rẻ, tuổi xe trung bình, số km trung bình → **nhóm chiếm thị phần lớn nhất**.
    # # - **Cụm 1:** Xe mới hơn – giá cao hơn, chạy ít hơn → **phân khúc chất lượng tốt**.
    # # - **Cụm 2:** Xe rất cũ – giá thấp nhất, số km cực cao → **phân khúc xuống cấp hoặc dữ liệu km không chính xác**.
    # # """)

    # bike_labels = {0: "Xe phổ thông giá rẻ, tuổi xe trung bình",
    #                1: "Xe tương đối mới, phân khúc cao cấp",
    #                2: "Xe cũ xuống cấp hoặc dữ liệu cung cấp không chính xác"}


    # ====== CLUSTER NEW BIKE ======
    st.header("Phân cụm xe mới")

    st.write("Vui lòng nhập các thông số của xe cần xác định")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Giá xe (VND)", min_value=500_000, step=100_000, value=1_000_000)
        min_price = st.number_input("Khoảng giá min", min_value=500_000, step=100_000, value=800_000)

    with col2:
        max_price = st.number_input("Khoảng giá max", min_value=500_000, step=100_000, value=1_200_000)
        mileage_km = st.number_input("Số km đã đi", min_value=0, step=100, value=1000)

    registration_year = st.slider("Năm đăng ký", 1980, 2025)

    if st.button("Phân cụm"):
        X_new = preprocess_user_input(price, min_price, max_price, mileage_km, registration_year)
        cluster = int(kmeans.predict(X_new)[0])
        st.success(f"Xe thuộc cụm số **{cluster}**")

        # st.write(bike_labels.get(cluster, "Không có mô tả cho cụm này"))

        # ======= HIỂN THỊ THẺ GIẢI THÍCH CỤM THEO KẾT QUẢ =======

        cluster_cards = {
            0: """
                <div class="cluster-card cluster-0">
                    <div class="cluster-title">Cụm 0 – Xe phổ thông giá rẻ</div>
                    <div class="cluster-desc">
                        Giá thấp – tuổi xe trung bình – số km chạy vừa phải.<br>
                        Phân khúc xe phổ thông, phù hợp đa số người mua.
                    </div>
                </div>
            """,
            1: """
                <div class="cluster-card cluster-1">
                    <div class="cluster-title">Cụm 1 – Xe cao cấp / ít chạy</div>
                    <div class="cluster-desc">
                        Xe mới – ít km – giá cao.<br>
                        Các dòng SH, Vespa, xe cao cấp, tình trạng tốt.
                    </div>
                </div>
            """,
            2: """
                <div class="cluster-card cluster-2">
                    <div class="cluster-title">Cụm 2 – Xe cũ / giá rẻ</div>
                    <div class="cluster-desc">
                        Giá thấp nhất – km rất cao – tuổi xe lớn.<br>
                        Phân khúc xe đã cũ hoặc có dấu hiệu xuống cấp.
                    </div>
                </div>
            """
        }

        # Hiển thị card tương ứng
        st.markdown(cluster_cards.get(cluster, ""), unsafe_allow_html=True)

