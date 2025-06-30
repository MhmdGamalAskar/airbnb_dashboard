
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from collections import Counter
from fuzzywuzzy import process, fuzz
from scipy import stats
from scipy.stats import mannwhitneyu
import calendar
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Airbnb Data Science", layout="wide")

st.markdown(
    """
    <img src="https://mohamedirfansh.github.io/Airbnb-Data-Science-Project/images/seattle.jpg">
    """,
    unsafe_allow_html=True
)

import pandas as pd

@st.cache_data  # Cache to avoid reloading every time
def load_data():
    return pd.read_csv(
        "/Users/mhmdgamal/Downloads/Airbnb Data/Listings.csv",
        encoding="ISO-8859-1",
        low_memory=False
    )

df = load_data()


# 🧹 Drop unnecessary columns early (optional)
columns_to_drop = [
    'name', 'host_location', 'host_response_time',
    'host_response_rate', 'host_acceptance_rate',
    'host_has_profile_pic', 'host_identity_verified'
]
df_clean = df.drop(columns=columns_to_drop)



def calculate_statistics(df):
    stats_df = pd.DataFrame(columns=[
        "Count", "Missing", "Unique", "Dtype", "Numerical",
        "Mode", "Mean", "Min", "25%", "50%", "75%", "Max", "Skew"
    ])
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_df.loc[col] = [
                df[col].count(),
                df[col].isna().sum(),
                len(df[col].unique()),
                df[col].dtype,
                "✅",
                df[col].mode()[0] if not df[col].mode().empty else "",
                df[col].mean(),
                df[col].min(),
                df[col].quantile(0.25),
                df[col].quantile(0.5),
                df[col].quantile(0.75),
                df[col].max(),
                df[col].skew()
            ]
        else:
            stats_df.loc[col] = [
                df[col].count(),
                df[col].isna().sum(),
                len(df[col].unique()),
                df[col].dtype,
                "❌",
                df[col].mode()[0] if not df[col].mode().empty else "",
                "", "", "", "", "", "", ""
            ]
    return stats_df

class PriceConversion:
    def __init__(self, api_key, exchange_file="exchange_rates.json"):
        self.api_key = api_key
        self.exchange_file = exchange_file
        self.exchange_rates = {}
        self.currency_map = {
            "Paris": "EUR",
            "New York": "USD",
            "Bangkok": "THB",
            "Rio de Janeiro": "BRL",
            "Sydney": "AUD",
            "Istanbul": "TRY",
            "Rome": "EUR",
            "Hong Kong": "HKD",
            "Mexico City": "MXN",
            "Cape Town": "ZAR"
        }

    def validate_cities(self, df):
        unique_cities = set(df["city"].unique())
        unknown_cities = unique_cities - set(self.currency_map.keys())
        if unknown_cities:
            print(f"⚠️ Warning: These cities are missing in currency_map: {unknown_cities}")


    def fetch_exchange_rates(self):
        print("⏳ Fetching exchange rates...")
        url = "https://api.currencyfreaks.com/latest"
        params = {"apikey": self.api_key}
        response = requests.get(url, params=params)
        data = response.json()
        usd_rates = data.get("rates", {})
        self.exchange_rates = {
            currency: round(1 / float(usd_rates[currency]), 4)
            for currency in self.currency_map.values()
            if currency in usd_rates
        }
        with open(self.exchange_file, "w") as f:
            json.dump(self.exchange_rates, f, indent=2)
        print("\n🔁 Exchange Rates (to USD):")
        for cur, rate in self.exchange_rates.items():
            print(f" - {cur}: {rate}")
        print("\n✔️ Exchange rates saved to", self.exchange_file)

    def load_exchange_rates(self):
        with open(self.exchange_file, "r") as f:
            self.exchange_rates = json.load(f)

    def fit(self, df):
        self.validate_cities(df)
        self.fetch_exchange_rates()
        return self

    def transform(self, df):
        if not self.exchange_rates:
            self.load_exchange_rates()

        def convert(row):
            city = row["city"]
            price = row["price"]
            currency = self.currency_map.get(city)
            rate = self.exchange_rates.get(currency)
            if rate:
                return round(price * rate, 2)
            return None

        df["price_usd"] = df.apply(convert, axis=1)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

# 🟢 بعد تحميل البيانات، استخدم التحويل:
API_KEY = "82560b3c831e428f9fe64312b95f20de"
Price_pipeline = PriceConversion(api_key=API_KEY)
df_clean = Price_pipeline.fit_transform(df_clean)










class PropertyTyper:
    def __init__(self, col_name='property_type', out_col=None, threshold=95, inplace=False, verbose=False):
        self.col_name = col_name
        self.out_col = out_col or (col_name if inplace else col_name + '_cleaned')
        self.threshold = threshold
        self.inplace = inplace
        self.verbose = verbose
        self.mapping = {}
        self.changes = []
        self.token_stats = {}

    def _basic_clean(self, series):
        return (series.astype(str)
                      .str.lower()
                      .str.strip()
                      .str.replace(r'\s+', ' ', regex=True))

    def _tokenize_and_analyze(self, cleaned_series):
        tokens = cleaned_series.str.split()
        all_words = [word for sublist in tokens for word in sublist]
        self.token_stats = Counter(all_words)
        if self.verbose:
            print(f"Top 10 frequent tokens: {self.token_stats.most_common(10)}")

    def _map_categories(self, text) :
        
        if pd.isnull(text):
            return 'Other'

        v = str(text).lower()

        
        if any(x in v for x in [
            'apartment', 'condominium', 'loft',
            'entire home/apt', 'entire place', 'entire floor',
            'entire guest suite', 'private room in guest suite',
            'room in guest suite', 'suite'
        ]):
            return 'Apartment'

        
        elif any(x in v for x in [
            'house', 'townhouse', 'cottage', 'bungalow', 'cabin',
            'chalet', 'barn', 'casa', 'casa particular',
            'vacation home', 'in-law'
        ]):
            return 'House'

       
        elif 'villa' in v:
            return 'Villa'

        
        elif any(x in v for x in [
            'hotel', 'hostel', 'aparthotel', 'boutique',
            'room in hotel', 'shared room in hotel',
            'room in hostel', 'private room in hostel', 'shared room in hostel',
            'dorm', 'entire hostel', 'entire dorm',
            'pension', 'room in pension', 'shared room in pension',
            'private room in pension'
        ]):
            return 'Hotel'

        
        
        elif any(x in v for x in ['bed and breakfast', 'b&b', 'minsu']):
            return 'Bed & Breakfast'

       
        elif 'resort' in v:
            return 'Resort'

        
        elif any(x in v for x in [
            'farm stay', 'camper/rv', 'camper', 'rv', 'campsite', 'tipi',
            'tent', 'island', 'boat', 'treehouse', 'yurt', 'igloo',
            'cave', 'bus', 'train', 'castle', 'dome', 'windmill',
            'hut', 'nature lodge', 'holiday park', 'pousada', 'kezhan',
             'parking space'
        ]):
            return 'Unique Stay'

        else:
            return 'Other'
            
    def fit(self, df):
        cleaned = self._basic_clean(df[self.col_name])
        self._tokenize_and_analyze(cleaned)

        unique_values = sorted(cleaned.unique())
        for name in unique_values:
            if name in self.mapping:
                continue
            matches = process.extract(name, unique_values, scorer=fuzz.token_sort_ratio)
            for match, score in matches:
                if score >= self.threshold:
                    self.mapping[match] = name
                    if match != name:
                        self.changes.append((match, name, score))

        if self.verbose:
            print(f"[PropertyTypeCleaner] Reduced from {len(unique_values)} to {len(set(self.mapping.values()))} unique values.")
            if self.changes:
                print("Changes made:")
                for old, new, score in sorted(self.changes, key=lambda x: -x[2]):
                    print(f"  '{old}' → '{new}' ({score}%)")

        return self

    def transform(self, df):
        cleaned = self._basic_clean(df[self.col_name])
        cleaned_mapped = cleaned.map(self.mapping).fillna(cleaned)

        if self.inplace:
            df[self.col_name] = cleaned_mapped
            target_col = self.col_name
        else:
            df[self.out_col] = cleaned_mapped
            target_col = self.out_col

        df['property_category'] = df[target_col].apply(self._map_categories)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
property_pipeline = PropertyTyper(out_col='property_category', inplace=False, verbose=True)
df_clean = property_pipeline.fit_transform(df_clean)



district_stats = df_clean.groupby('district').agg(
    count=('price_usd', 'count'),
    mean_price=('price_usd', 'mean'),
    total_revenue=('price_usd', 'sum')
).reset_index()

# Sort districts by total revenue and get the top 10 revenue-generating districts
top_revenue = district_stats.sort_values(by='total_revenue', ascending=False).head(10)

# Sort districts by growth potential (low volume, high price)
growth_opportunities = district_stats[district_stats['count'] < 500].sort_values(by='mean_price', ascending=False).head(10)




class InstantBookable:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.mapping = {
            't': 1, 'true': 1, 'yes': 1, '1': 1,
            'f': 0, 'false': 0, 'no': 0, '0': 0
        }

    def clean_value(self, val):
        if pd.isnull(val):
            return 0
        val_str = str(val).strip().lower()
        if val_str in self.mapping:
            return self.mapping[val_str]
        if self.verbose:
            print(f" Warning: Unexpected value '{val}' in instant_bookable → setting to 0")
        return 0

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        df['instant_bookable_cleaned'] = df['instant_bookable'].apply(self.clean_value)
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
instant_bookable_pipeline = InstantBookable(verbose=True)
df_clean = instant_bookable_pipeline.fit_transform(df_clean)









# 🌐 Language selector
language = st.sidebar.selectbox("🌐 Choose Language / اختر اللغة", ["English", "Arabic"])

# Airbnb Logo
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_Bélo.svg", width=120)

# ✅ ✅ ✅ PROJECT TAG (move it here)
project_tag = "🎓 Airbnb Data Science Project" if language == "English" else "🎓 مشروع تحليل بيانات Airbnb"
st.sidebar.success(project_tag)

# Developed by:
if language == "English":
    st.sidebar.markdown("👥 **Developed by:**")
    st.sidebar.markdown("""
- Mohamed Gamal Askar  
- Mohamed Essam  
- Sondos Manei  
    """)
else:
    st.sidebar.markdown("👥 **تم بواسطة:**")
    st.sidebar.markdown("""
- محمد جمال عسكر  
- محمد عصام  
- سندس منيع  
    """)

# Table of Contents
st.sidebar.markdown("📂 **Table of Contents**" if language == "English" else "📂 **جدول المحتويات**")

# Menu Items
menu_items = [
    ("🏠 Introduction", "🏠 المقدمة"),
    ("💡 Motivation", "💡 الدافع"),
    ("📊 Data Overview", "📊 نظرة عامة"),
    ("🧹 Data Cleansing", "🧹 تنظيف البيانات"),
    ("📌 Key Business Questions", "📌 الأسئلة الرئيسية"),
    ("📈 Visualizations", "📈 الرسوم البيانية")
]

menu_labels = [item[0] if language == "English" else item[1] for item in menu_items]
menu = st.sidebar.radio("", menu_labels, label_visibility="collapsed")







#===========================
if menu.startswith("🏠"):
    if language == "English":
        # Title + Caption
        st.title("🌍 Airbnb Market Analysis Dashboard")
        st.caption("This dashboard was developed as a graduation project to explore Airbnb data and help hosts and travelers make better decisions.")
        
        # Divider line
        st.markdown("---")

        # Overview
        st.markdown("""
### ✨ What is Airbnb?

Airbnb is a popular online marketplace that connects people who want to rent out their homes or spare rooms with guests seeking accommodations.  
Founded in **2008** in **San Francisco, California**, Airbnb has revolutionized the travel industry by offering a flexible alternative to traditional hotels.  
Today, it operates in **more than 220 countries and regions**, with over **7 million listings** worldwide.  
It empowers property owners to generate income while giving travelers unique and affordable lodging experiences.
""")

        # About the project
        st.markdown("""
### 🎓 About this Project

This dashboard was created as part of a **graduation data science project**. It leverages real Airbnb listings data to explore:
- What drives pricing?
- How can hosts increase revenue?
- How can travelers find the best deals?

It aims to combine **business insight** with **data science techniques** to support both Airbnb hosts and guests.
""")

    else:
        # عنوان + وصف
        st.title("🌍 لوحة تحليل سوق Airbnb")
        st.caption("تم تطوير هذه اللوحة كمشروع تخرج لاستكشاف بيانات Airbnb ومساعدة المضيفين والمسافرين على اتخاذ قرارات أفضل.")

        # خط فاصل
        st.markdown("---")

        # ما هي Airbnb
        st.markdown("""
### ✨ ما هي Airbnb؟

Airbnb هي منصة شهيرة على الإنترنت تربط الأشخاص الذين يرغبون في تأجير منازلهم أو غرف إضافية مع الضيوف الباحثين عن أماكن إقامة.  
تأسست عام **2008** في **سان فرانسيسكو، كاليفورنيا**، وقد أحدثت ثورة في صناعة السفر من خلال تقديم بديل مرن للفنادق التقليدية.  
اليوم تعمل في **أكثر من 220 دولة وإقليم**، مع أكثر من **7 ملايين إعلان** حول العالم.  
تمكّن المالكين من تحقيق دخل إضافي، وتمنح المسافرين تجارب إقامة فريدة وبأسعار مناسبة.
""")

        # عن المشروع
        st.markdown("""
### 🎓 عن المشروع

تم تطوير هذه اللوحة التفاعلية كجزء من **مشروع تخرج في علم البيانات**. تعتمد على بيانات حقيقية من Airbnb لتحليل:
- ما العوامل التي تؤثر في التسعير؟
- كيف يمكن للمضيفين زيادة أرباحهم؟
- كيف يمكن للمسافرين الحصول على أفضل الصفقات؟

يهدف المشروع إلى دمج **التحليل العملي** مع **أدوات علم البيانات** لدعم كل من المضيفين والمسافرين.
""")
        










#===========================



elif menu.startswith("💡"):
    if language == "English":
        st.header("💡 Practical Motivation")
        st.markdown("---")

        st.markdown("""
In recent years, **Airbnb has transformed the hospitality industry**, offering both convenience and cost-effective options to travelers, and income opportunities for hosts.  
As short-term rentals grow in popularity, it becomes essential to understand the **factors that drive pricing and success on Airbnb**.

With thousands of listings available, users face information overload. This raises essential questions:

#### 🤔 Hosts may ask:
- What type of property should I invest in to maximize rental income?
- Which features most influence guest interest and price?

#### 🤔 Travelers may wonder:
- How can I find the best value listings with the amenities I care about?

🎯 **This project aims to:**
- Identify the top factors influencing listing prices.
- Highlight features common to premium listings.
- Empower hosts with pricing insights to maximize profit.
- Help travelers find budget-friendly, feature-rich stays.

---

🔎 **Why this matters:**  
Understanding Airbnb pricing isn't just a host's concern. It's a powerful case study for applying **data science to real-world business problems** — blending economics, human behavior, and analytics.
        """)

    else:
        st.header("💡 الدافع العملي")
        st.markdown("---")

        st.markdown("""
في السنوات الأخيرة، أحدثت **Airbnb تحولًا كبيرًا في عالم الإقامة والسفر**، إذ وفرت للضيوف حلولًا مرنة واقتصادية، وأتاحت للمضيفين فرصًا رائعة لتحقيق دخل من خلال تأجير ممتلكاتهم.  
ومع تزايد شعبية الإيجارات قصيرة الأجل، أصبح من الضروري فهم **العوامل المؤثرة في تسعير الإعلانات ونجاحها**.

ومع وجود آلاف الإعلانات، يواجه المستخدمون صعوبة في الاختيار، مما يثير تساؤلات هامة:

#### 🤔 المضيفون قد يتساءلون:
- ما نوع العقار الأنسب للاستثمار لتحقيق أكبر عائد؟
- ما هي الميزات التي تؤثر أكثر على السعر وجذب الضيوف؟

#### 🤔 المسافرون قد يتساءلون:
- كيف أجد إعلانًا مميزًا بسعر مناسب وبالمزايا التي أفضلها؟

🎯 **يهدف هذا المشروع إلى:**
- تحديد العوامل الأهم التي تؤثر على تسعير الإعلانات.
- إبراز الميزات المشتركة بين العقارات الفاخرة.
- تزويد المضيفين برؤى تسعيرية لزيادة الأرباح.
- مساعدة المسافرين على العثور على خيارات مريحة واقتصادية.

---

🔎 **لماذا هذا مهم؟**  
تحليل تسعير Airbnb لا يخدم المضيف فقط، بل هو أيضًا **نموذج تطبيقي حقيقي لعلم البيانات** في حل مشاكل تجارية — يجمع بين السلوك البشري والتحليل الاقتصادي.
        """)













#===========================

elif menu.startswith("📊"):
    st.header("📊 Data Overview" if language == "English" else "📊 نظرة عامة على البيانات")
    st.markdown("---")

    # وصف عام
    if language == "English":
        st.subheader("🗂️ Dataset Description")
        st.markdown("""
The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews), and it contains detailed Airbnb listings in **Seattle**.  
It includes **over 279,000 records** with features such as listing details, host information, location, pricing, availability, and guest reviews.

It is:
- Structured tabular data.
- A mix of **categorical**, **numerical**, and **geospatial** features.
- Real-world data used to support pricing and business insights.
        """)
    else:
        st.subheader("🗂️ وصف مجموعة البيانات")
        st.markdown("""
تم الحصول على مجموعة البيانات من [Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews)، وهي تحتوي على إعلانات Airbnb تفصيلية في مدينة **سياتل**.  
تتضمن أكثر من **279,000 سجل** تحتوي على معلومات مثل تفاصيل العقار، بيانات المضيف، الموقع الجغرافي، التسعير، وتقييمات الضيوف.

البيانات تتضمن:
- بيانات منظمة في شكل جداول.
- مزيج من **بيانات فئوية ورقمية وجغرافية**.
- بيانات حقيقية تُستخدم لتحليل الأسعار والرؤى التجارية.
        """)

    # عدد الصفوف والأعمدة
    rows, cols = df_clean.shape
    if language == "English":
        st.info(f"✅ Dataset contains **{rows:,} rows** and **{cols:,} columns**.")
    else:
        st.info(f"✅ تحتوي مجموعة البيانات على **{rows:,} صف** و **{cols:,} عمود**.")

    # عرض أهم الأعمدة (15 فقط)
    main_columns = df_clean.columns.tolist()[:15]
    if language == "English":
        st.subheader("🧾 Preview of Key Columns")
        st.dataframe(df_clean[main_columns].head())
    else:
        st.subheader("🧾 معاينة الأعمدة الرئيسية")
        st.dataframe(df_clean[main_columns].head())

    # توزيع أنواع البيانات
    import matplotlib.pyplot as plt
    dtype_counts = df_clean.dtypes.apply(lambda x: str(x)).value_counts()
    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700', '#FF6347', '#32CD32']
    fig, ax = plt.subplots()
    ax.pie(dtype_counts, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    if language == "English":
        st.subheader("📊 Column Data Types Distribution")
    else:
        st.subheader("📊 توزيع أنواع البيانات في الأعمدة")
    st.pyplot(fig)

    # القيم المفقودة (top 5 فقط)
    missing_data = df_clean.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False).head(5)
    if not missing_data.empty:
        if language == "English":
            st.subheader("❗ Top 5 Columns with Missing Values")
            st.write(missing_data)
        else:
            st.subheader("❗ أهم 5 أعمدة تحتوي على قيم مفقودة")
            st.write(missing_data)
    
    # عرض إحصائيات الأعمدة
    stats_df = calculate_statistics(df_clean)
    
    # تنسيق الإحصائيات
    styled_stats = (
        stats_df.style
        .format(na_rep="-")
        .set_table_styles([
            {'selector': 'th', 'props': [('font-size', '18px')]},
            {'selector': 'td', 'props': [('font-size', '16px')]},
        ])
    )

    # عرض جدول الإحصائيات
    st.subheader("📊 Column Statistics")
    st.dataframe(styled_stats, height=500)










#===========================
elif menu.startswith("🧹"):
    st.header("🧹 Data Cleansing and Processing" if language == "English" else "🧹 تنظيف البيانات")
    st.markdown("---")

    # وصف عام لعملية التنظيف
    if language == "English":
        st.subheader("🛠️ Data Cleansing Overview")
        st.markdown("""
        Data cleaning is a crucial part of the data science process. For this project, we removed unnecessary columns, handled missing data, and corrected outliers in various columns like `host_since`, `instant_bookable`, and `price`.
        """)
    else:
        st.subheader("🛠️ نظرة عامة على تنظيف البيانات")
        st.markdown("""
            يعد تنظيف البيانات جزءًا أساسيًا في عملية علم البيانات. في هذا المشروع، قمنا بإزالة الأعمدة غير الضرورية، وتعاملنا مع البيانات المفقودة، وقمنا بتصحيح القيم المتطرفة في العديد من الأعمدة مثل `host_since`، `instant_bookable`، و `price`.
        """)

    # عرض إحصائيات البيانات بعد التنظيف
    stats_df = calculate_statistics(df_clean)
    
    # تنسيق الجدول
    styled_stats = (
        stats_df.style
        .format(na_rep="-")
        .set_table_styles([
            {'selector': 'th', 'props': [('font-size', '18px')]},
            {'selector': 'td', 'props': [('font-size', '16px')]},
        ])
    )

    # عرض جدول الإحصائيات
    with st.expander("📊 Data Statistics After Cleaning"):
        st.dataframe(styled_stats, height=500)

    # اختياري: عرض الأعمدة التي تحتوي على قيم مفقودة أو انحراف عالي
    st.markdown("### Columns with high missing data or skewness")
    high_missing_data = stats_df[stats_df['Missing'] > 20]  # Example threshold for high missing data (20%)
    #high_skew = stats_df[stats_df['Skew'] > 3]  # Example threshold for high skewness
    high_skew = stats_df[stats_df['Skew'].apply(pd.to_numeric, errors='coerce') > 3] 
    
    if not high_missing_data.empty:
        st.markdown("**High Missing Data:**")
        st.dataframe(high_missing_data)
    
    if not high_skew.empty:
        st.markdown("**High Skewness Columns:**")
        st.dataframe(high_skew)
    # 🌟 Host Experience Summary
    if language == "English":
        with st.expander("🧮 Host Experience Column Cleaning & Engineering"):
            st.markdown("""
    ### 🛠️ Steps Performed:
    1. **Converted dates:** Transformed `host_since` to datetime format.
    2. **Calculated host experience in years:**  
   `host_since_experience = (today - host_since) / 365.25`
    3. **Analyzed distribution skewness:**  
        - Skewness = **0.018** (approximately normal)
    4. **Imputation strategy:**  
     - Since skewness < 1, used **mean imputation**.
     - Imputed missing values with **9.21 years**.
   
    ### 📊 Statistics After Cleaning:
    - Count: **279,712**
    - Mean: **9.21 years**
    - Median: **9.37 years**
    - Standard deviation: **2.44 years**
    - Min: **4.34 years**
    - Max: **16.88 years**
    - Skewness: **0.018**

    ### 🔍 Insights:
        ✅ Clean, reliable feature.  
        ✅ Near-normal distribution.  
        ✅ Imputation preserved central tendency.

        ✅ **Outcome:**
            Cleaned `host_since` and engineered `host_since_experience`.  
        Ready for modeling 🚀.
        """)
    else:
      with st.expander("🧮 تنظيف ومعالجة عمود سنوات خبرة المضيف"):
        st.markdown("""
    ### 🛠️ الخطوات التي قمنا بها:
    1. **تحويل التواريخ:** تم تحويل عمود `host_since` إلى صيغة التاريخ.
    2. **حساب سنوات الخبرة:**  
   `host_since_experience = (اليوم - تاريخ البداية) ÷ 365.25`
    3. **تحليل التوزيع:**  
     - درجة الانحراف = **0.018** ➡️ توزيع طبيعي تقريبًا.
    4. **طريقة التعويض:**  
     - بما أن الانحراف < 1، استخدمنا **المتوسط**.
     - القيم المفقودة عوضناها بـ **9.21 سنة**.

    ### 📊 الإحصائيات:
        - عدد الصفوف: **279,712**
        - المتوسط: **9.21 سنة**
    - الوسيط: **9.37 سنة**
    - الانحراف المعياري: **2.44 سنة**
    - أقل قيمة: **4.34 سنة**
    - أعلى قيمة: **16.88 سنة**
    - درجة الانحراف: **0.018**

    ✅ **النتيجة النهائية:**
    تم تنظيف العمود وتجهيزه للنمذجة 🚀.
    """)


    # 🌟 Instant Bookable Summary
        # 🌟 Superhost Summary
    if language == "English":
      with st.expander("⭐ Superhost Status Cleaning & Engineering"):
        st.markdown("""
    ### 🛠️ What we did:

    1️⃣ Converted "t"/"f" to 1/0.

    2️⃣ Applied business rules:
    - Not superhost if:
     - Experience <1 year
     - <1 listing
    - No instant booking
     - Ratings below thresholds

    3️⃣ Filled remaining missing with 0.

    📊 **Results:**
        - 125 marked as not superhost.
        - 40 missing filled with 0.

    ✅ **Outcome:**
    Cleaned and ready for modeling 🚀.
    """)
    else:
        with st.expander("⭐ تنظيف وتجهيز عمود حالة السوبرهوست"):
            st.markdown("""
        ### 🛠️ ما الذي قمنا به:

    1️⃣ تحويل القيم النصية "t"/"f" إلى 1/0.

    2️⃣ تطبيق قواعد المنصة:
        - ليس سوبرهوست إذا:
 - الخبرة أقل من سنة
  - أقل من إعلان واحد
  - بدون حجز فوري
  - تقييمات أقل من الحدود

    3️⃣ تعبئة المفقود بالقيمة 0.

    📊 **النتائج:**
    - 125 سجل غير سوبرهوست.
    - 40 سجل مفقود تم تعويضه.

    ✅ **النتيجة النهائية:**
    العمود جاهز للنمذجة 🚀.
    """)





# 🌟 Host Total Listings Summary
    
    if language == "English":
     with st.expander("🧮 Host Total Listings Count Cleaning"):
         st.markdown("""
    ### 🛠️ Steps:

    - Analyzed distribution.
    - Skewness: **23.49**.
    - Used **median imputation** (1.0) for 165 missing.

    📊 **Stats:**
    - Median: 1.0
    - Mean: 24.58
    - Skewness: 23.49

    ✅ **Outcome:**
    Cleaned and consistent 🚀.
    """)
    else:
       with st.expander("🧮 تنظيف عمود عدد الإعلانات"):
        st.markdown("""
    ## 🛠️ الخطوات:

    - تحليل التوزيع.
    - الانحراف: **23.49**.
    - تعويض المفقود بالوسيط (1.0).

    📊 **الإحصائيات:**
    - الوسيط: 1.0
    - المتوسط: 24.58
    - الانحراف: 23.49

    ✅ **النتيجة النهائية:**
    العمود جاهز للاستخدام 🚀.
    """)





    # 🌟 Neighbourhood Summary
    if language == "English":
        with st.expander("🏘️ Neighbourhood Column Cleaning"):
            st.markdown("""
    ### 🛠️ Steps:

    - Lowercased & trimmed.
    - Fuzzy matching (95%).
    - Created `neighbourhood_cleaned`.

    📊 Reduced uniques from 660.

    ✅ **Outcome:**
    Standardized neighbourhood names.
    """)
    else:
        with st.expander("🏘️ تنظيف عمود الأحياء"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - تحويل الأحرف الصغيرة وإزالة الفراغات.
    - مطابقة غامضة (95%).
    - إنشاء عمود موحد.

    📊 تقليل القيم الفريدة من 660.

    ✅ **النتيجة النهائية:**
    أسماء موحدة جاهزة للتحليل.
    """)





    # 🌟 City Summary
    if language == "English":
        with st.expander("🌍 City Column Cleaning"):
            st.markdown("""
    ### 🛠️ Steps:

    - Title Case normalization.
    - Fuzzy matching.
    - Manual corrections.

    📊 No missing values.

    ✅ **Outcome:**
    Clean and standardized cities.
    """)
    else:
        with st.expander("🌍 تنظيف عمود المدينة"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - تحويل لصيغة العنوان.
    - مطابقة غامضة.
    - تصحيحات يدوية.

    📊 لا توجد قيم مفقودة.

    ✅ **النتيجة النهائية:**
    المدن موحدة وجاهزة.
    """)
            







    # 🌟 Geospatial Summary
    if language == "English":
        with st.expander("🗺️ Geospatial Data Validation"):
            st.markdown("""
    ### 🛠️ Steps:

    - Validated ranges.
    - Auto-fix invalids.
    - No missing values.

    ✅ **Outcome:**
    Ready for mapping & clustering.
    """)
    else:
        with st.expander("🗺️ التحقق من الإحداثيات الجغرافية"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - التحقق من النطاقات.
    - تصحيح القيم غير الصحيحة.
    - لا قيم مفقودة.

    ✅ **النتيجة النهائية:**
    جاهزة للخرائط والتحليل.
    """)

    # 🌟 District Summary
    if language == "English":
        with st.expander("🏙️ District Column Enrichment"):
            st.markdown("""
    ### 🛠️ Steps:

    - Reverse geocoding (~6,274 calls).
    - Filled ~84.5% missing.

    ✅ **Outcome:**
    Complete district info.
    """)
    else:
        with st.expander("🏙️ إثراء بيانات الحي"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - تحويل عكسي للإحداثيات.
    - تعبئة ~84.5% من القيم المفقودة.

    ✅ **النتيجة النهائية:**
    بيانات الحي مكتملة.
    """)










    # 🌟 Property Type Summary
    if language == "English":
        with st.expander("🏠 Property Type Cleaning & Categorization"):
            st.markdown("""
    ### 🛠️ Steps:

    - Cleaned text.
    - Fuzzy matching.
    - Created `property_category`.

    ✅ **Outcome:**
    8 categories ready.
    """)
    else:
        with st.expander("🏠 تنظيف نوع العقار"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - تنظيف النص.
    - مطابقة غامضة.
    - إنشاء `property_category`.

    ✅ **النتيجة النهائية:**
    تم تصنيف العقارات.
    """)

    # 🌟 Room Type Summary
    if language == "English":
        with st.expander("🛏️ Room Type Cleaning"):
            st.markdown("""
    ### 🛠️ Steps:

    - Normalized values.
    - Fuzzy matching.
    - Standardized 4 categories.

    ✅ **Outcome:**
    Consistent room types.
    """)
    else:
        with st.expander("🛏️ تنظيف نوع الغرفة"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - توحيد القيم.
    - مطابقة غامضة.
    - 4 تصنيفات موحدة.

    ✅ **النتيجة النهائية:**
    أنواع غرف موحدة.
    """)

    # 🌟 Accommodates Summary
    if language == "English":
        with st.expander("🛏️ Accommodates Column Cleaning"):
            st.markdown("""
    ### 🛠️ Steps:

    - Removed invalid rows.
    - Hierarchical imputation.

    ✅ **Outcome:**
    Clean accommodates values.
    """)
    else:
        with st.expander("🛏️ تنظيف عدد الضيوف"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - حذف القيم غير الصحيحة.
    - تعويض هرمي ذكي.

    ✅ **النتيجة النهائية:**
    عدد الضيوف جاهز.
    """)










    # 🌟 Bedrooms Summary
    if language == "English":
        with st.expander("🛏️ Bedrooms ML Imputation"):
            st.markdown("""
    ### 🛠️ Steps:

    - Tested 7 methods.
    - Gradient Boosting selected.

    ✅ **Outcome:**
    Clean bedrooms column.
    """)
    else:
        with st.expander("🛏️ تعويض غرف النوم"):
            st.markdown("""
    ### 🛠️ الخطوات:

    - تجربة 7 طرق.
    - اختيار Gradient Boosting.

    ✅ **النتيجة النهائية:**
    العمود جاهز تمامًا.
    """)
    if language == "English":
        with st.expander("💰 Price Column Cleaning & Conversion Pipeline"):
            st.markdown("""
    ### 🛠️ Steps Performed:

    1️⃣ Created a `PriceConversion` class to handle conversion of the `price` column to USD.

    2️⃣ Mapped each city to its local currency using `currency_map`.

    3️⃣ Implemented `validate_cities()` to ensure all cities are mapped correctly.

    4️⃣ Fetched live exchange rates from CurrencyFreaks API and saved them in `exchange_rates.json`.

    5️⃣ Converted all price values to USD and stored them in `price_usd`.

    6️⃣ Printed exchange rates for verification.

    7️⃣ Identified **28 listings** with `price_usd == 0`.

    8️⃣ Analyzed zero-price records and concluded they were likely test or erroneous listings.

    9️⃣ Removed all zero-price rows for data integrity.

    ---

    ### 📊 Exchange Rates Used:

    | Currency | USD Rate |
    |----------|----------|
    | EUR      | 1.172    |
    | USD      | 1.0      |
    | THB      | 0.0307   |
    | BRL      | 0.1823   |
    | AUD      | 0.653    |
    | TRY      | 0.0251   |
    | HKD      | 0.1274   |
    | MXN      | 0.0531   |
    | ZAR      | 0.0561   |

    ---

    ### 🔍 Insights:

    ✅ Converting prices to USD enables consistent cross-city analysis.

    ✅ Removing invalid zero-price rows prevents skewed distributions and model bias.

    ✅ The price column is now clean, numeric, and reliable.

    ---

    ✅ **Outcome:**

    - Price column fully cleaned and standardized.
    - Zero-price records removed.
    - Ready for modeling and analysis 🚀.
    """)
    else:
        with st.expander("💰 تنظيف وتحويل عمود السعر"):
            st.markdown("""
    ### 🛠️ الخطوات التي تم تنفيذها:

    1️⃣ إنشاء كلاس `PriceConversion` لمعالجة تحويل الأسعار إلى الدولار الأمريكي.

    2️⃣ ربط كل مدينة بعملتها المحلية باستخدام `currency_map`.

    3️⃣ تنفيذ `validate_cities()` للتأكد من تغطية جميع المدن.

    4️⃣ جلب أسعار الصرف الحية من خدمة CurrencyFreaks وتخزينها في `exchange_rates.json`.

    5️⃣ تحويل جميع القيم إلى عمود `price_usd`.

    6️⃣ طباعة أسعار الصرف للتأكد منها.

    7️⃣ تحديد **28 إعلانًا** بأسعار صفرية `price_usd == 0`.

    8️⃣ تحليل السجلات وتبين أنها اختبارات أو أخطاء إدخال.

    9️⃣ حذف جميع الصفوف ذات السعر الصفري لضمان جودة البيانات.

    ---

    ### 📊 أسعار الصرف المستخدمة:

    | العملة | السعر مقابل الدولار |
    |--------|---------------------|
    | EUR    | 1.172               |
    | USD    | 1.0                 |
    | THB    | 0.0307              |
    | BRL    | 0.1823              |
    | AUD    | 0.653               |
    | TRY    | 0.0251              |
    | HKD    | 0.1274              |
    | MXN    | 0.0531              |
    | ZAR    | 0.0561              |

    ---

    ### 🔍 الملاحظات:

    ✅ تحويل الأسعار إلى عملة موحدة (الدولار) يضمن مقارنة عادلة بين المدن.

    ✅ إزالة السجلات الصفرية يمنع انحراف التوزيع وتأثيره على النماذج التحليلية.

    ✅ أصبح عمود السعر نظيفًا بالكامل وجاهزًا للاستخدام.

    ---

    ✅ **النتيجة النهائية:**

    - تم تنظيف عمود السعر وتوحيده بالدولار الأمريكي.
    - حذف جميع الصفوف ذات الأسعار الصفرية.
    - البيانات جاهزة للنمذجة والتحليل 🚀.
    """)
            





    # 🌟 Nights Cleaning Summary
    if language == "English":
        with st.expander("🛌 Nights Cleaning Pipeline – minimum_nights & maximum_nights"):
            st.markdown("""
    ### 🛠️ Steps Performed:

    1️⃣ Built a reusable `NightCleaning` class to handle extreme outliers in `minimum_nights` and `maximum_nights`.

    2️⃣ Analyzed the **95th percentile thresholds**:
    - `minimum_nights` → **30 nights**
    - `maximum_nights` → **1125 nights**

    3️⃣ Conducted manual review of values above these thresholds.

    4️⃣ Applied final caps based on business logic:
    - `minimum_nights` capped at **1250 nights (~3.5 years)**
    - `maximum_nights` capped at **3650 nights (10 years)**

    5️⃣ Removed outlier rows exceeding these caps.

    ---

    ### 📊 Analysis:

    - Detected severe outliers:
    - Some `maximum_nights` up to **2 billion** → data errors.
    - A few `minimum_nights` ≥ **9999** → implausible.
    - Manual review found:
    - Legitimate long-term stays (1–3 years) → retained.
    - Extreme placeholders → removed.

    ---

    ### 🔍 Insights:

    ✨ The distribution is naturally **right-skewed**:
    - Most listings are for **1–30 nights**.
    - Some serve **mid-term and long-term rentals**.

    🎯 Applying thresholds ensures:
    - Accurate modeling and analytics.
    - Clean separation of short vs long-term rentals.
    - Removal of extreme distortions.

    ---

    ✅ **Outcome:**

    - Cleaned `minimum_nights` and `maximum_nights` columns.
    - Applied thresholds:
    - `min_nights` ≤ 1250
    - `max_nights` ≤ 3650
    - Rows removed:
    - `minimum_nights`: **1**
    - `maximum_nights`: **52**

    ✅ Dataset now contains realistic and interpretable stay durations, ready for modeling 🚀.
    """)
    else:
        with st.expander("🛌 تنظيف عمودي مدة الإقامة (minimum_nights و maximum_nights)"):
            st.markdown("""
    ### 🛠️ الخطوات التي تم تنفيذها:

    1️⃣ إنشاء كلاس `NightCleaning` قابل لإعادة الاستخدام لمعالجة القيم الشاذة الكبيرة في الأعمدة.

    2️⃣ تحليل **النسبة المئوية 95**:
    - `minimum_nights`: **30 ليلة**
    - `maximum_nights`: **1125 ليلة**

    3️⃣ مراجعة القيم يدويًا فوق هذه الحدود.

    4️⃣ تحديد الحدود النهائية بناءً على منطق العمل:
    - حد `minimum_nights`: **1250 ليلة (~3.5 سنوات)**
    - حد `maximum_nights`: **3650 ليلة (10 سنوات)**

    5️⃣ حذف الصفوف التي تجاوزت هذه الحدود.

    ---

    ### 📊 التحليل:

    - تم اكتشاف قيم شاذة جدًا:
    - `maximum_nights` يصل إلى **2 مليار** → أخطاء واضحة.
    - بعض `minimum_nights` ≥ **9999** → غير منطقية.
    - المراجعة اليدوية أظهرت:
    - وجود إقامات طويلة حقيقية (1–3 سنوات) → تم الإبقاء عليها.
    - القيم المتطرفة والاختبارية → تم حذفها.

    ---

    ### 🔍 الملاحظات:

    ✨ التوزيع طبيعي أن يكون **منحرفًا لليمين**:
    - معظم الإقامات من **1–30 ليلة**.
    - البعض يغطي الإقامات المتوسطة والطويلة.

    🎯 تطبيق الحدود يحقق:
    - دقة أعلى في النمذجة والتحليل.
    - وضوح فصل بين الإيجارات قصيرة وطويلة المدى.
    - إزالة التشتت الناتج عن القيم القصوى.

    ---

    ✅ **النتيجة النهائية:**

    - تم تنظيف عمودي `minimum_nights` و `maximum_nights`.
    - الحدود المطبقة:
    - `minimum_nights` ≤ 1250
    - `maximum_nights` ≤ 3650
    - الصفوف المحذوفة:
    - `minimum_nights`: **1**
    - `maximum_nights`: **52**

    ✅ أصبحت البيانات الآن نظيفة وواقعية وجاهزة للتحليل والنمذجة 🚀.
    """)
            





    # 🌟 Instant Bookable Summary
    if language == "English":
        with st.expander("⚡ Instant Bookable Column Cleaning Pipeline"):
            st.markdown("""
    ### 🛠️ Steps Performed:

    1️⃣ Designed an `InstantBookable` cleaning class to standardize values.

    2️⃣ Verified no missing values in `instant_bookable`.

    3️⃣ Applied consistent mapping:
    - ✅ **'t', 'true', 'yes', '1' → 1**
    - ❌ **'f', 'false', 'no', '0' → 0**
    - ⚠️ Any unknown or NaN → default to **0**

    4️⃣ Created a new binary column: `instant_bookable_cleaned`.

    ---

    ### 📊 Analysis:

    - **Missing values:** `0`
    - **Initial Distribution:**
        | Value | Count    | Proportion |
        |-------|----------|------------|
        | 'f'   | 163,957  | 58.7%      |
        | 't'   | 115,589  | 41.3%      |

    - No significant anomalies detected.

    ---

    ### 🔍 Insights:

    - `instant_bookable` is a strong categorical indicator of booking ease.
    - No advanced imputation required.
    - Pipeline ensures:
        - 🧼 Automatic handling of NaN or unexpected strings.
        - 🛡️ Robustness for future data.

    ---

    ✅ **Outcome:**

    - `instant_bookable` cleaned and converted to binary format (`instant_bookable_cleaned`).
    - No rows removed.
    - Ready for EDA and predictive modeling 🚀.
    """)
    else:
        with st.expander("⚡ تنظيف عمود الحجز الفوري (instant_bookable)"):
            st.markdown("""
    ### 🛠️ الخطوات التي تم تنفيذها:

    1️⃣ إنشاء كلاس `InstantBookable` لتوحيد القيم في العمود.

    2️⃣ التحقق من عدم وجود قيم مفقودة في `instant_bookable`.

    3️⃣ تطبيق خريطة التحويل القياسية:
    - ✅ **'t', 'true', 'yes', '1' → 1**
    - ❌ **'f', 'false', 'no', '0' → 0**
    - ⚠️ أي قيمة غير معروفة أو فارغة → تتحول تلقائيًا إلى **0**

    4️⃣ إنشاء عمود جديد ثنائي: `instant_bookable_cleaned`.

    ---

    ### 📊 التحليل:

    - **عدد القيم المفقودة:** `0`
    - **توزيع القيم قبل التنظيف:**
        | القيمة | العدد    | النسبة    |
        |--------|----------|-----------|
        | 'f'    | 163,957  | 58.7%     |
        | 't'    | 115,589  | 41.3%     |

    - لم يتم رصد أي شذوذات هامة.

    ---

    ### 🔍 الملاحظات:

    - عمود `instant_bookable` يمثل مؤشرًا قويًا لسلوك الحجز.
    - لم يكن هناك حاجة لتعويض معقد.
    - المسار يضمن:
        - 🧼 معالجة تلقائية لأي قيم غير معروفة مستقبلًا.
        - 🛡️ مرونة وموثوقية البيانات.

    ---

    ✅ **النتيجة النهائية:**

    - تم تنظيف وتحويل عمود `instant_bookable` إلى صيغة ثنائية (`instant_bookable_cleaned`).
    - لم يتم حذف أي صفوف.
    - جاهز للتحليل والنمذجة التنبؤية 🚀.
    """)
    # ===============================
    # 🌟 ملخص التنظيف
    # ===============================
    if language == "English":
        st.success("✅ Data cleansing completed successfully! The dataset is now ready for analysis and modeling.")
    else:
        st.success("✅ تم تنظيف البيانات بنجاح! البيانات الآن جاهزة للتحليل والنمذجة.")









    # ===============================
    # 🌐 رابط Airbnb
    # ===============================
    st.markdown(
        "[🌐 Visit Airbnb](https://www.airbnb.com)" if language == "English"
        else "[🌐 زيارة موقع Airbnb](https://www.airbnb.com)"
    )


        


    if language == "English":
        st.header("✅ Final Data Cleansing & Feature Engineering Summary")

        st.markdown("""
    🎯 **Key Outcomes:**

    - ✅ All columns cleaned, standardized, and enriched.
    - ✅ Prices converted to **USD** using live exchange rates.
    - ✅ Missing values handled with **smart hierarchical and ML-based imputation pipelines**.
    - ✅ Outliers in stay duration removed (caps applied to minimum_nights and maximum_nights).
    - ✅ Categorical variables encoded into consistent formats.
    - ✅ Geospatial coordinates validated and auto-fixed.
    - ✅ Instant booking status cleaned and converted to binary.
    - ✅ New engineered features created (e.g., `host_since_experience`, `price_usd`, `instant_bookable_cleaned`).

    ✅ **Your dataset is now fully prepared for:**
    - Predictive machine learning modeling.
    - Advanced segmentation and clustering.
    - Interactive visualization dashboards.
    - Accurate business insights and reporting.

    **👏 Excellent work! Your data is clean, consistent, and ready to drive value 🚀.**
    """)

        st.info("💡 Tip: You can now proceed to feature selection, modeling, or dashboard creation.")
    else:
        st.header("✅ ملخص نهائي لمرحلة تنظيف وتجهيز البيانات")

        st.markdown("""
    🎯 **النتائج النهائية:**

    - ✅ تنظيف وتوحيد جميع الأعمدة وإثراؤها.
    - ✅ تحويل الأسعار إلى **الدولار الأمريكي** باستخدام أسعار صرف حقيقية.
    - ✅ معالجة القيم المفقودة باستراتيجيات ذكية (تعويض هرمي ونماذج تعلم آلي).
    - ✅ إزالة القيم الشاذة في مدد الإقامة (تطبيق حدود min/max).
    - ✅ ترميز المتغيرات الفئوية بشكل موحد.
    - ✅ التحقق من الإحداثيات الجغرافية وتصحيحها تلقائيًا.
    - ✅ تنظيف عمود الحجز الفوري وتحويله إلى صيغة ثنائية.
    - ✅ إنشاء ميزات مشتقة جديدة مثل:
    - `host_since_experience`
    - `price_usd`
    - `instant_bookable_cleaned`

    ✅ **البيانات أصبحت جاهزة تمامًا لـ:**
    - نمذجة التنبؤ والتحليل المتقدم.
    - تقسيم واستهداف الشرائح المختلفة.
    - إعداد لوحات معلومات تفاعلية.
    - استخراج رؤى دقيقة وموثوقة.

    **👏 عمل رائع! البيانات الآن نظيفة وموثوقة وقابلة للاستخدام الفوري 🚀.**
    """)

        st.info("💡 نصيحة: يمكنك الآن الانتقال لمرحلة اختيار الميزات أو بناء النماذج أو إنشاء لوحات البيانات.")










elif menu.startswith("📌"):
    if language == "English":
        st.header("💡 Practical Motivation")
        st.markdown("---")

        st.markdown("""
In recent years, **Airbnb has transformed the hospitality industry**, offering both convenience and cost-effective options to travelers, and income opportunities for hosts.  
As short-term rentals grow in popularity, it becomes essential to understand the **factors that drive pricing and success on Airbnb**.

With thousands of listings available, users face information overload. This raises essential questions:

#### 🤔 Hosts may ask:
- What type of property should I invest in to maximize rental income?
- Which features most influence guest interest and price?

#### 🤔 Travelers may wonder:
- How can I find the best value listings with the amenities I care about?

🎯 **This project aims to:**
- Identify the top factors influencing listing prices.
- Highlight features common to premium listings.
- Empower hosts with pricing insights to maximize profit.
- Help travelers find budget-friendly, feature-rich stays.

---

### 🔎 Key Business Questions

1. **What is the relationship between the number of guests accommodated and the nightly price?**
2. **What is the relationship between Instant Bookable status and nightly price?**
3. **Which property categories generate the highest revenue potential?**
4. **Which districts or cities generate the highest revenue and which have the most growth potential?**
5. **What is the seasonal pattern of new hosts joining and new listings being created over the year?**
6. **How are prices distributed by room type (Entire Place – Private Room – Shared Room – Hotel Room)?**
7. **How does the combination of Room Type and Property Category influence average nightly price?**
8. **How are property categories distributed across different cities?**
9. **What is the relationship between the minimum stay requirement and the average nightly price?**

---

🔎 **Why this matters:**  
Understanding Airbnb pricing isn't just a host's concern. It's a powerful case study for applying **data science to real-world business problems** — blending economics, human behavior, and analytics.
        """)

    else:
        st.header("💡 الدافع العملي")
        st.markdown("---")

        st.markdown("""
في السنوات الأخيرة، أحدثت **Airbnb تحولًا كبيرًا في عالم الإقامة والسفر**، إذ وفرت للضيوف حلولًا مرنة واقتصادية، وأتاحت للمضيفين فرصًا رائعة لتحقيق دخل من خلال تأجير ممتلكاتهم.  
ومع تزايد شعبية الإيجارات قصيرة الأجل، أصبح من الضروري فهم **العوامل المؤثرة في تسعير الإعلانات ونجاحها**.

ومع وجود آلاف الإعلانات، يواجه المستخدمون صعوبة في الاختيار، مما يثير تساؤلات هامة:

#### 🤔 المضيفون قد يتساءلون:
- ما نوع العقار الأنسب للاستثمار لتحقيق أكبر عائد؟
- ما هي الميزات التي تؤثر أكثر على السعر وجذب الضيوف؟

#### 🤔 المسافرون قد يتساءلون:
- كيف أجد إعلانًا مميزًا بسعر مناسب وبالمزايا التي أفضلها؟

🎯 **يهدف هذا المشروع إلى:**
- تحديد العوامل الأهم التي تؤثر على تسعير الإعلانات.
- إبراز الميزات المشتركة بين العقارات الفاخرة.
- تزويد المضيفين برؤى تسعيرية لزيادة الأرباح.
- مساعدة المسافرين على العثور على خيارات مريحة واقتصادية.

---

### 🔎 أسئلة الأعمال الرئيسية

1. **ما هو العلاقة بين عدد الضيوف والـ nightly price؟**
2. **ما هو العلاقة بين حالة الحجز الفوري (Instant Bookable) والسعر الليلي؟**
3. **ما هي الفئات العقارية التي تولد أعلى إمكانيات للإيرادات؟**
4. **ما هي الأحياء أو المدن التي تولد أعلى الإيرادات وأيها يحتوي على أكبر إمكانيات للنمو؟**
5. **ما هو النمط الموسمي لدخول المضيفين الجدد وإنشاء الإعلانات الجديدة خلال العام؟**
6. **كيف يتم توزيع الأسعار حسب نوع الغرفة (مكان كامل – غرفة خاصة – غرفة مشتركة – غرفة فندق)؟**
7. **كيف تؤثر مزيج نوع الغرفة وفئة العقار على متوسط السعر الليلي؟**
8. **كيف يتم توزيع الفئات العقارية عبر المدن المختلفة؟**
9. **ما هو العلاقة بين الحد الأدنى للإقامة ومتوسط السعر الليلي؟**

---

🔎 **لماذا هذا مهم؟**  
تحليل تسعير Airbnb لا يخدم المضيف فقط، بل هو أيضًا **نموذج تطبيقي حقيقي لعلم البيانات** في حل مشاكل تجارية — يجمع بين السلوك البشري والتحليل الاقتصادي.
        """)








#===========================
elif menu.startswith("📈"):
    st.header(" ")
    tabs = st.tabs(["📊 Completed Business Visualizations", "📌 Key Business Questions To Be Answered","🎯 Final Strategy & Recommendations"])




    with tabs[0]:
        st.subheader("📊 Completed Business Visualizations")
        with st.expander("👥 Hosts Overview Visualization", expanded=True):

            def plot_hosts_overview(df):
                n_hosts = df['host_id'].nunique()
                n_listings = df['listing_id'].nunique()
                listings_per_host = df.groupby('host_id')['listing_id'].count()

                def classify_host(n):
                    if n == 1:
                        return 'Individual'
                    elif n <= 4:
                        return 'Small Professional'
                    else:
                        return 'Company'
                
                host_type_counts = listings_per_host.apply(classify_host).value_counts()

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Pie chart for Unique Hosts vs Listings
                axes[0].pie([n_hosts, n_listings], labels=['Unique Hosts', 'Listings'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("muted"))
                axes[0].set_title('Unique Hosts vs Listings' if language == "English" else "المضيفين الفريدين مقابل الإعلانات")

                # Pie chart for Host Types
                axes[1].pie(host_type_counts.values, labels=host_type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
                axes[1].set_title('Host Type Distribution' if language == "English" else "توزيع أنواع المضيفين")

                st.pyplot(fig)

                # Insights and Analysis
                if language == "English":
                    st.markdown("""
        ### 🧠 Business Insights:
        - 👨‍💼 **The majority of Airbnb hosts are Individuals (~82.5%)**, managing only one listing → the platform remains largely individual-driven.
        - 🏠 **Small Professionals (~14.2%)** are growing and may represent entrepreneurial operators.
        - 🏢 **Companies (~3.3%)** — though a minority — may manage a disproportionate share of total listings, likely influencing pricing & competition in key markets.
        - 📈 The distribution is highly Right-Skewed → suggesting a classic long-tail market:
            - Many casual hosts.
            - Few "Power Hosts" managing large portfolios.
        """)
                else:
                    st.markdown("""
        ### 🧠 الرؤى التجارية:
        - 👨‍💼 **الغالبية العظمى من المضيفين على Airbnb هم أفراد (~82.5%)** يديرون إعلانًا واحدًا فقط → تظل المنصة مدفوعة في الغالب من قبل الأفراد.
        - 🏠 **المحترفون الصغار (~14.2%)** في تزايد وقد يمثلون مشغلين رياديين.
        - 🏢 **الشركات (~3.3%)** — رغم أنهم أقلية — قد يديرون حصة غير متناسبة من إجمالي الإعلانات، مما قد يؤثر على التسعير والمنافسة في الأسواق الرئيسية.
        - 📈 التوزيع يميل إلى الانحراف الشديد → مما يشير إلى سوق طويل الذيل الكلاسيكي:
            - العديد من المضيفين العرضيين.
            - قلة من "المضيفين الأقوياء" الذين يديرون محفظات كبيرة.
        """)

            # Ensure datetime conversion before calling function
            df_clean['host_since'] = pd.to_datetime(df_clean['host_since'], errors='coerce')

            # Call the function
            plot_hosts_overview(df_clean)

    # تحديد اللغة
        #language = st.radio("Select Language", ("English", "Arabic"))







    # 📈 Host Registration Trend Over Time Visualization
        with st.expander("📅 Host Registration Trend Over Time"):
        # أضف الرسم البياني الخاص بتوزيع التسجيلات عبر الوقت هنا
            st.write("Line Chart for Host Registration Trend Over Time")
            def plot_host_registration_trend(df):
                # استخراج السنة من عمود 'host_since'
                df['host_since_year'] = df['host_since'].dt.year
                
                # حساب عدد المضيفين المسجلين في كل سنة
                host_registrations = df['host_since_year'].value_counts().sort_index()

                # رسم البيانات
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=host_registrations.index, y=host_registrations.values, marker='o', color='coral')
                plt.title('Host Registration Trend Over Time' if language == "English" else 'اتجاه تسجيل المضيفين عبر الزمن')
                plt.xlabel('Year of Registration' if language == "English" else 'سنة التسجيل')
                plt.ylabel('Number of Hosts Registered' if language == "English" else 'عدد المضيفين المسجلين')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(host_registrations.index, rotation=45)
                st.pyplot(plt)

            # فئات المضيفين حسب السنوات
            def host_segmentation(df_clean):
                total_hosts = df_clean.shape[0]
                
                early_hosts = df_clean[df_clean['host_since_year'] <= 2012].shape[0]
                growth_hosts = df_clean[(df_clean['host_since_year'] >= 2013) & (df_clean['host_since_year'] <= 2015)].shape[0]
                mature_hosts = df_clean[(df_clean['host_since_year'] >= 2016) & (df_clean['host_since_year'] <= 2019)].shape[0]
                covid_hosts = df_clean[df_clean['host_since_year'] >= 2020].shape[0]
                
                segmentation_data = {
                    'Era': ['Early adopters (≤2012)', 'Growth phase (2013-2015)', 'Mature phase (2016-2019)', 'COVID-era (2020+)'],
                    'Number of Hosts': [early_hosts, growth_hosts, mature_hosts, covid_hosts],
                    'Percentage of Total (%)': [
                        early_hosts / total_hosts * 100,
                        growth_hosts / total_hosts * 100,
                        mature_hosts / total_hosts * 100,
                        covid_hosts / total_hosts * 100
                    ]
                }

                segmentation_df = pd.DataFrame(segmentation_data)

                st.markdown("### 📊 Host Segmentation by Era" if language == "English" else "### 📊 تقسيم المضيفين حسب الفترات الزمنية")
                st.dataframe(segmentation_df.round(1))
                
                # Insights
                if language == "English":
                    st.markdown("""
                    🕰️ **Early adopters (≤2012):** ~9.2% → pioneers of the platform  
                    🚀 **Growth phase (2013-2015):** ~38.6% → majority joined during the platform's rapid expansion phase  
                    🏢 **Mature phase (2016-2019):** ~45.6% → stabilized adoption period, reflecting platform maturity  
                    🦠 **COVID-era (2020+):** ~6.6% → significant decline in new registrations due to pandemic disruptions
                    """)
                else:
                    st.markdown("""
                    🕰️ **المبادرون الأوائل (≤2012):** ~9.2% → رواد المنصة  
                    🚀 **مرحلة النمو (2013-2015):** ~38.6% → الغالبية انضموا خلال فترة التوسع السريع للمنصة  
                    🏢 **المرحلة الناضجة (2016-2019):** ~45.6% → فترة الاستقرار في التبني، مما يعكس نضج المنصة  
                    🦠 **فترة COVID (2020+):** ~6.6% → انخفاض كبير في التسجيلات الجديدة بسبب اضطرابات الجائحة
                    """)

            # عرض الرسومات والفئات في Streamlit
            st.header("📈 Host Registration Trend Over Time" if language == "English" else "📈 اتجاه تسجيل المضيفين عبر الزمن")
            plot_host_registration_trend(df_clean)
            host_segmentation(df_clean)
            def plot_superhost_status(df, language):
                df_clean['host_is_superhost'] = df_clean['host_is_superhost'].map({'t': 1, 'f': 0, True: 1, False: 0})
                superhost_counts = df_clean['host_is_superhost'].value_counts()
                superhost_labels = {1: 'Superhost', 0: 'Non-Superhost'}

                # Create the pie chart
                plt.figure(figsize=(2, 2))
                colors = ['#4CAF50', '#FF7043']  # Green for Superhost, Orange for Non-Superhost
                plt.pie(superhost_counts, labels=[superhost_labels[k] for k in superhost_counts.index],
                        autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0), textprops={'fontsize': 12})
                
                if language == "English":
                    plt.title('')  # Remove title from the pie chart
                else:
                    plt.title('')  # Remove title from the pie chart in Arabic
                
                st.pyplot(plt)

    # Insights





        with st.expander("📊 Superhost Status Distribution"):
        # أضف الرسم البياني الخاص بـ Superhost Status هنا
            st.write("Pie Chart for Superhost Status Distribution")
            def superhost_insights(df, language):
                total_hosts = df_clean.shape[0]
                non_superhosts = df_clean[df_clean['host_is_superhost'] == 0].shape[0]
                superhosts = df_clean[df_clean['host_is_superhost'] == 1].shape[0]

                non_superhost_percentage = (non_superhosts / total_hosts) * 100
                superhost_percentage = (superhosts / total_hosts) * 100

                if language == "English":
                    insights = f"""
                    🏅 **Superhost Status Insights:**

                    - **~{non_superhost_percentage:.1f}%** of hosts are Non-Superhosts.
                    - **~{superhost_percentage:.1f}%** of hosts have achieved Superhost status.

                    🎯 **Business Implication:**
                    - A large portion of hosts (around 82%) are **Non-Superhosts**, which suggests there's room for growth in providing incentives or assistance to help more hosts attain Superhost status.
                    - Only **18%** of hosts have achieved **Superhost** status, which could indicate that becoming a Superhost is a significant achievement and may require sustained performance on the platform.
                    """
                else:
                    insights = f"""
                    🏅 **رؤى حالة السوبرهوست:**

                    - **~{non_superhost_percentage:.1f}%** من المضيفين هم **Non-Superhosts**.
                    - **~{superhost_percentage:.1f}%** من المضيفين حققوا حالة **Superhost**.

                    🎯 **التأثير التجاري:**
                    - هناك نسبة كبيرة من المضيفين (حوالي 82%) هم **Non-Superhosts**، مما يشير إلى وجود فرصة للنمو من خلال توفير الحوافز أو الدعم لمساعدة المزيد من المضيفين في تحقيق حالة **Superhost**.
                    - فقط **18%** من المضيفين حققوا حالة **Superhost**، مما قد يعني أن تحقيق حالة **Superhost** هو إنجاز كبير ويتطلب أداءً مستمرًا على المنصة.
                    """

                st.markdown(insights)

            # Show visualizations and insights in Streamlit
            st.header("🏅 Superhost Status Distribution")
            #language = st.selectbox("Choose language", ["English", "Arabic"])  # You can replace this with the actual language selection in your app
            plot_superhost_status(df_clean, language)
            superhost_insights(df_clean, language)
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import FastMarkerCluster
        from scipy.stats import skew, kurtosis


        # Assume df_clean contains latitude and longitude columns





        # Latitude & Longitude
        
        with st.expander("🗺️ Latitude & Longitude → Listings Geographic Distribution"):
        # أضف الخريطة أو الرسم البياني هنا
            st.write("Map or Chart for Listings Geographic Distribution")
            if language == "English":
                st.header("🗺️ Latitude & Longitude → Listings Geographic Distribution")
            else:
                st.header("🗺️ خطوط العرض والطول → توزيع الإعلانات جغرافياً")

            # بيانات الإحداثيات
            lat = df_clean['latitude']
            lon = df_clean['longitude']
            locations = list(zip(lat, lon))

            # مركز الخريطة (اللاتيتود والخطوط الطولية المتوسطة)
            center_lat = df_clean['latitude'].mean()
            center_lon = df_clean['longitude'].mean()

            # إنشاء الخريطة الأساسية مركزة حول المتوسط
            map_airbnb = folium.Map(location=[center_lat, center_lon], tiles='CartoDB Positron', zoom_start=2)

            # إضافة FastMarkerCluster لتحسين الأداء مع مجموعات البيانات الكبيرة
            FastMarkerCluster(data=locations).add_to(map_airbnb)

            # عرض الخريطة في Streamlit مع الحجم أكبر
            st_folium(map_airbnb, width=1000, height=700)








        with st.expander("📊 Price Distribution Analysis", expanded=True):

            st.subheader("💵 Summary Statistics")

            # الحسابات
            mean_price = df_clean['price_usd'].mean()
            median_price = df_clean['price_usd'].median()
            mode_price = df_clean['price_usd'].mode()[0]
            min_price = df_clean['price_usd'].min()
            max_price = df_clean['price_usd'].max()
            skew_price = skew(df_clean['price_usd'])
            kurt_price = kurtosis(df_clean['price_usd'])

            # جدول إحصائي
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Median", "Mode", "Min", "Max", "Skewness", "Kurtosis"],
                "Value": [f"${mean_price:.2f}", f"${median_price}", f"${mode_price:.2f}",
                        f"${min_price:.2f}", f"${max_price:,.2f}",
                        f"{skew_price:.2f}", f"{kurt_price:.2f}"]
            })

            st.table(stats_df)

            # فلترة الأسعار
            filtered = df_clean[df_clean["price_usd"] < 1000]

            # Density Plot
            st.subheader("🌊 Price Density Plot (Under $1000)")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.kdeplot(filtered["price_usd"], shade=True, color='purple', clip=(1, None), ax=ax1)
            ax1.set_title("Distribution of Prices (Under $1000)", fontsize=14)
            ax1.set_xlabel("Price (USD)")
            ax1.set_ylabel("Density")
            ax1.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig1)

            # Box Plot
            st.subheader("📦 Box Plot of Prices (Under $1000)")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            sns.boxplot(data=filtered, x="price_usd", color="purple", ax=ax2)
            ax2.set_title("Box Plot of Prices (Under $1000)", fontsize=14)
            ax2.set_xlabel("Price (USD)")
            ax2.grid(axis='x', linestyle='--', alpha=0.3)
            st.pyplot(fig2)

            # Insights
            st.markdown("### 💡 Business Insights")
            st.markdown("""
            - ✅ **Affordability Dominance**: Most listings (80%) are priced below $250, making Airbnb highly attractive for budget-conscious travelers.
            - 👑 **Premium Market Opportunity**: Although rare (~5–10%), listings above $1000 represent a lucrative segment for luxury or special experiences.
            - 📈 **Skewness & Kurtosis**: High skew (**{:.2f}**) and kurtosis (**{:.2f}**) indicate heavy clustering around low prices and a long tail of expensive listings.
            - ⚖️ **Marketing Recommendation**:
                - 🔥 Focus on affordable listings (~80%) to drive traffic and reliability.
                - 💎 Target premium listings (~5–10%) for high-margin growth and brand prestige.
            """.format(skew_price, kurt_price))






        with st.expander("🗺️ Neighbourhood Distribution", expanded=True):
            st.subheader("🏙️ Top 20 Neighbourhoods by Number of Listings")

            top_neighbourhoods = df_clean['neighbourhood'].value_counts().head(20)

            fig1, ax1 = plt.subplots(figsize=(12, 6))
            sns.barplot(x=top_neighbourhoods.values, y=top_neighbourhoods.index, color='skyblue', ax=ax1)
            ax1.set_title("Top 20 Neighbourhoods by Number of Listings", fontsize=14)
            ax1.set_xlabel("Number of Listings")
            ax1.set_ylabel("Neighbourhood")
            ax1.grid(axis='x', linestyle='--', alpha=0.7)
            st.pyplot(fig1)

            total_listings = df_clean.shape[0]
            top20_listings = top_neighbourhoods.sum()
            prob_top20 = top20_listings / total_listings * 100
            prob_other = 100 - prob_top20

            # عرض النسبة كنص
            st.markdown(f"### 📊 Percentage of Listings in Top 20 Neighbourhoods: **{prob_top20:.1f}%**")

            # باي تشارت
            pie_labels = ['Top 20 Neighbourhoods', 'Other Neighbourhoods']
            pie_sizes = [prob_top20, prob_other]
            pie_colors = ['#66b3ff', '#ffcc99']

            fig2, ax2 = plt.subplots()
            ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
            ax2.axis('equal')
            st.pyplot(fig2)

            # Insights
            st.markdown("### 💡 Business Insights")
            st.markdown(f"""
            - 🏙️ **{prob_top20:.1f}%** of all listings are located in the Top 20 Neighbourhoods → shows moderate market concentration.
            - 🎯 A listing has a **{prob_top20:.1f}%** chance of being in a high-demand area.
            - 🌍 The remaining **{prob_other:.1f}%** are spread across smaller neighbourhoods → indicates geographic diversity.
            - 💼 **Business Implications**:
                - 📌 Focused marketing & pricing strategies should target Top 20 areas.
                - 🌱 Encourage supply growth in underserved areas to increase reach and balance.
            """)








        with st.expander("🏙️ City Distribution", expanded=True):
            st.subheader("🏙️ Top Cities by Number of Listings")

            top_cities = df_clean['city'].value_counts()

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=top_cities.index, y=top_cities.values, color='purple', ax=ax)
            ax.set_title("Top Cities by Number of Listings", fontsize=14)
            ax.set_xlabel("City")
            ax.set_ylabel("Number of Listings")
            ax.set_xticklabels(top_cities.index, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

            st.markdown("### 💡 Insights from City Distribution")
            st.markdown("""
            - 🏅 **Paris** leads the platform with the highest number of listings (≈ 64K listings) → very active short-term rental market.
            - 🌍 Major global cities such as **New York**, **Sydney**, **Rome**, and **Rio de Janeiro** also show strong presence on the platform.
            - 🏘️ Overall, Airbnb activity is highly concentrated in key urban centers, with top cities driving a large portion of the platform’s inventory.
            - 🔍 This concentration suggests that while Airbnb operates globally, it remains heavily driven by major tourist and business hubs.
            - 💼 **Business Perspective**: Targeting pricing strategies and marketing efforts around these top cities could yield strong impact.
            """)








        with st.expander("🏠 Property Type & Category Distribution", expanded=True):
            st.subheader("🏠 Property Category Distribution")

            # Property Category
            property_category_counts = df_clean['property_category'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=property_category_counts.index, y=property_category_counts.values, color='mediumseagreen', ax=ax1)
            ax1.set_title("Property Category Distribution")
            ax1.set_xlabel("Property Category")
            ax1.set_ylabel("Number of Listings")
            ax1.grid(axis='y', linestyle='--', alpha=0.3)
            for container in ax1.containers:
                ax1.bar_label(container, fmt='%d')
            plt.xticks(rotation=45)
            st.pyplot(fig1)

            # Property Type
            st.subheader("🏡 Top 20 Property Types")
            top_property_types = df_clean['property_type'].value_counts().head(20)
            fig2, ax2 = plt.subplots(figsize=(15, 8))
            sns.barplot(x=top_property_types.index, y=top_property_types.values, color='orchid', ax=ax2)
            ax2.set_title("Top 20 Property Types")
            ax2.set_xlabel("Property Type")
            ax2.set_ylabel("Number of Listings")
            ax2.grid(axis='y', linestyle='-', alpha=0.3)
            for container in ax2.containers:
                ax2.bar_label(container, fmt='%d')
            plt.xticks(rotation=75)
            st.pyplot(fig2)

            # Insights
            st.markdown("### 💡 Insights from Property Categories & Types")
            st.markdown("""
            - 🏢 **Apartments** dominate the Airbnb market, making up the largest share of listings.
            - 🏠 **Houses** and **Hotels** follow, but with noticeably lower volumes.
            - 🛌 **Bed & Breakfast** and **Unique Stays** form niche segments — they are smaller in number but may represent opportunities for specialized targeting or premium offerings.
            """)








        with st.expander("⚡️ Price vs District", expanded=True):

            st.markdown("### 📊 Statistical Price Variation by District")

            district_stats = df_clean.groupby('district')['price_usd'].describe()
            st.dataframe(district_stats)

            sample_values = df_clean["price_usd"].dropna().sample(1000, random_state=42)
            k2_stat, p_norm = stats.normaltest(sample_values)

            if p_norm < 0.05:
                test_name = "Kruskal–Wallis Test"
                test_func = stats.kruskal
                results = test_func(*[df_clean[df_clean["district"] == dist]["price_usd"].dropna().values
                                    for dist in df_clean["district"].unique()])
            else:
                test_name = "ANOVA"
                test_func = stats.f_oneway
                results = test_func(*[df_clean[df_clean["district"] == dist]["price_usd"].dropna().values
                                    for dist in df_clean["district"].unique()])

            st.write(f"**Distribution Test p-value:** {p_norm:.5f}")
            st.write(f"**Test Used:** {test_name}")
            st.write(f"**Test Statistic:** {results.statistic:.2f}")
            st.write(f"**p-value:** {results.pvalue:.5e}")

            st.markdown("### 📉 Average Price per District")

            means = df_clean.groupby('district')['price_usd'].mean().sort_values(ascending=False).reset_index()

            fig, ax = plt.subplots(figsize=(12,6))
            sns.barplot(data=means, x='district', y='price_usd', color='skyblue', ax=ax)
            ax.set_title("Average Price by District", fontsize=14)
            ax.set_xlabel("District", fontsize=12)
            ax.set_ylabel("Average Price (USD)", fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig)

            st.markdown("### 💡 Business Insights")
            st.markdown("""
            ✅ **Observations:**
            - The statistical test (**{}**) confirms a highly significant price difference across districts (**p ≈ {:.1e}**).
            - This means pricing is **not random** and is strongly affected by **location characteristics**.

            🌍 **Why Are Certain Districts More Expensive?**
            - **🏙️ Central or Capital Districts:** e.g., Manhattan, central Paris → high activity & amenities.
            - **🏖️ Coastal Areas:** e.g., Rio de Janeiro → scenic, tourist hotspots.
            - **🗺️ Landmarks Nearby:** e.g., Rome, Brooklyn → attract visitors.
            - **💼 Developed/Wealthy Areas:** e.g., Hong Kong → infrastructure & global demand.

            ⚡️ **Implication for Hosts:**
            - Highlight proximity to famous spots in listing.
            - Justify pricing based on area value.
            - Match pricing to district strength for max bookings 💰.
            """.format(test_name, results.pvalue))







        with st.expander("⚡️ Price vs Neighbourhoods", expanded=True):

            pd.set_option("display.max_rows", 660) 
            neighbourhood_stats = df_clean.groupby("neighbourhood")["price_usd"].describe()
            st.dataframe(neighbourhood_stats)

            # Normality test
            sample_values = df_clean["price_usd"].sample(1000, random_state=42)
            k2_stat, p_norm = stats.normaltest(sample_values)

            if p_norm < 0.05:
                test_name = "Kruskal–Wallis Test"
                test_func = stats.kruskal
                results = test_func(*[
                    df_clean[df_clean["neighbourhood"] == nb]["price_usd"].values
                    for nb in df_clean["neighbourhood"].unique()
                ])
            else:
                test_name = "ANOVA"
                test_func = stats.f_oneway
                results = test_func(*[
                    df_clean[df_clean["neighbourhood"] == nb]["price_usd"].values
                    for nb in df_clean["neighbourhood"].unique()
                ])

            st.markdown(f"""
            **📊 Statistical Test Results:**

            - Distribution Test p-value: `{p_norm:.5f}`
            - Test used: `{test_name}`
            - Result Statistic: `{results.statistic:.2f}`, p-value: `{results.pvalue:.5e}`
            """)

            # Plot
            means = df_clean.groupby('neighbourhood')['price_usd'].mean().sort_values(ascending=False).reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=means.head(30), x='neighbourhood', y='price_usd', color='teal', ax=ax)
            ax.set_title("Top 30 Neighbourhoods by Average Price (USD)", fontsize=14)
            ax.set_xlabel("Neighbourhood", fontsize=12)
            ax.set_ylabel("Average Price (USD)", fontsize=12)
            ax.tick_params(axis='x', rotation=90)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig)
            st.markdown("### 🌍 Why These Top 30 Neighbourhoods Are Expensive")
            st.markdown("""
            These top 30 neighbourhoods stand out due to their premium location, proximity to tourist attractions, well-established services, or exclusive coastal and scenic spots.  
            Here’s a breakdown of why each area commands higher average pricing:
            """)

            # جدول البيانات
            expensive_neighbourhoods = {
                "Neighbourhood": [
                    "São Cristóvão", "Alto da Boa Vista", "Fort Wadsworth", "Joá", "Bangu", "Anchieta", "São Conrado",
                    "Guaratiba", "Pittwater", "Prince's Bay", "Riverdale", "Tribeca", "Ward 78", "Briarwood", "Ward 54",
                    "Flatiron District", "Willowbrook", "Tottenville", "Mosman", "Lagoa", "Ward 62", "Sea Gate", "SoHo",
                    "Theater District", "Greenwood", "Woodrow", "Lower East Side", "Elysiange", "Tsuen Wan", "Fieldston"
                ],
                "Why It's Expensive": [
                    "Rich in heritage, museums, and cultural significance.",
                    "Located in a mountainous area, offering mild climate and upscale villas.",
                    "A coastal area with iconic sights and ocean-view properties.",
                    "An upscale coastal area with luxury villas and tourist appeal.",
                    "Centrally located with convenient access to amenities.",
                    "A quiet, family-friendly area with quality housing and services.",
                    "A premium tourist area with beaches and upscale residential buildings.",
                    "A scenic area with beautiful beaches and eco-friendly tourism spots.",
                    "A picturesque coastal area popular with tourists and vacationers.",
                    "An attractive area for families with ocean access and premium services.",
                    "A green, upscale area located near the Hudson River.",
                    "An iconic neighbourhood in Manhattan, famous for its luxury and culture.",
                    "A well-maintained area with a range of quality accommodation options.",
                    "A quiet, residential neighbourhood with a welcoming community.",
                    "A popular area with a strong focus on family-friendly living.",
                    "A central area with iconic buildings, restaurants, and commerce.",
                    "An area valued for its privacy and quality of life.",
                    "A serene, picturesque neighbourhood ideal for families.",
                    "A premium coastal area popular for its beaches and upscale living.",
                    "A tourist area with lake views and high-end accommodation.",
                    "A well-positioned area with access to premium services.",
                    "An exclusive, private area by the ocean with upscale homes.",
                    "An iconic area known for its restaurants, boutiques, and art scene.",
                    "A lively area with a concentration of theatres, entertainment, and tourism.",
                    "A quiet neighbourhood surrounded by greenery and quality services.",
                    "A family-friendly area with access to essential services and amenities.",
                    "A bustling area rich in culture, shopping, and heritage.",
                    "A tranquil neighbourhood valued for its privacy and quality living.",
                    "A well-connected area with access to daily services and shopping hubs.",
                    "An area celebrated for its green spaces and serene, family-focused living."
                ]
            }
            df_expensive = pd.DataFrame(expensive_neighbourhoods)
            st.dataframe(df_expensive, use_container_width=True)


            # Business Insights
            st.markdown("""
            ✅ **Key Business Insights:**

            - The statistical test confirms **significant price differences** across neighbourhoods.
            - The top areas (e.g., *São Cristóvão*, *Alta da Boa Vista*, *Tribeca*) justify higher prices due to:
                - 🏙️ Prime location & accessibility
                - 🏞️ Tourist appeal & scenic views
                - 💳 Property quality & affluence

            👉 **Pricing Strategy Recommendations:**

            - 🔝 For Premium Areas: Use higher pricing strategies and emphasize quality in listings.
            - 📈 For Affordable Areas: Focus on occupancy rate and highlight value-for-money features.

            🎯 **Final Takeaway:**  
            Aligning your pricing strategy with neighbourhood characteristics is essential to **maximize occupancy and revenue**.
            """)






    with tabs[1]:
        st.subheader("📌 Key Business Questions To Be Answered")



        with st.expander("🔎 What is the relationship between the number of guests accommodated and the nightly price", expanded=True):
            st.header("📊 Visualization")
            df_clean = df_clean[df_clean['accommodates'] > 0]
        
            # عرض تحليل العلاقة بين عدد الضيوف والسعر الليلي
            st.subheader("💵 Price vs Number of Guests")
            accommodates_stats = df_clean.groupby('accommodates')['price_usd'].describe()
            st.dataframe(accommodates_stats)

            # رسم العلاقة بين عدد الضيوف والسعر الليلي
            means = df_clean.groupby('accommodates')['price_usd'].mean().reset_index()
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            sns.barplot(data=means, x='accommodates', y='price_usd', color='teal', ax=ax1)
            for container in ax1.containers:
                ax1.bar_label(container, fmt='%.0f$', fontsize=10)
            ax1.set_title('Average Price by Number of Guests Accommodated', fontsize=14)
            ax1.set_xlabel('Number of Guests', fontsize=12)
            ax1.set_ylabel('Average Nightly Price (USD)', fontsize=12)
            ax1.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig1)

            # رسم توزيع عدد الإعلانات حسب عدد الضيوف
            counts = df_clean['accommodates'].value_counts().sort_index()
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            sns.barplot(x=counts.index, y=counts.values, color='slateblue', ax=ax2)
            for container in ax2.containers:
                ax2.bar_label(container, fmt='%.0f', fontsize=10)
            ax2.set_title('Number of Listings by Number of Guests Accommodated', fontsize=14)
            ax2.set_xlabel('Number of Guests', fontsize=12)
            ax2.set_ylabel('Number of Listings', fontsize=12)
            ax2.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig2)

            # تحليل للفئات الخاصة بـ 15 و 16 ضيف
            subset_15_16 = df_clean[df_clean['accommodates'].isin([15, 16])]
            summary_15_16 = subset_15_16.groupby('accommodates')['price_usd'].agg(['count','mean','median','std','min','max'])
            st.dataframe(summary_15_16)
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=summary_15_16.reset_index(), x='accommodates', y='mean', color='deepskyblue', ax=ax3)
            ax3.set_title('Average Price for Accommodates 15 & 16')
            ax3.set_xlabel('Number of Guests')
            ax3.set_ylabel('Average Nightly Price (USD)')
            for i, row in summary_15_16.reset_index().iterrows():
                ax3.text(i, row['mean'] + 10, f"${row['mean']:.0f}\n(n={row['count']})", ha='center', fontsize=10)
            ax3.set_ylim(0, max(summary_15_16['mean']) + 100)
            ax3.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig3)

            # تحليل متوسط الأسعار حسب نوع الملكية وعدد الضيوف
            grouped = df_clean.groupby(['accommodates', 'property_category'])['price_usd'].mean().reset_index()
            fig4, ax4 = plt.subplots(figsize=(14, 7))
            sns.lineplot(
                data=grouped,
                x='accommodates',
                y='price_usd',
                hue='property_category',
                marker='o',
                ax=ax4
            )
            ax4.set_title('Average Price by Number of Guests and Property Category', fontsize=14)
            ax4.set_xlabel('Number of Guests', fontsize=12)
            ax4.set_ylabel('Average Nightly Price (USD)', fontsize=12)
            ax4.legend(title='Property Category')
            ax4.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig4)
            st.subheader("💼 Property Category Insights")

    # رسم متوسط الأسعار حسب نوع الملكية
            category_stats = df_clean.groupby("property_category")["price_usd"].agg(['mean', 'count']).sort_values(by="mean", ascending=False).reset_index()
            fig5, ax5 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=category_stats, x="property_category", y="mean", color="purple", ax=ax5)

            for index, row in category_stats.iterrows():
                ax5.text(
                    index,
                    row['mean'] + 5,   
                    f"${row['mean']:.0f}\n(n={int(row['count'])})",
                    ha='center',
                    fontsize=9,
                    color='black'
                )

            ax5.set_xticklabels(category_stats['property_category'], rotation=30, ha="right")
            ax5.set_title("Average Price by Property Category", fontsize=14)
            ax5.set_xlabel("Property Category", fontsize=12)
            ax5.set_ylabel("Average Price (USD)", fontsize=12)
            ax5.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig5)

            # رسم Heatmap لعدد الإعلانات حسب عدد الضيوف ونوع الملكية
            group_counts_all = (
                df_clean
                .groupby(['accommodates', 'property_category'])
                .size()
                .reset_index(name='listing_count')
            )

            # عرض DataFrame للعدد
            st.write(group_counts_all)

            pivot_table = group_counts_all.pivot(
                index='accommodates',
                columns='property_category',
                values='listing_count'
            )

            fig6, ax6 = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot_table, annot=True, fmt='g', cmap='Blues', ax=ax6)
            ax6.set_title('Number of Listings by Accommodates and Property Category', fontsize=14)
            ax6.set_xlabel('Property Category', fontsize=12)
            ax6.set_ylabel('Number of Guests Accommodated', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig6)

        # Insights and Analysis:
            st.markdown("""
            ### 🎯 Insight Summary: Relationship between Number of Guests and Nightly Price

            **✅ 1️⃣ Relationship Overview**
            - There is a strong positive relationship between the number of guests accommodated and the average nightly price up to approximately 12–14 guests.
            - The average price starts around **$45 for 1 guest** and increases steadily to about $325 for 12–14 guests.
            - For 15–16 guests, prices slightly decrease. This is likely due to:
            - The limited number of listings in this segment.
            - Potential variations in property quality and market targeting.
            
            **✅ 2️⃣ Market Distribution Insights**
            - The market is heavily concentrated in smaller accommodations (1–4 guests).
            - Over 60% of all listings target this segment.
            - Listings for larger groups (8–16 guests) are much fewer, leading to:
            - Less competition.
            - Less stable average pricing.
            - Higher variability in guest experiences.

            **✅ 3️⃣ Impact of Property Category**
            - Luxury property types (e.g., Villa and Unique Stay) consistently command higher prices across all guest capacities.
            - Standard categories (Apartment, Hotel, House) have more moderate pricing, even as capacity increases.
            - This indicates property type plays a key role in driving premium pricing.
            
            ### 🎯 Business Focus & Targeting Recommendations:

            **✅ Focus Area 1: High-Volume Core Market**
            - Target small group accommodations (2–4 guests).
            - This segment provides the largest volume of listings and demand, ensuring stable occupancy.

            **✅ Focus Area 2: High-Profit Low-Competition Market**
            - Prioritize larger group accommodations (10–14 guests).
            - Despite fewer listings, they offer the highest average nightly rates, making them ideal for maximizing revenue.

            **✅ Focus Area 3: Premium Property Segments**
            - Promote Villas and Unique Stays, which outperform other categories in pricing.
            - Focus marketing on travelers looking for unique or luxury experiences.
            
            **✨ Summary Recommendation:**

            **Marketing campaigns should be split across two main strategies:**
            - **Volume Strategy:** Emphasize affordable accommodations for 2–4 guests to drive consistent bookings.
            - **Premium Strategy:** Focus on 10–14 guest units in Villa and Unique Stay categories for high-margin bookings and targeting affluent customers.
            """)






        with st.expander("🔎 What is the relationship between Instant Bookable status and nightly price?", expanded=True):
            st.header("📊 Instant Bookable Listings Analysis")

            # تحليل العلاقة بين حالة الحجز الفوري والسعر الليلي
            instant_stats = df_clean.groupby("instant_bookable_cleaned")["price_usd"].describe()
            st.dataframe(instant_stats)

            # عرض توزيع الإعلانات حسب حالة الحجز الفوري
            instant_counts = df_clean["instant_bookable_cleaned"].value_counts()
            st.write("Instant Bookable Distribution")
            
            labels = ['Not Instant Bookable', 'Instant Bookable']
            colors = ['#FF9999', '#66B3FF']

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.pie(
                instant_counts,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=140,
                wedgeprops={'edgecolor': 'white'}
            )
            ax1.set_title('Instant Bookable Distribution', fontsize=14)
            ax1.axis('equal')
            st.pyplot(fig1)

            # مقارنة الأسعار بين الحجز الفوري وغير الفوري باستخدام اختبار مان-ويتني
            instant_prices = df_clean[df_clean["instant_bookable_cleaned"] == 1]["price_usd"]
            non_instant_prices = df_clean[df_clean["instant_bookable_cleaned"] == 0]["price_usd"]

            stat, p_value = mannwhitneyu(instant_prices, non_instant_prices, alternative="two-sided")
            st.write(f"Mann–Whitney U Statistic: {stat:.2f}, p-value: {p_value:.5e}")

            # عرض المتوسطات بين الإعلانات الفورية وغير الفورية
            means = df_clean.groupby("instant_bookable_cleaned")["price_usd"].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(10,7))
            sns.barplot(data=means, x="instant_bookable_cleaned", y="price_usd", palette=["#1f77b4","#ff7f0e"], ax=ax2)

            for i, row in means.iterrows():
                ax2.text(i, row["price_usd"] + 5, f"${row['price_usd']:.0f}", ha="center", fontsize=11)

            ax2.set_xticks([0,1])
            ax2.set_xticklabels(["Not Instant Bookable", "Instant Bookable"])
            ax2.set_title("Average Price by Instant Booking Status", fontsize=14)
            ax2.set_xlabel("Instant Bookable Status", fontsize=12)
            ax2.set_ylabel("Average Nightly Price (USD)", fontsize=12)
            ax2.grid(axis="y", linestyle="--", alpha=0.3)
            st.pyplot(fig2)

            # تحليل العلاقة بين الفئة العقارية وحالة الحجز الفوري
            cross_tab = pd.crosstab(df_clean["property_category"], df_clean["instant_bookable_cleaned"])
            cross_tab["Not Instant Bookable %"] = 100 * cross_tab[0] / (cross_tab[0]+cross_tab[1])

            st.dataframe(cross_tab)
            
            # رسم بياني يظهر نسبة الإعلانات غير الفورية حسب الفئة العقارية
            cross_tab_sorted = cross_tab.sort_values("Not Instant Bookable %", ascending=False)
            fig3, ax3 = plt.subplots(figsize=(5,3))
            sns.barplot(
                x=cross_tab_sorted.index,
                y=cross_tab_sorted["Not Instant Bookable %"],
                palette="coolwarm", ax=ax3
            )

            ax3.set_title("Percentage of Not Instant Bookable by Property Category", fontsize=14)
            ax3.set_ylabel("Not Instant Bookable (%)")
            ax3.set_xlabel("Property Category")
            ax3.set_ylim(0, 100)
            ax3.set_xticklabels(cross_tab_sorted.index, rotation=45, ha='right')
            ax3.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig3)

            # Insights and Analysis
            st.markdown("""
            ### 📊 Insight Summary: Instant Bookable Listings Analysis

            **✅ 1️⃣ Overview of Instant Booking Adoption**
            - Only **41.3%** of listings enable Instant Booking, while the majority (**58.7%**) require manual approval.
            - This shows that many hosts still prefer to screen guests before accepting reservations.

            **✅ 2️⃣ Price Differences**
            - The average nightly price for Instant Bookable listings is **$89**, compared to **$102** for Non-Instant Bookable listings.
            - The **Mann–Whitney U Test** confirmed that this difference is statistically significant (**p < 0.00001**).
            - This suggests that Instant Bookable properties are generally priced lower, possibly to:
            - Encourage faster bookings.
            - Compensate for less control over guest selection.

            **✅ 3️⃣ Property Category Patterns**
            - The analysis of property categories revealed that:
                - **Apartments** and **Villas** have the highest proportion of Non-Instant Bookable listings (61% and 63% respectively).
                - **Hotels** and **Bed & Breakfasts** are much more likely to allow Instant Booking.
                - This indicates that individual owners (e.g., apartment and villa hosts) are more cautious, while professional operators (e.g., hotels) rely on Instant Booking as a standard practice.

            **✅ 4️⃣ Market Implications**
            - Listings that do not accept Instant Booking can:
                - Reduce guest conversion rates, as many travelers prefer immediate confirmation.
                - Potentially charge higher prices, leveraging exclusivity and manual vetting.

            The trade-off between control vs. volume is clear:
            - **Instant Bookable = Lower prices, higher turnover**.
            - **Not Instant Bookable = Higher prices, fewer bookings**.

            **🎯 Business Recommendation**
            To increase Instant Booking adoption—especially in high-value categories such as Villas and Houses—the platform should:
            - Offer damage insurance guarantees to hosts.
            - Highlight verified guest profiles and reviews.
            - Provide incentives or discounts on service fees for enabling Instant Booking.

            **✅ Key Takeaway**
            Instant Booking is underutilized among individual property owners. Addressing their trust and risk concerns can drive higher adoption, improve guest satisfaction, and ultimately increase platform revenue.
            """)





        with st.expander("🔎 Which property categories generate the highest revenue potential?", expanded=True):
            st.header("📊 Estimated Revenue by Property Category")

            # Calculating estimated revenue for each property category
            category_stats = df_clean.groupby("property_category")["price_usd"].agg(["count", "mean"]).reset_index()
            category_stats["estimated_revenue"] = category_stats["count"] * category_stats["mean"]

            # Sorting by estimated revenue
            category_stats_sorted = category_stats.sort_values(by="estimated_revenue", ascending=False)

            # Display the table
            st.dataframe(category_stats_sorted)

            # Bar plot of estimated revenue by property category
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=category_stats_sorted, 
                x="property_category", 
                y="estimated_revenue", 
                palette="viridis"
            )
            plt.title("Estimated Revenue Potential by Property Category", fontsize=14)
            plt.xlabel("Property Category", fontsize=12)
            plt.ylabel("Estimated Revenue (USD)", fontsize=12)
            plt.xticks(rotation=30, ha="right")

            # Adding value annotations on top of bars
            for index, row in category_stats_sorted.iterrows():
                value = row["estimated_revenue"]
                count = row["count"]

                if value >= 1_000_000:
                    label = f"${value/1_000_000:.1f}M\n(n={int(count):,})"
                    y_offset = value * 0.02
                    fontsize = 10
                elif value >= 1_000:
                    label = f"${value/1_000:.1f}K\n(n={int(count):,})"
                    y_offset = max(value * 0.05, 20000)  
                    fontsize = 9
                else:
                    label = f"${value:,.0f}\n(n={int(count):,})"
                    y_offset = 50000
                    fontsize = 9

                plt.text(
                    index,
                    value + y_offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold"
                )

            plt.tight_layout()
            st.pyplot(plt)

            # Insight summary in markdown format
            st.markdown("""
            🟢 **Insight Summary: Estimated Revenue by Property Category**

            **✅ 1️⃣ Market Overview**

            - **Apartments dominate the market**:
                - Over **219,000** listings, with an average price around **$92/night**.
                - They generate the highest total estimated revenue (~$20.3M), thanks to the massive volume.
            - **Resorts** have the highest average nightly price (**$375**) but the smallest market size (only 53 listings).
            - **Villas** also maintain a high price point (**$223/night**) but have limited market presence (~1,800 listings).
            - Other categories like **Unique Stays**, **Bed & Breakfasts**, and **Hotels** show moderate prices but relatively smaller listing counts.

            **✅ 2️⃣ Business Implications**

            - The high-revenue volume today depends mainly on **Apartments**.
            - However, **Resorts** and **Villas** represent **high-margin opportunities**:
                - Each booking generates significantly more income than standard apartments.
                - Their low adoption suggests untapped potential.

            **🎯 Recommendations for Strategy**
            
            **✅ Focus Area 1 – Maintain Market Volume**
            - Continue promoting **Apartments** to keep stable booking flow.
            - Offer loyalty programs and promotions to retain volume.

            **✅ Focus Area 2 – Grow High-Profit Segments**
            - Invest in marketing campaigns targeting:
                - **Resorts**: the highest revenue per booking but minimal listings.
                - **Villas**: high-price properties with growth potential.
                - **Unique Stays**: niche experiences that can attract premium customers.

            **✅ Focus Area 3 – Address Adoption Barriers**
            - For property owners hesitant to list high-value accommodations:
                - Provide trust guarantees, insurance coverage, and premium verification.
                - Offer reduced commission or special incentives for early adopters.
                - Highlight successful case studies to demonstrate earning potential.

            **✨ Summary**
            
            The current market relies heavily on apartment bookings, but unlocking the high-price categories (**Resorts**, **Villas**) could significantly increase profitability. Combining volume strategy with targeted expansion into premium listings is key to sustainable growth.
            """)
        with st.expander("🔎 What is the seasonal pattern of new hosts joining and new listings being created over the year?", expanded=True):
            st.header("📊 Seasonal Patterns: New Hosts & Listings by Month")
            
            # إعداد البيانات الشهرية
            df_clean["month_joined"] = df_clean["host_since"].dt.month

            monthly_hosts = df_clean.groupby("month_joined").agg(
                new_hosts=("host_id", "nunique"),
                listings=("listing_id", "count")
            ).reset_index()

            monthly_hosts["month_name"] = monthly_hosts["month_joined"].apply(lambda x: calendar.month_name[int(x)])

            month_order = list(calendar.month_name)[1:]  
            monthly_hosts["month_name"] = pd.Categorical(monthly_hosts["month_name"], categories=month_order, ordered=True)
            monthly_hosts = monthly_hosts.sort_values("month_name")
            
            # عرض الجدول
            st.subheader("Monthly Statistics Overview")
            st.dataframe(monthly_hosts[['month_name', 'new_hosts', 'listings']], use_container_width=True)
            
            # رسم المضيفين الجدد
            st.subheader("📈 New Hosts Joining by Month")
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            
            bars1 = ax1.bar(range(len(monthly_hosts)), monthly_hosts['new_hosts'], 
                            color=plt.cm.Blues(np.linspace(0.4, 1, len(monthly_hosts))))
            
            ax1.set_title("New Hosts Joining by Month", fontsize=16, pad=20)
            ax1.set_xlabel("Month", fontsize=12)
            ax1.set_ylabel("Number of New Hosts", fontsize=12)
            ax1.grid(axis="y", linestyle="--", alpha=0.3)
            
            # تسمية المحور السيني
            ax1.set_xticks(range(len(monthly_hosts)))
            ax1.set_xticklabels(monthly_hosts['month_name'], rotation=45, ha='right')
            
            # إضافة القيم فوق الأعمدة
            for idx, value in enumerate(monthly_hosts['new_hosts']):
                ax1.text(idx, value + value * 0.01, f'{value:,.0f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            
            # رسم الإعلانات الجديدة
            st.subheader("📋 Listings Created by Month")
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            
            bars2 = ax2.bar(range(len(monthly_hosts)), monthly_hosts['listings'], 
                            color=plt.cm.Purples(np.linspace(0.4, 1, len(monthly_hosts))))
            
            ax2.set_title("Listings Created by Month", fontsize=16, pad=20)
            ax2.set_xlabel("Month", fontsize=12)
            ax2.set_ylabel("Number of Listings", fontsize=12)
            ax2.grid(axis="y", linestyle="--", alpha=0.3)
            
            # تسمية المحور السيني
            ax2.set_xticks(range(len(monthly_hosts)))
            ax2.set_xticklabels(monthly_hosts['month_name'], rotation=45, ha='right')
            
            # إضافة القيم فوق الأعمدة
            for idx, value in enumerate(monthly_hosts['listings']):
                ax2.text(idx, value + value * 0.01, f'{value:,.0f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            
            # إحصائيات سريعة
            peak_hosts_month = monthly_hosts.loc[monthly_hosts['new_hosts'].idxmax(), 'month_name']
            peak_hosts_count = monthly_hosts['new_hosts'].max()
            peak_listings_month = monthly_hosts.loc[monthly_hosts['listings'].idxmax(), 'month_name']
            peak_listings_count = monthly_hosts['listings'].max()
            
            lowest_hosts_month = monthly_hosts.loc[monthly_hosts['new_hosts'].idxmin(), 'month_name']
            lowest_hosts_count = monthly_hosts['new_hosts'].min()
            
            # عرض الإحصائيات في أعمدة
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Peak New Hosts Month",
                    value=peak_hosts_month,
                    delta=f"{peak_hosts_count:,} hosts"
                )
            
            with col2:
                st.metric(
                    label="Peak Listings Month", 
                    value=peak_listings_month,
                    delta=f"{peak_listings_count:,} listings"
                )
                
            with col3:
                st.metric(
                    label="Lowest Activity Month",
                    value=lowest_hosts_month,
                    delta=f"{lowest_hosts_count:,} hosts"
                )
            
            # النص التحليلي النهائي
            st.markdown(f"""
            🎯 **Insight: Seasonal Patterns in Host Onboarding and Listings**

            ✅ **1️⃣ Clear Seasonality in Host Activity**
            - There is a significant increase in the number of new hosts between **May and July**.
            - The peak occurs in **{peak_hosts_month}** with **{peak_hosts_count:,} new hosts**.
            - This suggests that many hosts start their activity during summer holidays and vacation season.

            ✅ **2️⃣ Increase in Listings During Summer**
            - The number of new listings rises noticeably in **{peak_listings_month}** with **{peak_listings_count:,} new listings**.
            - This indicates that summer is the most active period for publishing properties.

            ✅ **3️⃣ Stable Activity During the Rest of the Year**
            - Other months show relatively stable figures.
            - The lowest activity is observed in **{lowest_hosts_month}** with only **{lowest_hosts_count:,} new hosts**.

            ✨ **Recommendation** 🎯 **Marketing Focus:**
            - **Intensify marketing campaigns** to encourage property owners to start hosting before the summer peak (March – May).
            - **Offer promotional incentives** and support to attract as many new listings as possible in preparation for high demand.
            - **Target off-season months** (February, September) with special campaigns to balance seasonal fluctuations.
            """)
        with st.expander("🔎 How are prices distributed by room type (Entire Place – Private Room – Shared Room – Hotel Room)?", expanded=True):
            st.header("💰 Price Distribution Analysis by Room Type")
            
            # تجميع إحصائيات نوع الغرفة
            room_stats = df_clean.groupby("room_type").agg(
                count=("listing_id", "count"),
                mean_price=("price_usd", "mean"),
                median_price=("price_usd", "median"),
                std_price=("price_usd", "std")
            ).reset_index()
            
            # ترتيب البيانات حسب متوسط السعر (من الأعلى للأقل)
            room_stats = room_stats.sort_values("mean_price", ascending=False)
            
            # عرض الجدول الإحصائي
            st.subheader("📊 Room Type Statistics Overview")
            
            # تنسيق الجدول للعرض
            room_stats_display = room_stats.copy()
            room_stats_display['mean_price'] = room_stats_display['mean_price'].round(2)
            room_stats_display['median_price'] = room_stats_display['median_price'].round(2)
            room_stats_display['std_price'] = room_stats_display['std_price'].round(2)
            room_stats_display['percentage'] = (room_stats_display['count'] / room_stats_display['count'].sum() * 100).round(1)
            
            # إعادة تسمية الأعمدة للعرض
            room_stats_display.columns = ['Room Type', 'Count', 'Mean Price ($)', 'Median Price ($)', 'Std Dev ($)', 'Percentage (%)']
            
            st.dataframe(room_stats_display, use_container_width=True)
            
            # رسم متوسط الأسعار
            st.subheader("📈 Average Nightly Price by Room Type")
            fig1, ax1 = plt.subplots(figsize=(12, 7))
            
            # استخدام ألوان Set2
            colors = plt.cm.Set2(np.linspace(0, 1, len(room_stats)))
            bars = ax1.bar(range(len(room_stats)), room_stats['mean_price'], color=colors)
            
            # إعداد المحاور والتسميات
            ax1.set_title("Average Nightly Price by Room Type", fontsize=16, pad=20)
            ax1.set_xlabel("Room Type", fontsize=12)
            ax1.set_ylabel("Average Price (USD)", fontsize=12)
            ax1.grid(axis='y', linestyle='--', alpha=0.3)
            
            # تسمية المحور السيني
            ax1.set_xticks(range(len(room_stats)))
            ax1.set_xticklabels(room_stats['room_type'], rotation=15, ha='right')
            
            # إضافة القيم فوق الأعمدة
            for idx, (_, row) in enumerate(room_stats.iterrows()):
                ax1.text(
                    idx,
                    row["mean_price"] + row["mean_price"] * 0.02,
                    f"${row['mean_price']:.0f}\n(n={int(row['count']):,})",
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold'
                )
            
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            
            # رسم توزيع عدد الإعلانات
            st.subheader("📋 Listing Distribution by Room Type")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # رسم دائري لتوزيع الإعلانات
            wedges, texts, autotexts = ax2.pie(
                room_stats['count'], 
                labels=room_stats['room_type'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            
            ax2.set_title("Distribution of Listings by Room Type", fontsize=16, pad=20)
            
            # تحسين النصوص
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
            
            st.pyplot(fig2, use_container_width=True)
            
            # إحصائيات سريعة
            highest_price_type = room_stats.iloc[0]['room_type']
            highest_price_value = room_stats.iloc[0]['mean_price']
            lowest_price_type = room_stats.iloc[-1]['room_type']
            lowest_price_value = room_stats.iloc[-1]['mean_price']
            most_common_type = room_stats.loc[room_stats['count'].idxmax(), 'room_type']
            most_common_count = room_stats['count'].max()
            
            # عرض الإحصائيات في أعمدة
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Highest Priced Room Type",
                    value=highest_price_type,
                    delta=f"${highest_price_value:.0f} avg"
                )
            
            with col2:
                st.metric(
                    label="Most Affordable Room Type", 
                    value=lowest_price_type,
                    delta=f"${lowest_price_value:.0f} avg"
                )
                
            with col3:
                st.metric(
                    label="Most Common Room Type",
                    value=most_common_type,
                    delta=f"{most_common_count:,} listings"
                )
            
            # جدول مقارنة سريع
            st.subheader("⚡ Quick Price Comparison")
            price_comparison = room_stats[['room_type', 'mean_price', 'count']].copy()
            price_comparison['price_range'] = price_comparison['mean_price'].apply(
                lambda x: "Premium (>$100)" if x > 100 
                else "Mid-range ($50-$100)" if x > 50 
                else "Budget (<$50)"
            )
            
            comparison_summary = price_comparison.groupby('price_range').agg(
                room_types=('room_type', lambda x: ', '.join(x)),
                total_listings=('count', 'sum'),
                avg_price_range=('mean_price', lambda x: f"${x.min():.0f} - ${x.max():.0f}")
            ).reset_index()
            
            st.dataframe(comparison_summary, use_container_width=True)
            
            # النص التحليلي النهائي
            st.markdown(f"""
            🎯 **Insight: Room Type Pricing and Distribution**

            ✅ **1️⃣ Premium Pricing Segments**
            - **{highest_price_type}** commands the **highest average nightly price** at **${highest_price_value:.0f}**, positioning it as a premium accommodation type.
            - **Hotel Rooms** and **Entire Places** typically fall in the premium category, targeting guests willing to pay for privacy and amenities.

            ✅ **2️⃣ Budget-Friendly Options**
            - **{lowest_price_type}** offers the **most affordable option** at **${lowest_price_value:.0f}** average, serving budget-conscious travelers.
            - **Private Rooms** provide a middle-ground option for guests seeking privacy at moderate prices.

            ✅ **3️⃣ Market Dominance**
            - **{most_common_type}** represents the **largest share of listings** with **{most_common_count:,} properties** ({(most_common_count/room_stats['count'].sum()*100):.1f}% of total supply).
            - This makes it the **main driver of platform revenue** and guest bookings.

            ✅ **4️⃣ Price Variability**
            - **Hotel Rooms** show **high price variability** (std ~{room_stats[room_stats['room_type']=='Hotel room']['std_price'].iloc[0]:.0f}), suggesting a broad range of property quality and amenities.
            - **Shared Rooms** remain a **small niche market** primarily serving budget-conscious travelers.

            💡 **Recommendations:**

            🎯 **Revenue Optimization:**
            - **Focus marketing efforts** on **{highest_price_type}** and other premium segments to attract higher-value bookings.
            - **Develop clear messaging and segmentation** to target different guest needs (privacy vs. price sensitivity).

            🎯 **Market Expansion:**
            - **Support budget listings** with promotions or guarantees to improve competitiveness in the **Private Room** segment.
            - **Leverage the dominance** of **{most_common_type}** listings to drive volume while exploring premium upselling opportunities.

            🎯 **Strategic Focus:**
            - **Premium Strategy**: Target business travelers and luxury seekers with Entire Places and Hotel Rooms.
            - **Volume Strategy**: Maintain strong supply in the dominant {most_common_type} category.
            - **Budget Strategy**: Enhance competitiveness of Shared and Private Room offerings.
            """)
        with st.expander("🔎 How does the combination of Room Type and Property Category influence average nightly price? \n\nHow are property categories distributed across different cities?", expanded=True):
            st.set_page_config(
                page_title="City Property Category Analysis",
                page_icon="🏠",
                layout="wide"
            )

            # Title
            st.title("🏠 City Property Category Analysis Dashboard")
            st.markdown("---")

            # Sample data creation (replace with your actual df_clean)
            @st.cache_data
            def create_sample_data():
                # Sample data based on your results
                data = {
                    'city': ['Bangkok', 'Bangkok', 'Bangkok', 'Bangkok', 'Bangkok', 'Bangkok', 'Bangkok', 'Bangkok',
                            'Cape Town', 'Cape Town', 'Cape Town', 'Cape Town', 'Cape Town', 'Cape Town', 'Cape Town', 'Cape Town',
                            'Hong Kong', 'Hong Kong', 'Hong Kong', 'Hong Kong', 'Hong Kong', 'Hong Kong', 'Hong Kong', 'Hong Kong',
                            'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul', 'Istanbul',
                            'Mexico City', 'Mexico City', 'Mexico City', 'Mexico City', 'Mexico City', 'Mexico City', 'Mexico City', 'Mexico City',
                            'New York', 'New York', 'New York', 'New York', 'New York', 'New York', 'New York', 'New York',
                            'Paris', 'Paris', 'Paris', 'Paris', 'Paris', 'Paris', 'Paris',
                            'Rio De Janeiro', 'Rio De Janeiro', 'Rio De Janeiro', 'Rio De Janeiro', 'Rio De Janeiro', 'Rio De Janeiro', 'Rio De Janeiro',
                            'Rome', 'Rome', 'Rome', 'Rome', 'Rome', 'Rome', 'Rome',
                            'Sydney', 'Sydney', 'Sydney', 'Sydney', 'Sydney', 'Sydney', 'Sydney', 'Sydney'],
                    'property_category': ['Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Resort', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Resort', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Resort', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Resort', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Resort', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Unique Stay', 'Villa',
                                        'Apartment', 'Bed & Breakfast', 'Hotel', 'House', 'Other', 'Resort', 'Unique Stay', 'Villa'],
                    'listing_count': [13200, 494, 2740, 2587, 152, 9, 37, 136,
                                    9827, 630, 351, 7216, 30, 7, 98, 926,
                                    5687, 144, 724, 460, 22, 1, 39, 8,
                                    17496, 689, 3424, 2417, 57, 123, 295,
                                    15656, 173, 643, 3500, 27, 1, 30, 23,
                                    31305, 42, 807, 4711, 24, 34, 31, 21,
                                    61084, 188, 2436, 844, 31, 24, 16,
                                    23019, 183, 315, 2913, 13, 43, 119,
                                    22300, 2677, 1006, 1239, 100, 60, 252,
                                    21074, 284, 631, 11235, 24, 2, 76, 304]
                }
                
                return pd.DataFrame(data)

            @st.cache_data
            def create_combo_price_data():
                # Sample combo data with property category + room type combinations
                combo_data = {
                    'property_category': ['Villa', 'Villa', 'Villa', 'Resort', 'Resort', 'House', 'House', 'House',
                                        'Unique Stay', 'Unique Stay', 'Hotel', 'Hotel', 'Hotel', 'Apartment', 'Apartment', 
                                        'Apartment', 'Bed & Breakfast', 'Bed & Breakfast', 'Other', 'Other'],
                    'room_type': ['Entire home/apt', 'Private room', 'Shared room', 'Entire home/apt', 'Private room',
                                'Entire home/apt', 'Private room', 'Shared room', 'Entire home/apt', 'Private room',
                                'Entire home/apt', 'Private room', 'Shared room', 'Entire home/apt', 'Private room',
                                'Shared room', 'Entire home/apt', 'Private room', 'Entire home/apt', 'Private room'],
                    'mean_price': [285.5, 180.2, 95.4, 320.8, 225.6, 195.3, 125.7, 78.9, 210.4, 145.8,
                                175.2, 110.5, 65.3, 145.8, 85.4, 45.2, 125.3, 75.8, 98.7, 65.4],
                    'count_listings': [1245, 892, 156, 87, 45, 8756, 3421, 567, 234, 189,
                                    12543, 4567, 1234, 45678, 12345, 3456, 1876, 2345, 789, 456]
                }
                
                combo_df = pd.DataFrame(combo_data)
                combo_df['category_room'] = combo_df['property_category'] + " - " + combo_df['room_type']
                combo_df = combo_df.sort_values(by='mean_price', ascending=False)
                
                return combo_df

            # Load data
            city_category_counts = create_sample_data()
            combo_avg_price = create_combo_price_data()

            # Create pivot table for heatmap
            city_category_pivot = city_category_counts.pivot(
                index='city',
                columns='property_category',
                values='listing_count'
            ).fillna(0)

            # Get top cities for bar chart
            top_cities = city_category_counts.groupby('city')['listing_count'].sum().sort_values(ascending=False).head(10).index.tolist()
            top_cities_data = city_category_counts[city_category_counts['city'].isin(top_cities)]

            # Main Analysis Section with Expanders
            with st.expander("🔎 Property Category + Room Type Price Analysis", expanded=True):
                st.subheader("How does the combination of Room Type and Property Category influence average nightly price?")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create the combo price chart
                    fig, ax = plt.subplots(figsize=(15, 10))
                    
                    bars = ax.bar(
                        range(len(combo_avg_price)),
                        combo_avg_price['mean_price'],
                        color=plt.cm.viridis(np.linspace(0, 1, len(combo_avg_price)))
                    )
                    
                    # Add value labels on bars
                    for i, (_, row) in enumerate(combo_avg_price.iterrows()):
                        ax.text(
                            i,
                            row['mean_price'] + 5,
                            f"${row['mean_price']:.0f}\n(n={int(row['count_listings'])})",
                            ha='center',
                            fontsize=7,
                            fontweight='bold'
                        )
                    
                    ax.set_title("Average Nightly Price by Property Category and Room Type", fontsize=16, fontweight='bold')
                    ax.set_xlabel("Property Category - Room Type", fontsize=12)
                    ax.set_ylabel("Average Nightly Price (USD)", fontsize=12)
                    ax.set_xticks(range(len(combo_avg_price)))
                    ax.set_xticklabels(combo_avg_price['category_room'], rotation=90, ha='right')
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("### 💰 Top Price Combinations")
                    for _, row in combo_avg_price.head(5).iterrows():
                        st.metric(
                            label=row['category_room'],
                            value=f"${row['mean_price']:.0f}",
                            delta=f"{row['count_listings']:,} listings"
                        )
                    
                    st.markdown("### 📊 Price Range Analysis")
                    max_price = combo_avg_price['mean_price'].max()
                    min_price = combo_avg_price['mean_price'].min()
                    avg_price = combo_avg_price['mean_price'].mean()
                    
                    st.metric("Highest Price", f"${max_price:.0f}")
                    st.metric("Lowest Price", f"${min_price:.0f}")
                    st.metric("Average Price", f"${avg_price:.0f}")
                
                # Insights section
                st.markdown("### 💡 Key Insights from Combo Analysis")
                st.markdown("""
                **🏆 Premium Combinations:**
                - **Resort + Entire home/apt**: Commands highest prices due to luxury and privacy
                - **Villa + Entire home/apt**: High-end properties with exclusive access
                - **Unique Stay + Entire home/apt**: Premium for distinctive experiences
                
                **📈 Price Drivers:**
                - **Room Type Impact**: Entire home/apt consistently commands premium over private/shared rooms
                - **Property Category**: Resorts and villas lead in pricing across all room types
                - **Volume vs Price**: Higher-priced combinations tend to have lower listing volumes
                
                **🎯 Strategic Opportunities:**
                - Focus on acquiring high-value property categories (resorts, villas)
                - Encourage hosts to offer entire home/apt options for maximum revenue
                - Target underrepresented high-price combinations for market expansion
                """)

            with st.expander("📊 City Property Category Overview", expanded=True):
                st.subheader("Distribution of Listings by City and Property Category")
                
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                total_listings = city_category_counts['listing_count'].sum()
                total_cities = city_category_counts['city'].nunique()
                total_categories = city_category_counts['property_category'].nunique()
                top_city = city_category_counts.groupby('city')['listing_count'].sum().idxmax()
                
                with col1:
                    st.metric("Total Listings", f"{total_listings:,}")
                with col2:
                    st.metric("Cities Analyzed", total_cities)
                with col3:
                    st.metric("Property Categories", total_categories)
                with col4:
                    st.metric("Top City", top_city)

            with st.expander("🏙️ Top Cities by Property Category", expanded=True):
                st.subheader("Listing Distribution Across Major Cities")
                
                # Create the stacked bar chart
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Create a pivot for the stacked bar chart
                pivot_for_plot = top_cities_data.pivot(index='city', columns='property_category', values='listing_count').fillna(0)
                
                # Plot stacked bar chart
                pivot_for_plot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', figsize=(14, 8))
                
                ax.set_title('Top Cities: Number of Listings per Property Category', fontsize=16, fontweight='bold')
                ax.set_xlabel('City', fontsize=12)
                ax.set_ylabel('Number of Listings', fontsize=12)
                ax.legend(title='Property Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Top performing categories
                st.markdown("### 🏆 Top Performing Property Categories")
                category_totals = city_category_counts.groupby('property_category')['listing_count'].sum().sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    for i, (category, count) in enumerate(category_totals.head(4).items()):
                        st.metric(f"{i+1}. {category}", f"{count:,}")
                
                with col2:
                    for i, (category, count) in enumerate(category_totals.tail(4).items()):
                        st.metric(f"{i+5}. {category}", f"{count:,}")

            with st.expander("🔥 Heatmap: Property Distribution Matrix", expanded=True):
                st.subheader("Comprehensive City vs Property Category Analysis")
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Create heatmap with annotations
                sns.heatmap(
                    city_category_pivot, 
                    annot=True, 
                    fmt='.0f', 
                    cmap='YlGnBu',
                    ax=ax,
                    cbar_kws={'label': 'Number of Listings'}
                )
                
                ax.set_title('Distribution of Listings by Property Category and City', fontsize=16, fontweight='bold')
                ax.set_xlabel('Property Category', fontsize=12)
                ax.set_ylabel('City', fontsize=12)
                
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with st.expander("💡 Key Insights & Analysis", expanded=True):
                st.markdown("""
                ## 🏠 Distribution of Listings by City and Property Category - Insights
                
                ### 📈 **Key Findings:**
                
                **🏢 Apartments Dominate the Market:**
                - In most major cities (especially Paris and New York), apartments are the most prevalent listing type
                - Paris alone has over **61,000 apartment listings**, indicating a highly saturated market
                - This represents the largest share of inventory across all analyzed cities
                
                **🏡 Houses and Villas Show Variation:**
                - The number of houses and villas varies significantly by city
                - Sydney has a high number of house listings (~11,000), while cities like Paris have relatively few
                - This suggests an opportunity to expand the offering of houses and villas in markets where they are underrepresented
                
                **🛏️ Bed & Breakfast and Unique Stays Are Niche Segments:**
                - These property types have a much smaller footprint in all cities
                - Promoting these categories could help attract hosts looking to differentiate their listings
                - Appeals to travelers seeking unique experiences beyond standard accommodations
                
                **🏖️ Resorts Are Extremely Limited:**
                - Resorts are almost nonexistent in many cities
                - This represents an untapped opportunity to target luxury accommodation providers
                - Expansion into the high-value segment of resort stays could be lucrative
                """)

            with st.expander("🎯 Business Implications & Strategy", expanded=True):
                st.markdown("""
                ## 🎯 Business Implications
                
                ### 🔍 **Strategic Opportunities:**
                
                **🎨 Focus on Differentiation:**
                - Rather than competing solely in highly saturated apartment markets
                - Consider strategies to grow supply in underrepresented property categories
                - Target: resorts, villas, unique stays for competitive advantage
                
                **📢 Targeted Campaigns:**
                - Launch targeted marketing campaigns in cities where specific property categories are less common
                - Build unique inventory to attract new segments of travelers
                - Focus on underserved niches for maximum impact
                
                **⚖️ Balanced Approach:**
                - Combine volume growth (apartments) with niche expansion (resorts and unique stays)
                - Build a diverse, resilient supply base
                - Reduce dependency on single property types
                
                ### 📊 **Recommended Actions:**
                
                1. **Market Penetration**: Focus on apartment-heavy markets for volume
                2. **Niche Development**: Develop resort and unique stay categories
                3. **Geographic Expansion**: Target underserved property types by city
                4. **Host Acquisition**: Recruit diverse property owners across categories
                """)

            with st.expander("📋 Detailed Data Tables", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 City Category Breakdown")
                    st.dataframe(
                        city_category_counts.sort_values('listing_count', ascending=False),
                        use_container_width=True,
                        height=400
                    )
                
                with col2:
                    st.subheader("🔄 Pivot Table View")
                    st.dataframe(
                        city_category_pivot,
                        use_container_width=True,
                        height=400
                    )

            with st.expander("📈 Summary Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🏆 Top Cities by Total Listings")
                    city_totals = city_category_counts.groupby('city')['listing_count'].sum().sort_values(ascending=False)
                    for city, total in city_totals.head(5).items():
                        st.write(f"**{city}**: {total:,} listings")
                
                with col2:
                    st.markdown("### 🏠 Property Category Totals")
                    category_totals = city_category_counts.groupby('property_category')['listing_count'].sum().sort_values(ascending=False)
                    for category, total in category_totals.items():
                        st.write(f"**{category}**: {total:,} listings")
                
                with col3:
                    st.markdown("### 📊 Market Share Analysis")
                    total_all = city_category_counts['listing_count'].sum()
                    for category, total in category_totals.head(5).items():
                        percentage = (total / total_all) * 100
                        st.write(f"**{category}**: {percentage:.1f}%")




        with st.expander("🔎 What is the relationship between the minimum stay requirement and the average nightly price?", expanded=True):
            st.set_page_config(
                page_title="District Revenue Analysis",
                page_icon="📊",
                layout="wide"
            )

            # Title
            st.title("📊 District Revenue Analysis Dashboard")
            st.markdown("---")

            # Sample data creation (replace with your actual df_clean)
            @st.cache_data
            def create_sample_data():
                # This is sample data - replace with your actual data loading
                districts = [
                    'Bangkok', 'Brooklyn', 'Ciudad de Mexico', 'Changwat Chachoengsao', 
                    'Changwat Nakhon Pathom', 'Changwat Nonthaburi', 'Changwat Pathum Thani',
                    'Changwat Samut Prakan', 'Estado de Mexico', 'Guangdong Sheng', 'Hong Kong',
                    'Ile-de-France', 'Lazio', 'Manhattan', 'New South Wales', 'Rio de Janeiro',
                    'Shenzhen', 'Staten Island', 'Tekirdag', 'Western Cape'
                ]
                
                # Sample data based on your results
                data = {
                    'district': ['Bangkok'] * 19114 + ['Brooklyn'] * 14466 + ['Ciudad de Mexico'] * 19939 + 
                            ['Changwat Chachoengsao'] * 1 + ['Changwat Nakhon Pathom'] * 4 + 
                            ['Changwat Nonthaburi'] * 1 + ['Changwat Pathum Thani'] * 5 + 
                            ['Changwat Samut Prakan'] * 230 + ['Estado de Mexico'] * 114 + 
                            ['Guangdong Sheng'] * 80 + ['Hong Kong'] * 6962 + ['Ile-de-France'] * 64623 + 
                            ['Lazio'] * 26996 + ['Manhattan'] * 16525 + ['New South Wales'] * 33630 + 
                            ['Rio de Janeiro'] * 26605 + ['Shenzhen'] * 43 + ['Staten Island'] * 289 + 
                            ['Tekirdag'] * 1 + ['Western Cape'] * 19085,
                    'price_usd': []
                }
                
                # Generate prices based on your mean prices
                mean_prices = {
                    'Bangkok': 63.62, 'Brooklyn': 119.06, 'Ciudad de Mexico': 61.01, 
                    'Changwat Chachoengsao': 30.60, 'Changwat Nakhon Pathom': 68.44,
                    'Changwat Nonthaburi': 18.33, 'Changwat Pathum Thani': 52.90,
                    'Changwat Samut Prakan': 62.46, 'Estado de Mexico': 89.16,
                    'Guangdong Sheng': 117.69, 'Hong Kong': 94.86, 'Ile-de-France': 132.76,
                    'Lazio': 124.03, 'Manhattan': 179.71, 'New South Wales': 145.02,
                    'Rio de Janeiro': 135.54, 'Shenzhen': 87.58, 'Staten Island': 109.22,
                    'Tekirdag': 0.88, 'Western Cape': 135.16
                }
                
                for district in data['district']:
                    base_price = mean_prices[district]
                    # Add some random variation
                    price = np.random.normal(base_price, base_price * 0.3)
                    price = max(price, 0.1)  # Ensure positive prices
                    data['price_usd'].append(price)
                
                return pd.DataFrame(data)

            # Load data
            df_clean = create_sample_data()

            # Calculate district statistics
            district_stats = df_clean.groupby('district').agg(
                count=('price_usd', 'count'),
                mean_price=('price_usd', 'mean'),
                total_revenue=('price_usd', 'sum')
            ).reset_index()

            # Top revenue districts
            top_revenue = district_stats.sort_values(by='total_revenue', ascending=False).head(10)

            # Growth opportunities (high price, low volume)
            growth_opportunities = district_stats[
                district_stats['count'] < 500  
            ].sort_values(by='mean_price', ascending=False).head(10)

            # Main Analysis Section with Expanders
            with st.expander("🏆 Top 10 Revenue-Generating Districts", expanded=True):
                st.subheader("Revenue Leaders Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create the revenue chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(
                        range(len(top_revenue)),
                        top_revenue['total_revenue'],
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_revenue)))
                    )
                    
                    ax.set_title("Top 10 Revenue-Generating Districts", fontsize=16, fontweight='bold')
                    ax.set_ylabel("Total Revenue (USD)", fontsize=12)
                    ax.set_xlabel("Districts", fontsize=12)
                    ax.set_xticks(range(len(top_revenue)))
                    ax.set_xticklabels(top_revenue['district'], rotation=30, ha='right')
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    # Add value labels on bars
                    for idx, (_, row) in enumerate(top_revenue.iterrows()):
                        revenue = row['total_revenue']
                        count = row['count']
                        
                        if revenue >= 1e6:
                            revenue_str = f"${revenue/1e6:.1f}M"
                        elif revenue >= 1e3:
                            revenue_str = f"${revenue/1e3:.1f}K"
                        else:
                            revenue_str = f"${revenue:.0f}"
                            
                        ax.text(
                            idx, 
                            revenue + revenue*0.02, 
                            f"{revenue_str}\n(n={count:,})",
                            ha='center',
                            fontsize=9,
                            fontweight='bold'
                        )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("### 📊 Top Revenue Districts")
                    for _, row in top_revenue.head(5).iterrows():
                        revenue = row['total_revenue']
                        if revenue >= 1e6:
                            revenue_str = f"${revenue/1e6:.1f}M"
                        elif revenue >= 1e3:
                            revenue_str = f"${revenue/1e3:.1f}K"
                        else:
                            revenue_str = f"${revenue:.0f}"
                        
                        st.metric(
                            label=row['district'],
                            value=revenue_str,
                            delta=f"{row['count']:,} listings"
                        )

            with st.expander("🚀 Growth Potential Districts (High Price, Low Volume)", expanded=True):
                st.subheader("High-Price, Low-Volume Districts Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create the growth opportunities chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(
                        range(len(growth_opportunities)),
                        growth_opportunities['mean_price'],
                        color=plt.cm.magma(np.linspace(0, 1, len(growth_opportunities)))
                    )
                    
                    ax.set_title("Top 10 High-Price, Low-Volume Districts (Growth Potential)", 
                                fontsize=16, fontweight='bold')
                    ax.set_ylabel("Average Nightly Price (USD)", fontsize=12)
                    ax.set_xlabel("Districts", fontsize=12)
                    ax.set_xticks(range(len(growth_opportunities)))
                    ax.set_xticklabels(growth_opportunities['district'], rotation=30, ha='right')
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    # Add value labels on bars
                    for idx, (_, row) in enumerate(growth_opportunities.iterrows()):
                        price = row['mean_price']
                        count = row['count']
                        
                        price_str = f"${price:.0f}"
                        
                        ax.text(
                            idx, 
                            price + price*0.02, 
                            f"{price_str}\n(n={count:,})",
                            ha='center',
                            fontsize=9,
                            fontweight='bold'
                        )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("### 🎯 Growth Opportunities")
                    for _, row in growth_opportunities.head(5).iterrows():
                        st.metric(
                            label=row['district'],
                            value=f"${row['mean_price']:.0f}",
                            delta=f"{row['count']:,} listings",
                            delta_color="inverse"
                        )

            with st.expander("💡 Insights & Recommendations", expanded=True):
                st.markdown("""
                ### 💡 Insight: Revenue Concentration & Growth Opportunities by District
                
                #### ✅ 1️⃣ Revenue Concentration
                
                The total revenue is highly concentrated in:
                - **Ile-de-France**: The top district with total revenue exceeding $7.8M
                - **New South Wales, Lazio, and Manhattan**: Strong and stable markets with high volumes of listings
                - These areas are considered mature markets where maintaining marketing investment is essential to sustain bookings
                
                #### ✅ 2️⃣ High-Price, Low-Volume Districts
                
                Districts such as **Staten Island, Shenzhen, and Guangdong Sheng** have:
                - High average nightly prices (between $75–$105)
                - Very low listing counts (<500 listings)
                - This combination suggests strong growth potential: increasing the number of listings in these districts could quickly multiply revenue due to higher pricing
                
                #### ✅ 3️⃣ Marketing and Expansion Opportunities
                
                - Low-volume, high-price areas are under-penetrated markets
                - Targeted marketing campaigns and host acquisition initiatives in these regions could unlock significant untapped revenue
                - These districts should be prioritized as growth focus areas
                
                ### 🎯 Recommendation
                
                Split your strategy into two key pillars:
                
                **🔹 Stabilization Pillar**
                - Focus on mature, high-revenue districts (e.g., Ile-de-France, New South Wales) to maintain steady booking flow
                
                **🔹 Growth Pillar**
                - Expand supply in high-price, low-volume districts (e.g., Staten Island, Shenzhen) to accelerate revenue growth and capture underserved demand
                """)

            with st.expander("📊 Detailed Data Tables", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Revenue Districts")
                    st.dataframe(
                        top_revenue[['district', 'count', 'mean_price', 'total_revenue']].round(2),
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("Growth Potential Districts")
                    st.dataframe(
                        growth_opportunities[['district', 'count', 'mean_price', 'total_revenue']].round(2),
                        use_container_width=True
                    )


            st.markdown("---")

    with tabs[2]:
        with st.expander("🎯 Final Strategy & Recommendations for Airbnb Hosts", expanded=True):
            st.markdown("""
            💡 **Where to Focus to Maximize Earnings?**

            ✅ **Luxury Properties (Resorts & Villas)**
            * Although these have very few listings, they achieve the highest nightly rates (600).
            * If you can prepare a villa or resort-style property, it’s a big opportunity to grow revenue with minimal competition.

            ✅ **Entire Place in Apartments or Houses**
            * Represents over 50% of all listings, especially in major cities (Paris, New York).
            * Consistently commands higher nightly prices compared to Private Rooms or Shared Rooms.
            * If you can host an Entire Place, it will generate higher earnings.

            ✅ **Mid-length Stays (4–7 nights)**
            * These stays have the highest average nightly rates (~$126).
            * Focus on attracting mid-length bookings by:
                * Offering weekly discounts.
                * Adding flexible cancellation policies.

            🌍 2️⃣ **Where to Run Marketing Campaigns?**

            ✅ **High Revenue, High-Volume Markets**
            * **Paris (Ile-de-France)** – The top revenue-generating district globally.
            * **New York, Manhattan & Brooklyn** – Stable demand with strong pricing.
            * **New South Wales** – Very large booking volume.
            🎯 *Use digital marketing to target seasonal travelers and drive frequent bookings in these markets.*

            ✅ **High-Price, Low-Volume Markets (Growth Potential)**
            * **Staten Island (New York)**
            * **Shenzhen (China)**
            * **Guangdong Sheng (China)**
            🎯 *Target campaigns here to capture underserved demand willing to pay premium rates.*

            🏷️ 3️⃣ **How to Price and Present Your Listing?**

            ✅ **If it’s an Entire Place →** Price it higher than Private or Shared Rooms.

            ✅ **In high-demand areas (top 10 revenue cities) →** Don’t hesitate to gradually increase prices.

            ✅ **If enabling Instant Booking →** Consider a small discount to encourage fast reservations (but keep control if you are concerned about guest quality).

            ✅ **Offer flexible terms for longer stays (>14 nights) to secure stable bookings.**

            🎯 4️⃣ **Practical Marketing Plan**

            1. 🔹 **Pre-summer (March–May):**
                * Prepare promotional campaigns early as July is the peak for new hosts and listings.
                * Offer Early Bird Discounts.
            2. 🔹 **Peak Season (July–September):**
                * Focus on mid-length and long-term bookings.
                * Add upsell offers for extending stays.
            3. 🔹 **Off-season:**
                * Provide discounts for long stays to attract business or relocation travelers.

            🏆 **Summary for Hosts**
            ✅ **If you want high booking volume:**
            * Focus on Entire Place apartments in large cities (Paris, New York).
            * Use competitive pricing, high-quality photos, and Instant Booking.

            ✅ **If you want maximum profit per booking:**
            * Invest in Villas or Resorts.
            * Target low-competition, high-price markets.

            ✅ **If you want stable occupancy year-round:**
            * Combine flexible stay lengths with attractive pricing.
            * Maintain high reviews and standout experiences.
            """)
